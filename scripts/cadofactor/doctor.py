"""Preflight "doctor" for cado-nfs.py (Roadmap E5, v3.3.0-modern).

A side-effect-free environment + resource feasibility check. Given (optionally) the
number N to factor, it inspects the build, the CPU/SIMD capabilities, the GPU, free
RAM/disk, the Python environment and any cluster schedulers, then prints a report
and a single GO / GO-WITH-WARNINGS / NO-GO verdict. Nothing is written, no network
call is made, and no computation is started -- this only *reads* the environment.

Run it via ``cado-nfs.py --doctor [N]`` (or ``--doctor-json`` for machine output).
See docs/usability-v330.md. The heavy lifting (feasibility, wall-time envelope, GPU
triage) is reused from :mod:`cadofactor.planner`; this module adds the host probes.
"""

import os
import shutil
import subprocess
import tempfile

from cadofactor import planner

# Core pipeline binaries that a real factorization needs; paths are relative to
# the build/<hostname> directory (pathdict["lib"]). If any is missing the build is
# incomplete and we return NO-GO.
CORE_BINARIES = [
    ("polyselect/polyselect", "polynomial selection"),
    ("sieve/makefb", "factor base"),
    ("sieve/las", "lattice siever"),
    ("filter/purge", "filtering (purge)"),
    ("filter/merge", "filtering (merge)"),
    ("linalg/bwc/bwc.pl", "linear algebra (BWC)"),
    ("sqrt/sqrt", "algebraic square root"),
]

# Rough, deliberately-coarse tiers for the dominant memory/disk consumer (the BWC
# matrix at the linear-algebra phase, and the relation files on disk). Honest
# guidance, not a precise model -- requirements vary with parameters.
def _resource_tier(digits):
    """(ram_gb, disk_gb, note) rough lower bounds for a balanced semiprime.

    >>> _resource_tier(85)[0]
    2
    >>> _resource_tier(125)[0]
    24
    >>> _resource_tier(160)[2].startswith('cluster')
    True
    """
    if digits < 100:
        return 2, 5, "comfortable on a desktop"
    if digits < 120:
        return 8, 20, "desktop-feasible; watch RAM at the linalg phase"
    if digits < 140:
        return 24, 60, "large; a cluster and the GPU options help a lot"
    return 64, 200, "cluster territory; single-machine numbers are unreliable"


STATUS_RANK = {"ok": 0, "info": 0, "warn": 1, "fail": 2}


def _verdict(checks):
    """Overall verdict from the worst check status.

    >>> _verdict([{"status": "ok"}, {"status": "info"}])
    'GO'
    >>> _verdict([{"status": "ok"}, {"status": "warn"}])
    'GO_WITH_WARNINGS'
    >>> _verdict([{"status": "warn"}, {"status": "fail"}])
    'NO_GO'
    """
    worst = max((STATUS_RANK.get(c["status"], 0) for c in checks), default=0)
    return {0: "GO", 1: "GO_WITH_WARNINGS", 2: "NO_GO"}[worst]


def _check(name, status, detail):
    return {"name": name, "status": status, "detail": detail}


def _read_meminfo():
    """(total_gb, available_gb) from /proc/meminfo, or (None, None)."""
    try:
        vals = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                vals[k.strip()] = int(rest.strip().split()[0])  # kB
        tot = vals.get("MemTotal")
        avail = vals.get("MemAvailable", vals.get("MemFree"))

        def gb(kb):
            return round(kb / 1024.0 / 1024.0, 1) if kb else None
        return gb(tot), gb(avail)
    except (OSError, ValueError, IndexError):
        return None, None


def _cpu_flags():
    """Set of interesting x86 SIMD flags present (avx2/avx512f/avx512ifma/...)."""
    want = {"avx2", "avx512f", "avx512dq", "avx512ifma", "vpclmulqdq"}
    found = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    found = want & set(line.split())
                    break
    except OSError:
        pass
    return found


def _gpu_info():
    """Best-effort NVIDIA device summary via nvidia-smi (read-only). None if absent."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8)
        line = out.stdout.strip().splitlines()
        return line[0].strip() if line else None
    except (OSError, subprocess.SubprocessError):
        return None


def _schedulers():
    """Which cluster/distribution tools are on PATH (advisory)."""
    return [c for c in ("ssh", "sbatch", "sinfo", "srun", "qsub", "mpirun")
            if shutil.which(c)]


def run_doctor(n=None, digits=None, threads=None, host_speed=1.0,
               gpu_present=None, gpu_build=False, bindir=None, workdir=None):
    """Build the structured doctor report (a JSON-serialisable dict)."""
    if digits is None and n:
        digits = planner.digits_of(n)
    if threads is None:
        threads = planner.detect_threads()
    if gpu_present is None:
        gpu_present = planner.detect_gpu()

    checks = []

    # --- build completeness -------------------------------------------------
    if not bindir or not os.path.isdir(bindir):
        checks.append(_check("build", "fail",
                             "build directory not found (run `make` first)"))
    else:
        missing = [(p, what) for p, what in CORE_BINARIES
                   if not os.path.exists(os.path.join(bindir, p))]
        if missing:
            checks.append(_check(
                "build", "fail",
                "missing core binaries: " +
                ", ".join("%s (%s)" % (p, w) for p, w in missing) +
                " -- rebuild with `make`"))
        else:
            checks.append(_check("build", "ok",
                                 "all %d core pipeline binaries present in %s"
                                 % (len(CORE_BINARIES), bindir)))

    # --- python environment -------------------------------------------------
    py_missing = []
    for mod in ("sqlite3",):
        try:
            __import__(mod)
        except ImportError:
            py_missing.append(mod)
    server_missing = []
    for mod in ("flask", "requests"):
        try:
            __import__(mod)
        except ImportError:
            server_missing.append(mod)
    if py_missing:
        checks.append(_check("python", "fail",
                             "missing required module(s): " + ", ".join(py_missing)))
    elif server_missing:
        checks.append(_check(
            "python", "warn",
            "sqlite3 ok, but " + ", ".join(server_missing) +
            " missing -- needed for the work-unit server / distributed mode "
            "(run scripts/setup-venv.sh)"))
    else:
        checks.append(_check("python", "ok",
                             "sqlite3 + flask + requests importable"))

    # --- CPU / SIMD ---------------------------------------------------------
    flags = _cpu_flags()
    simd = "AVX-512" if "avx512f" in flags else ("AVX2" if "avx2" in flags
                                                 else "baseline")
    extra = []
    if "avx512ifma" in flags:
        extra.append("IFMA")
    if "vpclmulqdq" in flags:
        extra.append("VPCLMULQDQ")
    detail = "%d logical threads, %s%s" % (
        threads, simd, (" + " + "/".join(extra)) if extra else "")
    if "avx512f" not in flags and "avx2" in flags:
        detail += " (AVX-512 kernels are SDE-validated only on this CPU)"
    checks.append(_check("cpu", "ok", detail))

    # --- memory -------------------------------------------------------------
    tot_gb, avail_gb = _read_meminfo()
    need_ram = need_disk = res_note = None
    if digits:
        need_ram, need_disk, res_note = _resource_tier(digits)
    if avail_gb is None:
        checks.append(_check("memory", "info", "could not read /proc/meminfo"))
    elif need_ram and avail_gb < need_ram:
        checks.append(_check(
            "memory", "warn",
            "%.1f GiB available; ~%d GiB recommended for %d digits (%s)"
            % (avail_gb, need_ram, digits, res_note)))
    else:
        msg = "%.1f GiB available" % avail_gb
        if need_ram:
            msg += " (>= ~%d GiB recommended for %d digits)" % (need_ram, digits)
        checks.append(_check("memory", "ok", msg))

    # --- disk ---------------------------------------------------------------
    probe_dir = workdir or tempfile.gettempdir()
    try:
        free_gb = round(shutil.disk_usage(probe_dir).free / 1024.0**3, 1)
        if need_disk and free_gb < need_disk:
            checks.append(_check(
                "disk", "warn",
                "%.1f GiB free in %s; ~%d GiB recommended for %d digits"
                % (free_gb, probe_dir, need_disk, digits)))
        else:
            msg = "%.1f GiB free in %s" % (free_gb, probe_dir)
            if need_disk:
                msg += " (>= ~%d GiB recommended)" % need_disk
            checks.append(_check("disk", "ok", msg))
    except OSError:
        checks.append(_check("disk", "info",
                             "could not stat %s" % probe_dir))

    # --- GPU ----------------------------------------------------------------
    info = _gpu_info()
    if gpu_present and gpu_build:
        checks.append(_check("gpu", "ok",
                             "GPU front-end built and device present" +
                             (": " + info if info else "")))
    elif gpu_present and not gpu_build:
        checks.append(_check(
            "gpu", "warn",
            "device present" + ((": " + info) if info else "") +
            ", but this build has no GPU front-end -- set -DENABLE_GPU=ON in "
            "local.sh and rebuild to use --gpu-prefactor / GPU linalg"))
    elif not gpu_present and gpu_build:
        checks.append(_check("gpu", "info",
                             "GPU front-end built but no NVIDIA device detected"))
    else:
        checks.append(_check("gpu", "info",
                             "no GPU front-end / device -- CPU-only (fine)"))

    # --- schedulers (advisory) ---------------------------------------------
    sched = _schedulers()
    checks.append(_check("cluster", "info",
                         ("distribution tools on PATH: " + ", ".join(sched))
                         if sched else "no ssh/Slurm/PBS tools on PATH "
                         "(single-machine only)"))

    # --- plan (feasibility + wall-time envelope), if we know N --------------
    plan = None
    if digits:
        plan = planner.make_plan(digits=digits, threads=threads,
                                 host_speed=host_speed, gpu=gpu_present,
                                 gpu_build=gpu_build)
        if plan["feasibility"] == "too_small":
            checks.append(_check("size", "fail",
                                 "N has %d digits -- below NFS's ~60-digit floor; "
                                 "use ECM/P-1/P+1 instead" % digits))
        elif plan["feasibility"] in ("large", "very_large"):
            checks.append(_check("size", "warn",
                                 "%d digits -- %s" % (digits,
                                 planner._FEASIBILITY_TEXT[plan["feasibility"]])))
        else:
            checks.append(_check("size", "ok", "%d digits -- %s"
                                 % (digits,
                                    planner._FEASIBILITY_TEXT[plan["feasibility"]])))

    return {
        "schema": "cado-nfs-doctor/1",
        "digits": digits,
        "threads": int(threads),
        "verdict": _verdict(checks),
        "checks": checks,
        "plan": plan,
    }


_GLYPH = {"ok": "[ OK ]", "info": "[info]", "warn": "[WARN]", "fail": "[FAIL]"}
_VERDICT_TEXT = {
    "GO": "GO -- the environment looks ready.",
    "GO_WITH_WARNINGS": "GO (with warnings) -- usable, but review the [WARN] "
                        "lines above.",
    "NO_GO": "NO-GO -- fix the [FAIL] line(s) above before running.",
}


def format_doctor(report):
    """Render a doctor report dict as a human-readable text block (str)."""
    lines = ["cado-nfs doctor -- preflight check", "=" * 36]
    for c in report["checks"]:
        lines.append("%s %-8s %s" % (_GLYPH.get(c["status"], "[?]"),
                                     c["name"] + ":", c["detail"]))
    if report.get("plan"):
        w = report["plan"]["wall_seconds"]
        lines.append("")
        lines.append("Estimated wall time (%d threads): ~%s (%s - %s)"
                     % (report["threads"],
                        planner._fmt_duration(w["central"]),
                        planner._fmt_duration(w["low"]),
                        planner._fmt_duration(w["high"])))
        for note in report["plan"].get("gpu_notes", []):
            lines.append("  * " + note)
    lines.append("")
    lines.append(_VERDICT_TEXT.get(report["verdict"], report["verdict"]))
    return "\n".join(lines)
