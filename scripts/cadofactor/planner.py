"""
Factor planner + host calibration for cado-nfs.py (v3.2.0-modern, Track E2/E3).

Two related, *non-invasive* features:

- ``--plan`` / ``--plan-json`` (E3, the "factor planner"): given ``N`` (and the
  host's thread count), estimate feasibility, a wall-time *envelope*, the rough
  per-phase split, and a single-machine-vs-cluster recommendation, plus GPU
  triage (``--gpu-prefactor`` / GPU linear algebra) -- then exit *without*
  running anything. It is a planning aid, not a promise.

- ``--autotune`` (E2, the "per-machine tuner"): after the parameter file is
  resolved, calibrate only the **safe scheduling knobs** to the detected host --
  the local client/thread layout (``slaves.nrclients`` + per-client threads) and
  the work-unit *granularity* (``tasks.sieve.qrange``, ``tasks.polyselect.adrange``).
  It deliberately **never touches the number-theoretic smoothness bounds**
  (``lim*``/``lpb*``/``mfb*``/``I``): those determine relation yield and matrix
  structure, so changing them could alter or break the factorization. Only the
  *chunking* of identical work changes, so ``product == N`` is preserved by
  construction.

The wall-time model is anchored on the measured single-desktop numbers in
``BENCHMARKS.md`` (i9-10850K, 20 threads): c60 18.5 s, c70 27.2 s, c80 74.4 s,
c90 197.9 s, plus the documented order-of-magnitude envelope (c100 ~10 min,
c110 ~1 hr). It is interpolated log-linearly in digit count and scaled to the
host's thread count with a simple Amdahl model. NFS wall time has inherent
~+/-15-20 % run-to-run variance (mostly randomized polynomial selection), and
per-core speed differences between hosts are *not* measured -- so the estimate
is reported as a range and clearly labelled a heuristic.

No third-party dependencies; import-safe and side-effect-free.
"""

import math
import os
import re


# -- reference machine (BENCHMARKS.md, CADO-NFS 3.x-modern) -------------------

# Measured end-to-end wall time (seconds), balanced RSA-like semiprimes, on the
# reference box at REF_THREADS logical threads. c60-c90 are measured; c100/c110
# are the documented order-of-magnitude envelope ("~10 min", "~1 hr").
REF_THREADS = 20
_WALL_ANCHORS = {
    60: 18.5,
    70: 27.2,
    80: 74.4,
    90: 197.9,
    100: 600.0,    # documented envelope: "~10 min"
    110: 3600.0,   # documented envelope: "~1 hr"
}

# Representative parallel fraction of the wall (sieving-dominated). Sieving is
# ~45-67 % of CPU and embarrassingly parallel; filtering/polyselect add some.
# Held constant for the estimate (it actually falls with size as linear algebra
# grows -- see BENCHMARKS.md -- which is folded into the wide error band).
_PARALLEL_FRACTION = 0.7

# Rough per-phase share of wall, from the c80-c90 per-phase CPU split
# (BENCHMARKS.md s1). Order-of-magnitude only; used to colour the plan.
_PHASE_SHARE = [
    ("polynomial selection", 0.06),
    ("lattice sieving",      0.62),
    ("filtering",            0.12),
    ("linear algebra",       0.14),
    ("square root",          0.06),
]

# Run-to-run variance band reported around the central estimate.
_VARIANCE = 0.20


def digits_of(n):
    """Decimal digit count of |n|.

    >>> digits_of(10**59)
    60
    >>> digits_of(-(10**89))
    90
    """
    return len(str(abs(int(n))))


def estimate_walltime(digits, threads=REF_THREADS, host_speed=1.0):
    """Estimate the central wall-time (seconds) to factor a balanced semiprime of
    ``digits`` decimal digits on a host with ``threads`` logical threads, each
    ``host_speed``x the reference per-core speed (1.0 = reference class).

    Anchored on BENCHMARKS.md, log-linear in digit count, Amdahl-scaled to the
    thread count. Returns a float; callers apply the variance band.

    >>> # reproduces the reference anchors at the reference thread count
    >>> round(estimate_walltime(60))
    18
    >>> round(estimate_walltime(90))
    198
    >>> # halving the cores lengthens it, but sub-linearly (Amdahl)
    >>> estimate_walltime(90, threads=10) > estimate_walltime(90, threads=20)
    True
    >>> estimate_walltime(90, threads=10) < 2 * estimate_walltime(90, threads=20)
    True
    >>> # a faster per-core host shortens it
    >>> estimate_walltime(90, host_speed=2.0) < estimate_walltime(90)
    True
    """
    anchors = sorted(_WALL_ANCHORS)
    d = float(digits)
    # log-linear interpolation / extrapolation in digit count
    if d <= anchors[0]:
        lo, hi = anchors[0], anchors[1]
    elif d >= anchors[-1]:
        lo, hi = anchors[-2], anchors[-1]
    else:
        lo = max(a for a in anchors if a <= d)
        hi = min(a for a in anchors if a >= d)
        if lo == hi:
            ref_wall = _WALL_ANCHORS[lo]
            return _amdahl_scale(ref_wall, threads, host_speed)
    wlo = math.log(_WALL_ANCHORS[lo])
    whi = math.log(_WALL_ANCHORS[hi])
    t = (d - lo) / (hi - lo)
    ref_wall = math.exp(wlo + t * (whi - wlo))
    return _amdahl_scale(ref_wall, threads, host_speed)


def _amdahl_scale(ref_wall, threads, host_speed):
    """Scale a reference-machine wall time (REF_THREADS threads, speed 1.0) to a
    host with ``threads`` threads at ``host_speed`` per-core speed. Only the
    parallel fraction benefits from extra threads; per-core speed scales both."""
    threads = max(1, int(threads))
    host_speed = max(1e-6, float(host_speed))
    p = _PARALLEL_FRACTION
    serial = (1.0 - p) * ref_wall
    parallel = p * ref_wall * REF_THREADS
    return (serial + parallel / threads) / host_speed


def _fmt_duration(seconds):
    """Human-friendly duration.

    >>> _fmt_duration(8)
    '8 s'
    >>> _fmt_duration(95)
    '1.6 min'
    >>> _fmt_duration(4200)
    '1.2 h'
    >>> _fmt_duration(180000)
    '2.1 days'
    """
    s = float(seconds)
    if s < 90:
        return "%.0f s" % s
    if s < 3600:
        return "%.1f min" % (s / 60.0)
    if s < 172800:
        return "%.1f h" % (s / 3600.0)
    return "%.1f days" % (s / 86400.0)


def detect_threads():
    """Logical CPU count for this host (honours the NCPUS_FAKE override used by
    the test-suite), falling back to 1."""
    fake = os.environ.get("NCPUS_FAKE")
    if fake:
        try:
            return max(1, int(fake))
        except ValueError:
            pass
    return os.cpu_count() or 1


def detect_gpu():
    """Best-effort: is an NVIDIA GPU usably present? Checks /proc and the device
    nodes (cheap, no subprocess); returns True/False. Never raises."""
    try:
        if any(os.path.exists(p) for p in
               ("/dev/nvidia0", "/dev/nvidiactl")):
            return True
        proc = "/proc/driver/nvidia/gpus"
        if os.path.isdir(proc) and os.listdir(proc):
            return True
    except OSError:
        pass
    return False


def make_plan(n=None, digits=None, threads=None, host_speed=1.0, gpu=None,
              gpu_build=False):
    """Build a structured factoring plan. Provide either ``n`` or ``digits``.

    Returns a dict (JSON-serialisable). ``gpu`` = a GPU is present on the host;
    ``gpu_build`` = this CADO build has the GPU front-end compiled in.

    >>> p = make_plan(digits=90, threads=20, gpu=False)
    >>> p["digits"], p["feasibility"]
    (90, 'supported')
    >>> p["recommend_cluster"]
    False
    >>> make_plan(digits=45)["feasibility"]
    'too_small'
    >>> make_plan(digits=140, threads=20)["recommend_cluster"]
    True
    """
    if digits is None:
        if n is None:
            raise ValueError("make_plan needs n or digits")
        digits = digits_of(n)
    if threads is None:
        threads = detect_threads()

    central = estimate_walltime(digits, threads=threads, host_speed=host_speed)
    low = central * (1.0 - _VARIANCE)
    high = central * (1.0 + _VARIANCE)

    if digits < 60:
        feasibility = "too_small"
    elif digits <= 110:
        feasibility = "supported"
    elif digits <= 129:
        feasibility = "large"
    else:
        feasibility = "very_large"

    # cluster recommendation keyed off the central estimate + size
    recommend_cluster = (central > 86400.0) or (digits >= 130)
    cluster_helpful = (central > 3600.0) and not recommend_cluster

    phases = [(name, central * share) for name, share in _PHASE_SHARE]

    # GPU triage
    gpu_notes = []
    if digits >= 60:
        gpu_notes.append(
            "Try --gpu-prefactor first: a batched-GPU-ECM pass can strip a "
            "small/medium factor (<= ~35-40 digits) and skip NFS entirely "
            "(48x/25x/10x the full CPU at 128/256/512-bit; BENCHMARKS s3).")
    if digits >= 100:
        gpu_notes.append(
            "For the linear-algebra phase at this size, the GPU BWC backend "
            "(tasks.linalg.bwc.mm_impl=gpu, CADO_GPU_VECRESIDENT=1) is ~4.5x "
            "the tuned CPU backend and grows with N (BENCHMARKS s4).")
    if (gpu is None or gpu) and not gpu_build:
        gpu_notes.append(
            "Note: this CADO build has no GPU front-end compiled in; set "
            "-DENABLE_GPU=ON in local.sh and rebuild to use the above.")
    if gpu is False:
        gpu_notes.append(
            "No NVIDIA GPU detected on this host; GPU options are unavailable.")

    return {
        "schema": "cado-nfs-plan/1",
        "digits": digits,
        "bits": int(round(digits * math.log2(10))) if digits else 0,
        "threads": int(threads),
        "host_speed": float(host_speed),
        "feasibility": feasibility,
        "wall_seconds": {"low": low, "central": central, "high": high},
        "phases": [{"name": nm, "seconds": sec} for nm, sec in phases],
        "recommend_cluster": bool(recommend_cluster),
        "cluster_helpful": bool(cluster_helpful),
        "gpu_present": gpu,
        "gpu_build": bool(gpu_build),
        "gpu_notes": gpu_notes,
    }


_FEASIBILITY_TEXT = {
    "too_small": "TOO SMALL for NFS (< 60 digits). Use ECM / P-1 / P+1 / "
                 "trial division instead -- NFS is not the right tool below "
                 "~60 digits.",
    "supported": "supported -- well within single-desktop range.",
    "large":     "large -- feasible on this desktop but slow; a cluster or the "
                 "GPU options below help a lot.",
    "very_large": "VERY LARGE (>= 130 digits) -- distributed mode strongly "
                  "recommended; the single-machine estimate is unreliable here.",
}


def format_plan(plan):
    """Render a plan dict as a human-readable text block (str)."""
    d = plan["digits"]
    w = plan["wall_seconds"]
    out = []
    out.append("# cado-nfs factor planner (estimate -- not a guarantee)")
    out.append("")
    out.append("  input size      : %d digits (~%d bits)" % (d, plan["bits"]))
    out.append("  this host       : %d threads%s" % (
        plan["threads"],
        "" if plan["host_speed"] == 1.0
        else ", per-core speed x%.2f (assumed)" % plan["host_speed"]))
    out.append("  feasibility     : %s" % _FEASIBILITY_TEXT.get(
        plan["feasibility"], plan["feasibility"]))
    out.append("")
    if plan["feasibility"] != "too_small":
        out.append("  estimated wall  : %s  (range %s - %s, +/-%d%% NFS variance)"
                   % (_fmt_duration(w["central"]),
                      _fmt_duration(w["low"]), _fmt_duration(w["high"]),
                      int(_VARIANCE * 100)))
        out.append("  rough per-phase :")
        for ph in plan["phases"]:
            out.append("      %-20s ~%s" % (ph["name"],
                                            _fmt_duration(ph["seconds"])))
        out.append("")
        if plan["recommend_cluster"]:
            out.append("  strategy        : DISTRIBUTED recommended -- this is a "
                       "multi-day single-machine job. See scripts/cluster-launch.sh "
                       "and README (distributed mode).")
        elif plan["cluster_helpful"]:
            out.append("  strategy        : single machine works, but a cluster "
                       "would cut wall time substantially (sieving is the "
                       "embarrassingly-parallel ~60%). See scripts/cluster-launch.sh.")
        else:
            out.append("  strategy        : single machine is fine.")
        if plan["gpu_notes"]:
            out.append("")
            out.append("  GPU triage      :")
            for note in plan["gpu_notes"]:
                out.append("      - " + note)
    out.append("")
    out.append("  Caveats: wall time has ~+/-15-20% run-to-run variance "
               "(randomized polyselect); per-core speed differences between "
               "hosts are not measured (clock/IPC/cache can shift this by a "
               "large factor); >=c110 figures are an order-of-magnitude "
               "envelope. Calibrate with bench/las-microbench.sh on your box.")
    return "\n".join(out)


# -- E2: safe per-host scheduling calibration --------------------------------

def host_layout(threads=None, sieve_threads=2):
    """Recommend a *local* client/thread layout for a single-machine run on a
    host with ``threads`` logical threads: a number of clients each running
    ``per_client`` threads, with nrclients*per_client ~= threads. ``sieve_threads``
    is the preferred per-client thread count (2 favours sieving locality), capped
    so we never produce zero clients.

    Pure scheduling -- this changes how identical work is distributed, never the
    work itself. Returns (nrclients, per_client).

    >>> host_layout(threads=20)
    (10, 2)
    >>> host_layout(threads=8)
    (4, 2)
    >>> host_layout(threads=3)
    (3, 1)
    >>> host_layout(threads=1)
    (1, 1)
    """
    if threads is None:
        threads = detect_threads()
    threads = max(1, int(threads))
    per = max(1, int(sieve_threads))
    if threads < 2 * per:
        # too few threads to give each client `per`; fall back to 1 thread/client
        per = 1
    nrclients = max(1, threads // per)
    return nrclients, per


def granularity_factor(threads=None):
    """A bounded multiplier for work-unit granularity (qrange / adrange) on this
    host, relative to the reference's REF_THREADS. More threads -> larger
    work-units (less orchestration overhead); fewer -> finer (better balancing).
    Clamped to [0.5, 4.0] so it can never explode a preset's chunk size.

    >>> granularity_factor(threads=20)
    1.0
    >>> granularity_factor(threads=80)
    2.0
    >>> granularity_factor(threads=2) >= 0.5
    True
    >>> granularity_factor(threads=1000) <= 4.0
    True
    """
    if threads is None:
        threads = detect_threads()
    threads = max(1, int(threads))
    raw = (threads / float(REF_THREADS)) ** 0.5
    return round(min(4.0, max(0.5, raw)), 2)


def autotune_overrides(params_get, threads=None):
    """Compute the safe scheduling overrides for ``--autotune``. ``params_get`` is
    a callable ``key -> value-or-None`` used to read current (preset) values so we
    only *scale* an existing granularity rather than invent one. Returns a list of
    ``(key, value, reason)`` tuples (strings) the caller applies + reports.

    Number-theoretic bounds are deliberately absent -- only client layout and
    work-unit granularity are produced here.
    """
    if threads is None:
        threads = detect_threads()
    overrides = []

    nrclients, per = host_layout(threads)
    overrides.append(("slaves.nrclients", str(nrclients),
                      "%d local clients x %d threads ~= %d logical cores"
                      % (nrclients, per, threads)))

    gf = granularity_factor(threads)
    if gf != 1.0:
        for key in ("tasks.sieve.qrange", "tasks.polyselect.adrange"):
            cur = params_get(key)
            scaled = _scale_int(cur, gf)
            if scaled is not None:
                overrides.append((key, scaled,
                                  "x%.2f host granularity (was %s)" % (gf, cur)))
    return overrides


def _scale_int(value, factor):
    """Scale a stringy integer parameter by ``factor``, returning a string, or
    None if it isn't a plain integer.

    >>> _scale_int("100000", 2.0)
    '200000'
    >>> _scale_int(None, 2.0) is None
    True
    >>> _scale_int("not-an-int", 2.0) is None
    True
    """
    if value is None:
        return None
    s = str(value).strip()
    if not re.fullmatch(r"\d+", s):
        return None
    return str(int(round(int(s) * factor)))
