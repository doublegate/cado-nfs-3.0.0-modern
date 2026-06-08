"""
Interactive parameter wizard for cado-nfs.py (v3.4.0-modern, Track E12).

``cado-nfs.py --wizard`` walks a new operator through the few decisions that
actually matter -- the number (or its size), how many threads to use, whether to
engage the GPU pre-factoring front-end, where to send a completion notification,
and single-machine vs cluster -- and prints a ready-to-run command line plus a
short rationale, reusing the planner's feasibility / cluster / GPU triage so the
advice matches ``--plan``.

The decision logic is a pure, side-effect-free function (:func:`recommend`) so it
is doctested; the interactive prompt loop (:func:`run_wizard`) is a thin shell
around it. Nothing here changes a number-theoretic bound -- the wizard only
assembles flags the user could have typed by hand.

>>> rec = recommend(digits=90, threads=20, gpu=True, gpu_build=True,
...                  notify="ntfy=my-topic")
>>> rec["feasibility"]
'supported'
>>> rec["command"]
'cado-nfs.py <N> -t 20 --gpu-prefactor --notify ntfy=my-topic'
>>> rec["cluster"]
False
>>> # a big input flips the cluster recommendation and drops local-only flags
>>> big = recommend(digits=140, threads=20, gpu=False, gpu_build=False)
>>> big["cluster"]
True
>>> "cluster-launch.sh" in big["notes"][-1]
True
>>> # too-small inputs are refused with guidance, no command
>>> recommend(digits=40, threads=4, gpu=False, gpu_build=False)["command"] is None
True
"""

from cadofactor import planner


def recommend(n=None, digits=None, threads=None, gpu=False, gpu_build=False,
              notify=None):
    """Assemble a recommended command line + rationale from the answers. Returns
    a dict ``{command, feasibility, cluster, gpu_prefactor, notes}``; ``command``
    is None when the input is out of CADO's supported range."""
    plan = planner.make_plan(n=n, digits=digits, threads=threads, gpu=gpu,
                             gpu_build=gpu_build)
    digits = plan["digits"]
    notes = []

    if plan["feasibility"] == "too_small":
        notes.append("CADO-NFS targets numbers > 59 digits; for %d digits use a "
                     "dedicated small-number tool (e.g. GMP-ECM, msieve SIQS, or "
                     "Pari/GP) -- NFS would be slower and is unsupported." % digits)
        return {"command": None, "feasibility": "too_small", "cluster": False,
                "gpu_prefactor": False, "notes": notes}
    if plan["feasibility"] == "very_large":
        notes.append("At %d digits this is a record-class computation; a single "
                     "machine is not enough -- see the cluster guidance below." % digits)

    parts = ["cado-nfs.py", "<N>"]
    if threads:
        parts += ["-t", str(threads)]

    use_gpu_pre = bool(gpu and gpu_build)
    if use_gpu_pre:
        parts.append("--gpu-prefactor")
        notes.append("A GPU is present and this build has the GPU front-end: "
                     "--gpu-prefactor strips a small/medium factor before NFS "
                     "(a separate stage with no Amdahl ceiling).")
    elif gpu and not gpu_build:
        notes.append("A GPU is present but this build lacks the GPU front-end; "
                     "rebuild with -DENABLE_GPU=ON to use --gpu-prefactor.")

    if notify:
        parts += ["--notify", notify]
        notes.append("A completion/failure notification will be sent via '%s'."
                     % notify)

    cluster = plan["recommend_cluster"]
    if cluster:
        notes.append("Estimated wall time is large for one machine; generate a "
                     "batch script with --suggest-slurm-config (or --suggest-pbs-"
                     "config), or fan clients out with scripts/cluster-launch.sh.")
    else:
        wall = planner._fmt_duration(plan["wall_seconds"]["central"])
        notes.append("This size is comfortable on a single machine (~%s on this "
                     "host); no cluster needed." % wall)

    return {
        "command": " ".join(parts),
        "feasibility": plan["feasibility"],
        "cluster": cluster,
        "gpu_prefactor": use_gpu_pre,
        "notes": notes,
    }


def format_recommendation(rec):
    """Render a recommendation dict as the wizard's final screen (string)."""
    lines = ["", "=== Recommended command " + "=" * 40]
    if rec["command"] is None:
        lines.append("(no NFS run recommended for this input)")
    else:
        lines.append("")
        lines.append("    " + rec["command"])
        lines.append("")
        lines.append("Replace <N> with the integer to factor (key=value params "
                     "must come before -t/flags).")
    lines.append("")
    lines.append("Why:")
    for note in rec["notes"]:
        lines.append("  - " + note)
    return "\n".join(lines)


def _ask(prompt, default=None, cast=str, choices=None, _input=input):
    """Prompt until a valid answer (``_input`` is injectable for tests). Empty
    input takes ``default``; ``choices`` restricts the accepted values."""
    suffix = " [%s]" % default if default is not None else ""
    while True:
        raw = _input("%s%s: " % (prompt, suffix)).strip()
        if not raw and default is not None:
            return default
        if not raw:
            continue
        if cast is bool:
            if raw.lower() in ("y", "yes", "1", "true"):
                return True
            if raw.lower() in ("n", "no", "0", "false"):
                return False
            print("  please answer yes or no")
            continue
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print("  not a valid value")
            continue
        if choices and val not in choices:
            print("  choose one of: %s" % ", ".join(map(str, choices)))
            continue
        return val


def run_wizard(detected_threads=None, gpu_present=False, gpu_build=False,
               _input=input, _print=print):
    """Interactive front-end: ask the few questions, then print the recommended
    command. Returns the recommendation dict (also for testing)."""
    if detected_threads is None:
        detected_threads = planner.detect_threads()
    _print("cado-nfs parameter wizard (Track E12) -- answer a few questions; "
           "nothing runs until you copy the command at the end.\n")

    size = _ask("Number of decimal digits of N (or paste N itself)",
                cast=str, _input=_input)
    n = digits = None
    s = size.strip()
    if len(s) > 4 and s.isdigit():
        n = int(s)
    else:
        try:
            digits = int(s)
        except ValueError:
            digits = 100

    threads = _ask("Threads to use", default=detected_threads, cast=int,
                   _input=_input)
    use_gpu = False
    if gpu_present:
        use_gpu = _ask("A GPU was detected -- use --gpu-prefactor?", default=True,
                       cast=bool, _input=_input)
    notify = None
    if _ask("Send a completion notification?", default=False, cast=bool,
            _input=_input):
        notify = _ask("Notify channel (e.g. desktop, ntfy=TOPIC, "
                      "email=ADDR)", default="desktop", cast=str, _input=_input)

    rec = recommend(n=n, digits=digits, threads=threads, gpu=use_gpu,
                    gpu_build=gpu_build, notify=notify)
    _print(format_recommendation(rec))
    return rec


if __name__ == "__main__":
    import doctest
    doctest.testmod()
