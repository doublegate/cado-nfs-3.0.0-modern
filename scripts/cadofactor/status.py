"""
Lightweight run-status reporting for cado-nfs.py (v3.1.0-modern, Track 3.1).

A single process-wide reporter that the orchestration updates as it goes:
- the phase loop (CompleteFactorization.run) calls set_phase() when a stage starts;
- the per-work-unit verification() calls update_progress() with the achievement
  fraction and ETA that the task framework already computes.

Two outputs, both optional and off by default (no behaviour change unless asked):
- ``--json-status FILE``: a machine-readable snapshot rewritten atomically on
  every update (for dashboards / tooling / the /status endpoint in Track 3.2);
- ``--progress``: a compact single-line human progress indicator on stderr
  (``\\r``-updated). Pair it with ``--screenlog WARNING`` for a clean line, since
  by default the verbose INFO log shares stderr.

This module has no third-party dependencies and is import-safe: if it is never
configured, every hook is a cheap no-op.
"""

import json
import os
import sys
import threading
import time
import datetime


class _Reporter:
    def __init__(self):
        self._lock = threading.RLock()
        self._json_path = None
        self._progress = False
        self._stderr_isatty = False
        self._enabled = False
        # v3.4.0 Track E9/E10/E11: terminal-event finish hooks (each a callable
        # taking the final state dict -- used by the notifier and the run-history
        # recorder) and an NDJSON structured event log.
        self._finish_hooks = []
        self._json_log_path = None
        self._started_mono = None
        self._state = {
            "schema": "cado-nfs-status/1",
            "state": "starting",       # starting | running | done | error
            "name": None,
            "computation": None,
            "input_digits": None,
            "phase": None,             # human-readable current stage title
            "phase_index": None,       # 1-based position in the task list
            "phase_total": None,
            "phase_percent": None,     # 0..100 within the current phase (WU phases)
            "eta": None,               # human-readable arrival time, or "Unknown"
            "wu_done": None,
            "wu_total": None,
            "factors": None,
            "elapsed": None,           # wall seconds at finish (E9/E10)
            "started": None,           # ISO8601
            "updated": None,           # ISO8601
        }

    # -- configuration (called once, from cado-nfs.py) -----------------------

    def add_finish_hook(self, hook):
        """Register a callable invoked with the final state dict when the run
        reaches a terminal state (Track E9 notifier, E11 run recorder). Hooks are
        best-effort: an exception in one never breaks the run or the others."""
        if hook is not None:
            with self._lock:
                self._finish_hooks.append(hook)

    def configure(self, json_path=None, progress=False, name=None,
                  computation=None, input_digits=None, json_log=None):
        with self._lock:
            self._json_path = json_path
            self._progress = bool(progress)
            self._json_log_path = json_log
            self._started_mono = time.monotonic()
            self._stderr_isatty = bool(getattr(sys.stderr, "isatty",
                                               lambda: False)())
            self._enabled = bool(json_path) or self._progress
            now = self._now()
            self._state.update({
                "state": "running",
                "name": name,
                "computation": computation,
                "input_digits": input_digits,
                "started": now,
            })
            if self._enabled:
                self._flush_locked()
            self._log_event_locked("run_start", {
                "name": name, "computation": computation,
                "input_digits": input_digits})

    def is_enabled(self):
        return self._enabled

    # -- updates (called from the orchestration; cheap no-ops if disabled) ----

    def set_phase(self, title, index=None, total=None):
        # The in-memory state is tracked unconditionally (cheap) so an in-process
        # reader (the /status endpoint, Track 3.2) always sees live progress; only
        # the file/stderr *output* is gated, inside _flush_locked.
        with self._lock:
            self._state.update({
                "phase": title,
                "phase_index": index,
                "phase_total": total,
                # a new phase resets the WU progress fields
                "phase_percent": None,
                "eta": None,
                "wu_done": None,
                "wu_total": None,
            })
            self._flush_locked()
            self._log_event_locked("phase_start", {
                "phase": title, "phase_index": index, "phase_total": total})

    def update_progress(self, percent=None, eta=None, wu_done=None,
                        wu_total=None):
        with self._lock:
            if percent is not None:
                # CADO's own achievement estimate can briefly overshoot 100%
                # (more work-units received than the range estimate); clamp for
                # a clean progress display.
                self._state["phase_percent"] = round(
                    min(100.0, max(0.0, float(percent))), 1)
            if eta is not None:
                self._state["eta"] = eta
            if wu_done is not None:
                self._state["wu_done"] = wu_done
            if wu_total is not None:
                self._state["wu_total"] = wu_total
            self._flush_locked()

    def finish(self, factors=None, state="done"):
        with self._lock:
            elapsed = (time.monotonic() - self._started_mono
                       if self._started_mono is not None else None)
            self._state.update({
                "state": state,
                "factors": list(factors) if factors is not None else None,
                "phase": "complete" if state == "done" else self._state["phase"],
                "phase_percent": 100.0 if state == "done" else
                                 self._state["phase_percent"],
                "elapsed": round(elapsed, 1) if elapsed is not None else None,
            })
            self._flush_locked()
            if self._progress:
                # leave the final line on screen
                sys.stderr.write("\n")
                sys.stderr.flush()
            self._log_event_locked("run_finish", {
                "state": state, "elapsed": self._state["elapsed"],
                "factors": self._state["factors"]})
            # E9/E11: fire the terminal-event hooks (notifier, run recorder). A
            # hook failure must never break a completed factorization, so each is
            # isolated. Snapshot is passed by value.
            final = dict(self._state)
            for hook in self._finish_hooks:
                try:
                    hook(final)
                except Exception:
                    pass

    # -- output --------------------------------------------------------------

    def _flush_locked(self):
        self._state["updated"] = self._now()
        if self._json_path:
            self._write_json_locked()
        if self._progress:
            self._write_progress_line_locked()

    def _log_event_locked(self, event_type, fields):
        """Append one NDJSON event line to the structured event log (Track E10),
        if --json-log is configured. Best-effort: a log error never breaks a run.
        Each line is a self-contained object with an ISO8601 ``ts`` and ``event``
        type, suitable for ``tail -f | jq`` or shipping to a log pipeline."""
        if not self._json_log_path:
            return
        rec = {"ts": self._now(), "event": event_type}
        rec.update({k: v for k, v in fields.items() if v is not None})
        try:
            with open(self._json_log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _write_json_locked(self):
        try:
            tmp = self._json_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._state, f, indent=2)
                f.write("\n")
            os.replace(tmp, self._json_path)  # atomic on POSIX
        except OSError:
            # status reporting must never break a computation
            pass

    def _write_progress_line_locked(self):
        s = self._state
        bits = []
        if s["phase_index"] and s["phase_total"]:
            bits.append("[%d/%d]" % (s["phase_index"], s["phase_total"]))
        if s["phase"]:
            bits.append(str(s["phase"]))
        if s["phase_percent"] is not None:
            bits.append("%.1f%%" % s["phase_percent"])
        if s["wu_done"] is not None and s["wu_total"]:
            bits.append("wu %d/%d" % (s["wu_done"], s["wu_total"]))
        if s["eta"] and s["eta"] != "Unknown":
            bits.append("ETA " + str(s["eta"]))
        line = "  ".join(bits)
        try:
            if self._stderr_isatty:
                sys.stderr.write("\r\033[K" + line)
            else:
                sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except (OSError, ValueError):
            pass

    @staticmethod
    def _now():
        return datetime.datetime.now().isoformat(timespec="seconds")

    def snapshot(self):
        """Return a copy of the current status dict (for an in-process reader)."""
        with self._lock:
            return dict(self._state)

    def prometheus(self):
        """Render the current status as a Prometheus text-exposition string
        (Track E10), served at ``/metrics`` on both work-unit servers for
        Grafana/alerting. All gauges; the run state is a labelled enum gauge so a
        single ``cado_nfs_state`` series tracks starting/running/done/error."""
        s = self.snapshot()
        out = [
            "# HELP cado_nfs_up The cado-nfs status reporter is live.",
            "# TYPE cado_nfs_up gauge",
            "cado_nfs_up 1",
            "# HELP cado_nfs_state Current run state (1 for the active label).",
            "# TYPE cado_nfs_state gauge",
        ]
        cur = s.get("state") or "starting"
        for st in ("starting", "running", "done", "error"):
            out.append('cado_nfs_state{state="%s"} %d' % (st, 1 if st == cur
                                                          else 0))

        def _g(name, value, help_text):
            if value is None:
                return
            try:
                v = float(value)
            except (TypeError, ValueError):
                return
            out.append("# HELP %s %s" % (name, help_text))
            out.append("# TYPE %s gauge" % name)
            out.append("%s %g" % (name, v))

        _g("cado_nfs_input_digits", s.get("input_digits"),
           "Decimal digit count of the input N.")
        _g("cado_nfs_phase_index", s.get("phase_index"),
           "1-based index of the current phase.")
        _g("cado_nfs_phase_total", s.get("phase_total"),
           "Total number of phases.")
        _g("cado_nfs_phase_percent", s.get("phase_percent"),
           "Completion percent within the current phase (0-100).")
        _g("cado_nfs_wu_done", s.get("wu_done"),
           "Work-units completed in the current phase.")
        _g("cado_nfs_wu_total", s.get("wu_total"),
           "Estimated work-units for the current phase.")
        _g("cado_nfs_elapsed_seconds", s.get("elapsed"),
           "Wall-clock seconds at terminal state.")
        factors = s.get("factors")
        _g("cado_nfs_factors_total", len(factors) if factors else None,
           "Number of factors found (terminal state).")
        return "\n".join(out) + "\n"


# process-wide singleton
STATUS = _Reporter()
