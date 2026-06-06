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
import datetime


class _Reporter:
    def __init__(self):
        self._lock = threading.RLock()
        self._json_path = None
        self._progress = False
        self._stderr_isatty = False
        self._enabled = False
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
            "started": None,           # ISO8601
            "updated": None,           # ISO8601
        }

    # -- configuration (called once, from cado-nfs.py) -----------------------

    def configure(self, json_path=None, progress=False, name=None,
                  computation=None, input_digits=None):
        with self._lock:
            self._json_path = json_path
            self._progress = bool(progress)
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
            self._state.update({
                "state": state,
                "factors": list(factors) if factors is not None else None,
                "phase": "complete" if state == "done" else self._state["phase"],
                "phase_percent": 100.0 if state == "done" else
                                 self._state["phase_percent"],
            })
            self._flush_locked()
            if self._progress:
                # leave the final line on screen
                sys.stderr.write("\n")
                sys.stderr.flush()

    # -- output --------------------------------------------------------------

    def _flush_locked(self):
        self._state["updated"] = self._now()
        if self._json_path:
            self._write_json_locked()
        if self._progress:
            self._write_progress_line_locked()

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


# process-wide singleton
STATUS = _Reporter()
