"""
Multi-run history database for cado-nfs.py (v3.4.0-modern, Track E11).

A small SQLite ledger of completed runs at ``~/.cado-nfs/runs.db`` (override with
the ``CADO_RUNS_DB`` env). One row is appended when a run reaches a terminal state
(via a status finish hook), recording the input size, computation, host, thread
count, wall time and outcome. It powers ``--list-runs`` / ``--compare-runs`` for
campaign tracking, and is the training source for the A7 data-driven autotuner
(``planner.regression_estimate``).

Recording is best-effort and side-effect-isolated: a DB error never affects a
completed factorization (the finish hook in status.py swallows exceptions). The
DB holds no secrets -- only sizes, timings and the host name -- and ``N`` is
stored as text so arbitrarily large inputs are preserved exactly.

>>> import tempfile, os
>>> dbp = os.path.join(tempfile.mkdtemp(), "runs.db")
>>> ev = {"name": "c90", "computation": "FACT", "input_digits": 90,
...       "state": "done", "elapsed": 197.9, "factors": ["p", "q"]}
>>> rid = record(ev, n="900...001", host="ref-box", threads=20, db_path=dbp)
>>> rid >= 1
True
>>> rows = list_runs(db_path=dbp)
>>> len(rows)
1
>>> rows[0]["digits"], rows[0]["computation"], rows[0]["state"]
(90, 'FACT', 'done')
>>> rows[0]["threads"], round(rows[0]["elapsed"], 1), rows[0]["nfactors"]
(20, 197.9, 2)
>>> "c90" in format_runs(rows)
True
"""

import os
import socket
import sqlite3


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    name        TEXT,
    n           TEXT,
    digits      INTEGER,
    computation TEXT,
    host        TEXT,
    threads     INTEGER,
    elapsed     REAL,
    state       TEXT,
    nfactors    INTEGER
);
"""


def default_db_path():
    """Resolve the run-history DB path: ``CADO_RUNS_DB`` if set, else
    ``~/.cado-nfs/runs.db`` (the parent directory is created on first write)."""
    env = os.environ.get("CADO_RUNS_DB")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".cado-nfs", "runs.db")


def _connect(db_path, create=True):
    if db_path is None:
        db_path = default_db_path()
    if create:
        d = os.path.dirname(os.path.abspath(db_path))
        if d:
            os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if create:
        conn.executescript(_SCHEMA)
    return conn


def record(event, n=None, host=None, threads=None, db_path=None):
    """Append one terminal-state run to the history DB. ``event`` is the
    status final-state dict (name / computation / input_digits / state /
    elapsed / factors). Returns the new row id (or None on any error)."""
    try:
        conn = _connect(db_path, create=True)
    except sqlite3.Error:
        return None
    try:
        from datetime import datetime
        factors = event.get("factors") or []
        with conn:
            cur = conn.execute(
                "INSERT INTO runs (ts, name, n, digits, computation, host, "
                "threads, elapsed, state, nfactors) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"),
                 event.get("name"),
                 str(n) if n is not None else None,
                 event.get("input_digits"),
                 event.get("computation"),
                 host if host is not None else socket.gethostname(),
                 threads,
                 event.get("elapsed"),
                 event.get("state"),
                 len(factors)))
            return cur.lastrowid
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def list_runs(db_path=None, limit=None, digits=None, host=None,
              state="done"):
    """Return recorded runs (newest first) as a list of dict rows. Optional
    filters: ``digits`` (input size), ``host``, ``state`` (None = any)."""
    try:
        conn = _connect(db_path, create=False)
    except sqlite3.Error:
        return []
    try:
        q = "SELECT * FROM runs"
        clauses, args = [], []
        if digits is not None:
            clauses.append("digits = ?")
            args.append(int(digits))
        if host is not None:
            clauses.append("host = ?")
            args.append(host)
        if state is not None:
            clauses.append("state = ?")
            args.append(state)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id DESC"
        if limit is not None:
            q += " LIMIT %d" % int(limit)
        try:
            return [dict(r) for r in conn.execute(q, args).fetchall()]
        except sqlite3.Error:
            return []
    finally:
        conn.close()


def _fmt_elapsed(seconds):
    """Compact h/m/s for the table.

    >>> _fmt_elapsed(42.0)
    '42s'
    >>> _fmt_elapsed(197.9)
    '3m18s'
    >>> _fmt_elapsed(None)
    '-'
    """
    if seconds is None:
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return "%dh%02dm" % (h, m)
    if m:
        return "%dm%02ds" % (m, sec)
    return "%ds" % sec


def format_runs(rows):
    """Render run rows as a fixed-width table (string)."""
    if not rows:
        return "(no runs recorded yet; ~/.cado-nfs/runs.db is empty)"
    hdr = "%-4s %-19s %-12s %5s %-5s %-14s %4s %8s %-6s" % (
        "id", "when", "name", "dig", "comp", "host", "thr", "wall", "state")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append("%-4s %-19s %-12s %5s %-5s %-14s %4s %8s %-6s" % (
            r["id"], (r["ts"] or "")[:19], (r["name"] or "")[:12],
            r["digits"] if r["digits"] is not None else "-",
            (r["computation"] or "")[:5], (r["host"] or "")[:14],
            r["threads"] if r["threads"] is not None else "-",
            _fmt_elapsed(r["elapsed"]), (r["state"] or "")[:6]))
    return "\n".join(lines)


def compare_runs(spec, db_path=None):
    """Return a focused comparison string (Track E11). ``spec``:
    - empty / None  -> the most recent runs;
    - a digit count -> all runs at that input size, with mean/min/max wall;
    - 'A:B'         -> the two runs with ids A and B side by side."""
    spec = (spec or "").strip()
    if ":" in spec:
        a, b = spec.split(":", 1)
        all_rows = {str(r["id"]): r for r in list_runs(db_path=db_path,
                                                       state=None)}
        picked = [all_rows[x.strip()] for x in (a, b) if x.strip() in all_rows]
        if len(picked) < 2:
            return "could not find both run ids %r" % spec
        return _format_pair(picked[0], picked[1])
    if spec.isdigit():
        rows = list_runs(db_path=db_path, digits=int(spec))
        if not rows:
            return "no completed runs at %s digits" % spec
        walls = [r["elapsed"] for r in rows if r["elapsed"] is not None]
        out = [format_runs(rows)]
        if walls:
            out.append("")
            out.append("wall time at %s digits over %d run(s): "
                       "min %s  mean %s  max %s" % (
                           spec, len(walls), _fmt_elapsed(min(walls)),
                           _fmt_elapsed(sum(walls) / len(walls)),
                           _fmt_elapsed(max(walls))))
        return "\n".join(out)
    return format_runs(list_runs(db_path=db_path, limit=20))


def _format_pair(ra, rb):
    fields = [("id", "id"), ("when", "ts"), ("name", "name"),
              ("digits", "digits"), ("computation", "computation"),
              ("host", "host"), ("threads", "threads"),
              ("wall", "elapsed"), ("state", "state")]
    lines = ["%-12s %-22s %-22s" % ("field", "run A", "run B")]
    lines.append("-" * len(lines[0]))
    for label, key in fields:
        va = ra.get(key)
        vb = rb.get(key)
        if key == "elapsed":
            va, vb = _fmt_elapsed(va), _fmt_elapsed(vb)
        lines.append("%-12s %-22s %-22s" % (label, str(va)[:22], str(vb)[:22]))
    if ra.get("elapsed") and rb.get("elapsed"):
        ratio = rb["elapsed"] / ra["elapsed"] if ra["elapsed"] else 0
        lines.append("")
        lines.append("B/A wall ratio: %.2fx" % ratio)
    return "\n".join(lines)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
