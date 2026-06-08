# Usability / observability core — v3.4.0-modern (E9–E12, A7)

The v3.4.0 operator-experience core, all of which runs and is validated on the
reference box (Intel i9-10850K, 20 threads; single RTX 3090). It is the
shippable, measurable half of the cycle — the honestly-gated research half (C7
headline measured win; C5+/C6+/B5/A6) is covered in its own docs. See
[`ROADMAP-v3.4.0-modern.md`](ROADMAP-v3.4.0-modern.md).

Everything here is **purely additive and off by default**: no flag changes the
number-theoretic bounds or the factorization itself, and every new path is a
no-op unless explicitly requested. The new orchestration modules are doctested
(`test_python_{notify,runs,wizard,planner,…}`).

---

## E9 — Completion / failure notifications

A long factorization can run for hours or days; the operator should not have to
watch a terminal to learn it finished or died. `scripts/cadofactor/notify.py`
sends a short notification when the run reaches a terminal state, selected by
`--notify CHANNELS` (comma-separated `kind[=target]`):

| Channel | Target | Transport |
|---|---|---|
| `desktop` | — | `notify-send` (Linux) / `osascript` (macOS) |
| `ntfy=TOPIC` | topic | ntfy.sh push (server from `[notifications]`/`CADO_NTFY_SERVER`) |
| `slack=WEBHOOK` | webhook URL | Slack incoming-webhook `{"text":…}` |
| `discord=WEBHOOK` | webhook URL | Discord webhook `{"content":…}` |
| `webhook=URL` | URL | generic POST of the JSON event |
| `email=ADDR` | recipient | SMTP via `smtplib` |

Dependency-free (urllib / smtplib / subprocess). Secrets (Slack/Discord URLs,
SMTP credentials) come from the environment (`CADO_SMTP_*`, `CADO_NTFY_SERVER`)
or the live `[notifications]` parameter block — never the committed parameter
snapshot. A channel that fails to send logs a warning and is skipped: **a
notification problem can never abort or fail a completed factorization** (the
finish hook isolates every exception).

It is wired through the process-wide status singleton
(`status.py::STATUS.add_finish_hook`), which every terminal path already calls
(normal completion, error, and the GPU-prefactor-only completion), so one hook
covers them all. Example:

```
./cado-nfs.py <N> --notify desktop,ntfy=my-topic -t 8
```

## E10 — Structured event log + Prometheus `/metrics`

Two machine-facing observability outputs:

* **`--json-log FILE`** appends an NDJSON event log — one self-contained JSON
  object per line, with an ISO-8601 `ts` and an `event` type: `run_start`
  (name/computation/digits), `phase_start` (phase + index/total), `run_finish`
  (state/elapsed/factors). Ideal for `tail -f | jq` or shipping to Loki/Grafana.
  It is independent of `--json-status` (a live single-snapshot file). A 59-digit
  smoke run emits 14 lines: `run_start`, the 12 `phase_start`s, `run_finish`.

* **`/metrics`** on both work-unit servers. The Flask `api_server.py` renders the
  live run status (`status.py::prometheus()`) as Prometheus text exposition —
  `cado_nfs_up`, a labelled-enum `cado_nfs_state{state=…}`, and gauges for
  `input_digits`, `phase_index/total`, `phase_percent`, `wu_done/total`,
  `elapsed_seconds`, `factors_total`. The Rust `cado-wu-server` adds
  `cado_wu_*` work-unit gauges (by status + percent + serving) from the wudb.
  The two are complementary (driver phase/ETA vs server work-unit counts) and
  always available alongside the existing `/status` and `/dashboard`.

## E11 — Multi-run history DB

`scripts/cadofactor/runs.py` keeps a small SQLite ledger at `~/.cado-nfs/runs.db`
(override with `CADO_RUNS_DB`). One row is appended per terminal run (via a
status finish hook, best-effort): timestamp, name, `N` (as text, so huge inputs
are exact), digits, computation, host, threads, wall time, state, factor count.
No secrets — only sizes, timings, and the host name.

* **`--list-runs`** prints the history as a table.
* **`--compare-runs SPEC`** focuses it: empty (recent runs), a digit count (all
  runs at that size with min/mean/max wall time), or `A:B` (two run ids
  side-by-side with a B/A wall ratio).

It is also the training source for A7.

## E12 — Per-phase ETA + wizard + completions

* **Per-phase ETA.** `cado-nfs-monitor-rs` and the `/dashboard` now reset their
  trailing-window rate estimate when the phase index changes, so the trend ETA is
  scoped to the *current* phase (its `phase_percent` restarts at 0 each phase, so
  spanning the discontinuity would corrupt the rate). Both also show an
  **ETA (all phases)** that adds the current-phase remainder to a flat estimate
  for the phases still to come — a coarse end-to-end number the per-phase ETA
  cannot give. Rows are labelled `ETA (phase)` and `ETA (all phases)`.

* **`--wizard`.** An interactive parameter wizard (`scripts/cadofactor/wizard.py`)
  asks the few decisions that matter — size, threads, GPU pre-factoring,
  notifications, single-machine vs cluster — then prints a ready-to-run command
  line and the rationale, reusing the `--plan` feasibility/cluster/GPU triage.
  Nothing runs until you copy the command. The decision logic is a pure,
  doctested function; only the prompt loop is interactive.

* **Context-aware completions + man EXAMPLES.** `scripts/build-completions.py`
  introspects the argparse spec, so the new flags appear in the bash/zsh/fish
  completions automatically (now 63 option strings); the file-valued flags
  (`--json-log`, `--galois-detect`, …) complete real paths. `misc/man/cado-nfs.1`
  gains a "Monitoring, notifications & history" options block and worked EXAMPLES
  for `--notify`/`--json-log`, `--list-runs`/`--compare-runs`/`--calibrate`, and
  `--wizard`.

## A7 — Data-driven autotuner

The static `--plan` wall-time model is anchored on one reference box. Once a host
has accumulated real runs in `runs.db`, `planner.py` refines the estimate from
the host's *own* measurements — **without touching any number-theoretic bound**
(it is purely a prediction refinement):

* **`--calibrate`** backs out this host's per-core speed factor versus the
  reference, as the geometric mean of the per-run implied factors
  (`model_at_speed1(digits,threads) / measured_wall`), and prints the suggested
  `--host-speed` value plus the regression fit quality.

* **`regression_estimate`** fits an ordinary-least-squares log-linear cost model
  (`log(wall) ~ a + b·digits`, after normalising each run to the reference thread
  count) over the host's runs, and `--plan` folds the empirical estimate in
  *alongside* the static model, clearly labelled with its sample count and basis
  (`regression` with ≥3 runs spanning ≥10 digits, else a single `calibrated`
  multiplicative correction). The more you run, the sharper the plan.

Validation (seeded, deterministic): five runs synthesised at a clean 1.5× host
speed reproduce `--calibrate → 1.500×` and an R² of 0.978; the `--plan` empirical
refinement lands within the reported ±20 % variance band of the static
model ÷ 1.5.

---

## Honesty / gates (unchanged fork ethos)

* No NFS algorithm or parameter is changed; every feature is additive and
  default-off.
* The orchestration modules are doctested and run under the configured
  `PYTHON_EXECUTABLE` (the venv with Flask/requests).
* Notification/observability/history failures are isolated and never affect a
  factorization.
* No single-machine *speed* win is claimed here — this is operator experience and
  prediction accuracy. The measured speed item this cycle is C7 (GPU P-1/P+1);
  the rest of the research track is reported with its honest non-wins.
