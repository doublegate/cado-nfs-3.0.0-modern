# Usability / utility / help (Roadmap E4–E8)

> **Status: in progress (v3.3.0-modern).** This is the shippable, *measurable*
> operator-experience core of the cycle — everything here runs and helps on the
> reference box today. Sections are filled as each item lands; see
> [`ROADMAP-v3.3.0-modern.md`](ROADMAP-v3.3.0-modern.md) for the cycle framing.

The cycle *extends*, rather than rebuilds, the existing UX surface: the
`--json-status` / `--progress` reporters (`scripts/cadofactor/status.py`), the
`/status` + `/dashboard` endpoints on both the Flask `api_server.py` and the Rust
`cado-wu-server`, the ratatui monitor binary
(`rust/cado-nfs-client/src/bin/monitor.rs`), the `--plan` / `--plan-json` /
`--autotune` / `--suggest-params` flags (`scripts/cadofactor/toplevel.py`), the
factor planner (`scripts/cadofactor/planner.py`), and `scripts/cluster-launch.sh`.

## E4 — Live dashboard + ETA — DONE

Both views gained self-computed live metrics that don't depend on the server
reporting them, derived from the poll time-series.

**`cado-nfs-monitor-rs` (ratatui TUI).** A `Telemetry` ring buffer keeps the last
~120 s of `(time, percent, wu_done)` polls and derives, each tick:

* **ETA (trend)** — `(100 - percent) / (trailing %/min)`, shown beside the
  server-reported ETA (now labelled *ETA (server)*), so the two cross-check;
* **throughput** — work-units/min from `d(wu_done)/dt`;
* **host CPU** — local busy % from a `/proc/stat` idle/total delta over the poll
  interval;
* **host GPU** — local `nvidia-smi` utilisation + memory (read-only).

CPU/GPU are labelled *(local)* — they reflect the host the monitor runs on (the
compute node in a single-machine run). `--once` prints a one-shot summary plus a
250 ms CPU sample and a GPU sample (the trend metrics need ≥2 polls, so they are
TUI/dashboard-only). The per-phase position (`[i/t] phase`) is shown in the header.

**`/dashboard` HTML (Flask + Rust servers).** The dependency-free page now keeps the
same browser-side trailing-window history and shows **ETA (trend)** and
**throughput** rows alongside **ETA (server)** — the browser cannot read the
server's `/proc`/GPU, so CPU/GPU stay monitor-only (honest limit). Numeric
work-unit counts (`wu_done`/`wu_total`) are surfaced through the monitor's
`Snapshot` for both status schemas.

## E5 — `--doctor` preflight — DONE

`cado-nfs.py --doctor [N]` (and `--doctor-json`) runs a **side-effect-free**
preflight in `scripts/cadofactor/doctor.py` (dispatched from `toplevel.py`
alongside `--plan`, before any parameter file is required, so it works with or
without `N`). Checks, each reported as `[ OK ] / [info] / [WARN] / [FAIL]`:

* **build** — the build dir and the 7 core pipeline binaries (`polyselect`,
  `makefb`, `las`, `purge`, `merge`, `bwc.pl`, `sqrt`) are present (FAIL → NO-GO);
* **python** — `sqlite3` (FAIL if missing) plus `flask`/`requests` (WARN if
  missing — needed for the server / distributed mode);
* **cpu** — logical threads + SIMD level, honestly noting that AVX-512 kernels are
  SDE-validated only on a non-AVX-512 CPU;
* **memory** / **disk** — available RAM and free disk vs a coarse, clearly-labelled
  digit-tier estimate (WARN if short);
* **gpu** — device present + GPU front-end built (via `nvidia-smi`, read-only);
  WARN if a device is present but the build has no GPU front-end;
* **cluster** — which of `ssh`/`sbatch`/`sinfo`/`srun`/`qsub`/`mpirun` are on PATH;
* **size** — feasibility (FAIL below the ~60-digit NFS floor).

When `N` is given it also prints the `planner` wall-time envelope + GPU triage.
The overall verdict is the worst check status: **GO / GO (with warnings) /
NO-GO**. The feasibility, wall-time and GPU-triage logic is reused from
`planner.py` (not duplicated). Pure-logic helpers (`_verdict`, `_resource_tier`)
carry doctests.

## E6 — Shell completions + man pages — DONE

**`cado-nfs.py` completions (bash/zsh/fish).** Generated from the *single source of
truth* — the argparse parser in `cadofactor/toplevel.py` — by
`scripts/build-completions.py`, which introspects `parser._actions` and emits the
three shells into `misc/completions/cado-nfs.{bash,zsh,fish}`. The generated files
are committed (so end users need nothing) and regenerated whenever a flag changes:

```sh
PYTHONPATH=scripts cado-nfs.venv/bin/python3 scripts/build-completions.py
```

Path-taking flags (`--parameters/-p`, `--workdir/--wdir/-w`, `--json-status`) get
file completion; `-t/--server-threads` suggests `all` + common counts; all 51
option strings are covered. Each generated file carries a "regenerate, do not
hand-edit" header and passes `bash -n` / `zsh -n` / `fish -n`.

**Rust binaries.** `cado-nfs-client-rs`, `cado-wu-server-rs`, and
`cado-nfs-monitor-rs` each gained `--completions <bash|zsh|fish|elvish|powershell>`
(via `clap_complete`), printing the script to stdout and exiting. A small
pre-parse scan emits the script *before* clap's required-argument check, so the
flag works standalone (e.g. `cado-wu-server-rs --completions bash` without
`--db`); the clap field still documents it in `--help` and validates the value.

**Man page.** `misc/man/cado-nfs.1` (hand-written troff) covers the synopsis, the
planning/status/resource/GPU options, the `CADO_*` environment variables, examples
(incl. the 59-digit smoke factor), files, and see-also. Lints clean under `groff
-man -z`.

**Install.** CMake installs the completions into the conventional per-shell
locations (`share/bash-completion/completions/`, `share/zsh/site-functions/`,
`share/fish/vendor_completions.d/`) and the man page into `share/man/man1/`.

## E7 — Checkpoint/resume robustness + clarity — DONE

Honest scope: the machinery already exists; E7 makes it **discoverable** and adds a
top-level cadence knob — it does not add mid-work-unit checkpoints (work units are
small and atomic, so that buys little).

**What is resumable, per phase** (an interrupted run is resumed by passing its
snapshot: `cado-nfs.py NAME.parameters_snapshot.NNN`):

| phase | on resume |
|-------|-----------|
| polynomial selection | re-runs the cheap remaining search; completed records persist |
| **lattice sieving** | the work-unit DB (SQLite, `wudb`) tracks every completed unit; only *incomplete* units re-run — this is the bulk of the wall, and it is already crash-safe |
| filtering (purge/merge) | re-derived from the relation set (fast; not separately checkpointed) |
| **linear algebra (BWC)** | krylov writes recoverable V-vector checkpoints every `tasks.linalg.bwc.interval` iterations; `bwc.pl` restarts from the latest consistent checkpoint across all sub-vectors |
| square root | re-runs (short) |

**`--checkpoint-interval ITERS`** surfaces the previously-buried BWC krylov cadence
(`tasks.linalg.bwc.interval`, preset default ~2000) as a first-class flag: smaller
loses less linear-algebra work on an interrupt, at a small I/O cost. It is wired in
`toplevel.py` via `set_simple` and logged when applied. Sieving resumability is
independent of it (per-work-unit from the DB).

**Honest non-goals:** there is no mid-sieve / mid-work-unit checkpoint (a unit is
the atomic restart grain, and units are sized small); distributed mid-run
fault-tolerance beyond the existing work-unit retry + BWC checkpoints is out of
scope for v3.3.0.

## E8 — Slurm/PBS integration — DONE

**`scripts/cluster-launch.sh` gains a PBS/Torque path.** `--pbs` generates and
submits a `qsub` **job array** (mirroring the existing Slurm `--sbatch`): `#PBS -J
0-N` array, `-l select=1:ncpus=8[:ngpus=G]`, `-q <queue>` (from `--partition`), `-l
walltime` (from `--time`), `cd $PBS_O_WORKDIR`, GPU-pinned clients via
`CUDA_VISIBLE_DEVICES` (one per GPU). The per-task client line is now built by a
single `array_body <index-var>` helper shared by `--sbatch`
(`SLURM_ARRAY_TASK_ID`) and `--pbs` (`PBS_ARRAY_INDEX`), so the two stay in lock
step. `--dry-run` prints the generated script without submitting.

**`cado-nfs.py --suggest-slurm-config` / `--suggest-pbs-config`** print a
ready-to-edit submission script **sized to N**: the walltime comes from the
`planner` estimate (the single-machine high estimate, a safe over-bound for a small
cluster), the node count scales with the digit count, and the body wraps
`cluster-launch.sh` for the sieving-client fan-out with clearly-marked `EDIT_ME`
placeholders (partition/queue, account, server URL + cert SHA1) and a two-step
recipe (start the server, then submit). The generator is
`planner.format_batch_script(plan, scheduler, ...)` (+ `_clock_walltime`), both
doctested.
