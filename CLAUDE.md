# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CADO-NFS is an implementation of the Number Field Sieve (NFS) for integer factorization and discrete logarithms. C (C99) + C++ (**C++20 required** as of 3.0.0) for the core, Python 3 (with **Flask**/requests) for orchestration (`cado-nfs.py`).

**This repo is `doublegate/cado-nfs-modern`** (renamed 2026-06-06 from `cado-nfs-3.0.0-modern`; GitHub auto-redirects the old URL) — a modernization + performance fork of upstream [CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) 3.0.0 (LGPL-2.1). Internal version is **`3.1.0-modern`** (`CMakeLists.txt`: numeric `3.1.0` + `-modern` suffix — the fork now carries its own minor line; upstream NFS algorithms/parameters are unchanged). From 3.1.0 it adds, on top of the 3.0.0-modern build/SIMD/GPU-cofactor/Rust tracks, **GPU linear algebra** (BWC SpMV + full vector residency), a **GPU pre-NFS ECM front-end** (`--gpu-prefactor`), **AVX-512 VPCLMULQDQ + IFMA** kernels (SDE-validated), and an expanded **orchestration/UX** layer. `main` tracks the latest release (`v3.1.0-modern`); `v3.0.0-modern` and `v2.3.1-modern` are preserved under their tags. See `CHANGELOG.md` for the full patch list, `README.md` for attribution, and `docs/` for the deep-dives.

## Build

```bash
bash scripts/setup-venv.sh   # ONE-TIME: create cado-nfs.venv (Flask + requests)
make            # configures + builds via scripts/call_cmake.sh (out-of-tree)
make cmake      # re-run cmake configuration (do this after editing local.sh)
make tidy       # DANGER: deletes the entire build tree (interactive y/N prompt)
```

- **3.0.0 needs Flask + requests at *configure* time.** The committed `local.sh` points cmake at `$PWD/cado-nfs.venv/bin/python3`, so create that venv once with `scripts/setup-venv.sh` before the first `make`. Without it, cmake configuration fails on the missing interpreter.
- The top-level `Makefile` is a thin wrapper over `scripts/call_cmake.sh`; it does **not** build directly. Do not run `cmake` by hand expecting `local.sh` to be read.
- Build output goes to `build/<hostname>/` (named by hostname, so one source tree serves multiple machines). There is no `build/` until you build.
- **Configuration is via `local.sh`**, not direct cmake flags: copy `local.sh.example` to `local.sh`, edit (`CC`, `CXX`, `CFLAGS`, `CXXFLAGS`, `GMP`, `MPI`, `PREFIX`, `build_tree`, ...), then `make cmake` to reconfigure. `local.sh` is sourced by `call_cmake.sh`, **not** by cmake itself — a plain `cmake /path` build ignores it. This fork's committed `local.sh` sets `-O3 -march=native -mtune=native` (Phase 1: ~7% on the siever; see `CHANGELOG.md`) plus the venv `PYTHON_EXECUTABLE`.
- **GMP (v5+) is mandatory and must be built with `--enable-shared`** or compilation fails. Optional: MPI, hwloc, curl. Locate GMP via `GMP` / `GMP_LIBDIR` / `GMP_INCDIR`.
- **Toolchain floor (3.0.0):** C99 + C++20; GCC >= 10 / LLVM Clang >= 12 / Apple Clang >= 16 / Intel ICX >= 2023; CMake >= 3.18.

## Test

```bash
make check                          # run the test suite (ctest under the hood)
make check ARGS="-j"                # parallel
make check ARGS="-R test_memusage"  # run tests matching a regex
```

Expensive tests are opt-in: `export CHECKS_EXPENSIVE=yes && make cmake && make check`. Tests live in `tests/`, mirror the source tree, and are named `test_*.c` / `test_*.sh`.

### Verifying a change fast (do this after editing C/C++)

The full suite is ~795 tests. When you touch one subsystem, run only its tests with `-R <regex>` instead of the whole suite — each test has a `builddep_<name>` companion that ctest builds automatically, so a targeted run also recompiles just what it needs. Two ways to invoke:

```bash
make check ARGS="-R 'bwc|bitlinalg|lingen'"        # from the source root (rebuilds deps first)
cd build/$(hostname) && ctest -R 'bwc' --output-on-failure   # faster, if the tree is already built
```

Map of what to run after patching each directory (ctest has **no labels** — selection is purely by test-name regex):

| You patched | Run `make check ARGS="-R '<regex>'"` | Matches |
|-------------|--------------------------------------|---------|
| `sieve/` | `sievetest\|F9_` | `sievetest_I`, `F9_sievetest*`, `F9_makefbtest`, `F9_dupsuptest`, `F9_fakereltest` |
| `linalg/` (incl. `linalg/bwc/`) | `bwc\|bitlinalg\|lingen\|matmul` | `test-bwc-*`, `test_bitlinalg_*`, `bwc_staged_krylov`, `dispatch-matmul-*`, `lingen` |
| `polyselect/` | `polyselect` | polynomial-selection tests |
| `sqrt/` | `sqrt\|testsm` | square-root tests |
| `numbertheory/` | `numbertheory` | `numbertheory-*` |
| `gf2x/`, `linalg/m4ri`-style low-level | `mpfq\|bitlinalg` | `mpfq_test_*`, matrix-op tests |

`linalg/bwc/cpubinding.cpp` (hwloc CPU-binding) is exercised by the `bwc` tests **and** by any real factorization's Linear Algebra phase — a `./cado-nfs.py` smoke test is the surest end-to-end check for it. For the whole pipeline, a 59-digit smoke factorization (`cado-nfs.venv/bin/python3 ./cado-nfs.py 90377629292003121684002147101760858109247336549001090677693 -t 4`) runs in ~15 s.

## Run

Main entry point is `./cado-nfs.py`. It needs the Python `sqlite3` module **and** Flask/requests (the work-unit server is Flask now), so run it under the venv: `cado-nfs.venv/bin/python3 ./cado-nfs.py ...`. It deduces the `build/<hostname>/` binary dir automatically — invoke it from the source root, do not call binaries by path.

```bash
cado-nfs.venv/bin/python3 ./cado-nfs.py <N>          # factor N on all local cores (default -t all)
cado-nfs.venv/bin/python3 ./cado-nfs.py <N> -t 2     # cap at 2 threads
cado-nfs.venv/bin/python3 ./cado-nfs.py /path/XXX.parameters_snapshot.YYY   # resume an interrupted run
```

Optimized for numbers > 85 digits; < 60 digits is unsupported. Strip small prime factors (trial division / P-1 / P+1 / ECM) before using it. Parameter presets per size live in `parameters/`. **Arg-order quirk:** `key=value` params must come *before* `-t`/other flags (`./cado-nfs.py <N> server.ssl=no -t 4`).

## Directory map (NFS stages)

| Dir | Stage / contents |
|-----|------------------|
| `polyselect/` | Polynomial selection (first NFS stage) |
| `sieve/` | Lattice siever (`las`), relation collection, factor base |
| `filter/` | Filtering: merge / purge / balance the relation matrix |
| `linalg/` | Linear algebra — Block Wiedemann (BWC), MPI-capable |
| `sqrt/` | Square root in the number field (final stage) |
| `numbertheory/`, `utils/`, `misc/` | Number-theory helpers, generic utilities, profiling |
| `scripts/cadofactor/` | Python orchestration: task scheduling, work-unit distribution (Flask `api_server.py`, `cadotask.py`, `wudb.py`) |
| `rust/` | Phase-4 Rust orchestration: static-binary client + async work-unit server (see below) |
| `config/` | CMake compiler/dependency detection |
| `parameters/` | Per-size factorization parameter presets (c60, c90, ...) |
| `gf2x/` | GF(2)[x] arithmetic (separate configure) |
| `docs/` | Theory explainers + the GPU / Rust deep-dives (see below) |

## Gotchas

- Some ancient GCC versions miscompile CADO-NFS (4.1.2 / 4.2.0–4.2.2). Irrelevant on a modern box, but noted upstream.
- GMP <= 6.0 + multi-threaded sqrt: pass `tasks.sqrt.threads=1` or upgrade to GMP >= 6.1.0.
- Numbers > 200 digits need `FLAGS_SIZE` set in `local.sh` to enable 64-bit counters.
- Distributed/server mode needs SSH public-key auth and `localhost` -> 127.0.0.1; see `README` for the SSH config block.
- No `.clang-format` or enforced style exists. Match the style of surrounding code; do not bulk-reformat. (Fork-specific files like `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/*`, `rust/*` were added by this fork.)
- `-march=native` in the committed `local.sh` is host-specific; for a distributable build, override it in your own `local.sh`.

## This fork (rebased on upstream 3.0.0)

3.0.0 already subsumes the 2.3.x portability fixes the prior `2.3.1-modern` fork carried (hwloc 2.x, Python 3 stdlib moves, OpenSSL 3.x SSL) and brings, for free, the Bouvier–Imbert batch cofactorization (eprint 2018/669) and `I>16` sieving. What *this* fork adds on top, validated and measured (full detail in `CHANGELOG.md`):

**3.0.0-modern tracks (foundation):**

- **Build (Phase 1):** `local.sh` sets `-O3 -march=native` (the only real CPU win — LTO/PGO were measured and rejected). Builds on CMake 4.x / GCC 16 / Python 3.14.
- **SIMD (Phase 2):** AVX2 on the siever was ruled out by profiling; an AVX-512 VPCLMULQDQ gf2x base-case kernel is validated bit-exact under Intel SDE (perf-gated on real AVX-512 silicon). See `gf2x/already_tuned/x86_64_vpclmul/`.
- **GPU cofactorization (Phase 3):** a complete batched ECM cofactorization backend (`sieve/ecm/gpu/`, CMake CUDA), validated bit-exact vs the CPU path. Honest finding: cofactorization is Amdahl-bounded (~8% of siever time) → **no net single-machine speedup** at these sizes. See `docs/gpu-cofactorization.md`.
- **Rust orchestration (Phase 4):** the `rust/` workspace — `cado-nfs-client-rs` (static-binary work-unit client) and `cado-wu-server-rs` (async axum/tokio server over the same `wudb` SQLite schema). Keeps the exact HTTP/JSON protocol, so it interoperates with an unmodified `cado-nfs.py`. The Rust server can be **swapped in** for the Flask `api_server.py` in a live run by setting `CADO_RUST_WU_SERVER=<path to cado-wu-server-rs>` (a Python shim, `scripts/cadofactor/external_api_server.py`, launches it; `cadotask.py` switches to it). TLS + IP whitelist + cert-pinning all work through the swap. `cadotask.py` (the high-level task DAG) is deliberately left in Python. See `docs/rust-orchestration.md` and the `rust/*-test.sh` validators.

**3.1.0-modern additions (every shipped change gated by `product == N` or a bit-exact validation; HW-blocked items are documented designs, not committed unvalidated code):**

- **GPU linear algebra:** a real `mm_impl=gpu` BWC SpMV backend (`linalg/bwc/matmul-gpu.cu`, b64/b128) with M+Mᵀ resident as CSR, a coalesced warp-per-row kernel, and **full vector residency** — the steady krylov/mksol/secure loop runs entirely on device buffers (2D comm + GPU `x_dotprod` + device `addmul_tiny` ported). Env: `CADO_GPU_VECRESIDENT=1 CADO_GPU_DEVCOMM=1`. The win grows with N (~7.9 Gnz/s at c120-scale, ~4.4× the tuned `bucket`). Intra-node multi-GPU partition via `CADO_GPU_NPART` (default 1; validated at N=1, true multi-GPU HW-gated); multi-node residency is a documented design. See `docs/gpu-linalg.md`. Hook definitions live in `linalg/bwc/matmul-gpu-hooks.cpp` (in `matmul_common`, so `bench_matcache --impl gpu` links).
- **GPU pre-NFS factoring:** `cado-nfs.py --gpu-prefactor` (`misc/gpu_prefactor/`, `scripts/cadofactor/gpu_prefactor.py`) strips factors with batched multi-precision GPU ECM before NFS — a separate stage with no Amdahl ceiling (~49×/26×/12× the full CPU at 128/256/512-bit). See `docs/gpu-prefactor.md`.
- **AVX-512 (correctness-only, CI-gated under SDE):** gf2x VPCLMULQDQ auto-detection (`gf2x/config/features.m4` `CHECK_VPCLMUL_SUPPORT`, run-test-gated so non-AVX-512 hosts keep pclmul — safe) and an IFMA GF(p) Montgomery modmul kernel (`bench/ifma-modmul.c`). Both bit-exact under Intel SDE (`bench/{vpclmul,ifma}-validate.sh`; SDE at `/opt/intel-sde/sde64`), gated by `.github/workflows/avx512-validate.yml`.
- **Orchestration/UX:** parameter interpolation + `--suggest-params` (`toplevel.py`); `--json-status`/`--progress` (`status.py`) + `GET /status` & `/dashboard` on both servers; `clap` CLIs for the Rust binaries; `scripts/cluster-launch.sh` (SSH/Slurm); `cado-nfs-monitor-rs` (ratatui).
- **Honest negatives (measured, recorded):** siever-trained PGO retry (+3%, rejected); no safe hot-scalar micro-opt (hot loops already SIMD/unrolled/prefetched).
- **Versioning:** `CADO_VERSION_STRING` is `3.1.0-modern` (`CMakeLists.txt`: `CADO_VERSION_MINOR 1` + `-modern` suffix). The numeric minor now **deliberately diverges** from upstream 3.0.0 to mark the fork's substantial original work; no test depends on the string.

### Docs

- `docs/number-field-sieve.md` — in-depth NFS mathematics explainer.
- `docs/number-field-sieve-plain-english.md` — the same in layman's terms.
- `docs/gpu-cofactorization.md` — GPU ECM cofactorization: measured results + honest Amdahl analysis (+ 3.1.0 cofactor scale-out / product-tree designs).
- `docs/gpu-prefactor.md` — (3.1.0) GPU pre-NFS ECM factoring front-end: why it sidesteps Amdahl, the multi-precision Montgomery ECM, measured CPU-vs-GPU.
- `docs/gpu-linalg.md` — (3.1.0) GPU BWC SpMV backend + full vector residency: kernel, transfer analysis, at-scale sweep, multi-GPU partition + multi-node residency design.
- `docs/gpu-ecm-mixedrep.md` — (3.2.0, A2) mixed-representation ECM: the CPU path already does it (upstream "mishmash"); a validated twisted-Edwards `a=−1` GPU stage-1 (bit-exact vs the ladder via the birational map) is ~1.5–2.9× the Montgomery ladder, growing with modulus width.
- `docs/rust-orchestration.md` — the Phase-4 Rust client/server, the protocol, and the in-process swap.
- `BENCHMARKS.md` — performance sweep + methodology.
- Upstream: `@README` (build/run/distributed), `@README.dlp` (discrete log), `@README.Python` (orchestration internals).
