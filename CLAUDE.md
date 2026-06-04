# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CADO-NFS is an implementation of the Number Field Sieve (NFS) for integer factorization and discrete logarithms. C (C99) + C++ (C++98 required, C++11 used when available) for the core, Python 3 for orchestration (`cado-nfs.py`).

**This repo is `doublegate/cado-nfs-2.3.1-modern`** — a modernization fork of upstream [CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) 2.3.0 (LGPL-2.1), patched to build and run on current toolchains. Internal version is **2.3.1** (`CMakeLists.txt`). Algorithms/parameters are unchanged from upstream; this fork is a portability layer only. See `CHANGELOG.md` for the full patch list and `README.md` for attribution.

## Build

```bash
make            # configures + builds via scripts/call_cmake.sh (out-of-tree)
make cmake      # re-run cmake configuration (do this after editing local.sh)
make tidy       # DANGER: deletes the entire build tree
```

- The top-level `Makefile` is a thin wrapper over `scripts/call_cmake.sh`; it does **not** build directly. Do not run `cmake` by hand expecting `local.sh` to be read.
- Build output goes to `build/<hostname>/` (named by hostname, so one source tree serves multiple machines). There is no `build/` until you build.
- **Configuration is via `local.sh`**, not direct cmake flags: copy `local.sh.example` to `local.sh`, edit (`CC`, `CXX`, `CFLAGS`, `CXXFLAGS`, `GMP`, `MPI`, `PREFIX`, `build_tree`, ...), then `make cmake` to reconfigure. `local.sh` is sourced by `call_cmake.sh`, **not** by cmake itself — a plain `cmake /path` build ignores it.
- **GMP (v5+) is mandatory and must be built with `--enable-shared`** or compilation fails. Optional: MPI, hwloc, curl. Locate GMP via `GMP` / `GMP_LIBDIR` / `GMP_INCDIR`.

## Test

```bash
make check                          # run the test suite (ctest under the hood)
make check ARGS="-j"                # parallel
make check ARGS="-R test_memusage"  # run tests matching a regex
```

Expensive tests are opt-in: `export CHECKS_EXPENSIVE=yes && make cmake && make check`. Tests live in `tests/`, mirror the source tree, and are named `test_*.c` / `test_*.sh`.

### Verifying a change fast (do this after editing C/C++)

The full suite is 515 tests. When you touch one subsystem, run only its tests with `-R <regex>` instead of the whole suite — each test has a `builddep_<name>` companion that ctest builds automatically, so a targeted run also recompiles just what it needs. Two ways to invoke:

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

`linalg/bwc/cpubinding.cpp` (hwloc CPU-binding) is exercised by the `bwc` tests **and** by any real factorization's Linear Algebra phase — a `./cado-nfs.py` smoke test is the surest end-to-end check for it. For the whole pipeline, a 59-digit smoke factorization (`./cado-nfs.py 90377629292003121684002147101760858109247336549001090677693 -t 4`) runs in ~30 s over HTTPS.

## Run

Main entry point is `./cado-nfs.py` (Python 3, needs the `sqlite3` module). It deduces the `build/<hostname>/` binary dir automatically — invoke it from the source root, do not call binaries by path.

```bash
./cado-nfs.py <N>           # factor N on all local cores (default -t all)
./cado-nfs.py <N> -t 2      # cap at 2 threads
./cado-nfs.py /path/XXX.parameters_snapshot.YYY   # resume an interrupted run
```

Optimized for numbers > 85 digits; < 60 digits is unsupported. Strip small prime factors (trial division / P-1 / P+1 / ECM) before using it. Parameter presets per size live in `parameters/`.

## Directory map (NFS stages)

| Dir | Stage / contents |
|-----|------------------|
| `polyselect/` | Polynomial selection (first NFS stage) |
| `sieve/` | Lattice siever (`las`), relation collection, factor base |
| `filter/` | Filtering: merge / purge / balance the relation matrix |
| `linalg/` | Linear algebra — Block Wiedemann (BWC), MPI-capable |
| `sqrt/` | Square root in the number field (final stage) |
| `numbertheory/`, `utils/`, `misc/` | Number-theory helpers, generic utilities, profiling |
| `scripts/cadofactor/` | Python orchestration: task scheduling, work-unit distribution |
| `config/` | CMake compiler/dependency detection |
| `parameters/` | Per-size factorization parameter presets (c60, c90, ...) |
| `gf2x/` | GF(2)[x] arithmetic (separate configure) |

## Gotchas

- Some GCC versions miscompile CADO-NFS: avoid 4.1.2 / 4.2.0 / 4.2.1 / 4.2.2.
- GMP <= 6.0 + multi-threaded sqrt: pass `tasks.sqrt.threads=1` or upgrade to GMP >= 6.1.0.
- Numbers > 200 digits need `FLAGS_SIZE` set in `local.sh` to enable 64-bit counters.
- Distributed/server mode needs SSH public-key auth and `localhost` -> 127.0.0.1; see `README` for the SSH config block.
- No `.clang-format` or enforced style exists — this is a fork of upstream v2.3.0. Match the style of surrounding code; do not bulk-reformat. (Fork-specific files like `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md` were added by this fork.)

## Modern-toolchain port (this checkout)

Upstream 2.3.0 is from 2017; building/running it on a current box (cmake 4.x, GCC 16, hwloc 2.x, OpenSSL 3.x, Python 3.14) required these fixes — already applied in this fork (which is therefore versioned 2.3.1). Keep them in mind before "reverting upstream":

- `CMakeLists.txt` — `CADO_VERSION_PATCHLEVEL` bumped 0 → 1 (version string `2.3.1`); affects the compiled-in version and install paths only, no test depends on it.
- `local.sh` (committed in this fork) — sets `CMAKE_EXTRA_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"` (cmake 4.x dropped pre-3.5 policy compat) and `CFLAGS="-O2 -fcommon"` (GCC 10+ defaults to `-fno-common`, which turns the project's tentative globals like `bw` into multiple-definition link errors).
- `gf2x/lowlevel/gen_bb_mul_code.c` — one `//` comment changed to `/* */`; this generator is compiled by gf2x's **build-system compiler** in ISO C90, where `//` is illegal.
- `linalg/bwc/cpubinding.cpp` — ported the removed hwloc-1.x topology flags (`HWLOC_TOPOLOGY_FLAG_IO_DEVICES`/`IO_BRIDGES`) to the hwloc-2.x `hwloc_topology_set_io_types_filter(..., HWLOC_TYPE_FILTER_KEEP_NONE)` API, version-guarded with `#if HWLOC_API_VERSION >= 0x00020000`.
- Python orchestration (`scripts/cadofactor/`): `fractions.gcd`→`math.gcd` (cadotask.py, removed in 3.9); `collections.X`→`collections.abc.X` (wudb.py, removed in 3.10).
- HTTPS work-unit server/client (`wuserver.py` + `cado-nfs-client.py`): server cert bumped 1024→2048-bit and `PROTOCOL_SSLv23`→`PROTOCOL_TLS_SERVER` (OpenSSL 3.x rejects small keys / deprecated protocol); client `urlopen(..., cafile=)`→`urlopen(..., context=ssl.create_default_context(cafile=...))` (the `cafile=` parameter was removed in Python 3.12) and now honours `NO_CN_CHECK`; `FixedHTTPServer.handle_error` added to swallow the benign cert-download-probe disconnects. The dormant `FixedSSLSocket` block (guarded to Python 3.2–3.3) is inert on 3.14.

**Running locally:** SSL now works on Python 3.14 — the default `./cado-nfs.py <N>` runs over HTTPS end-to-end (and multi-machine mode should too). `server.ssl=no` still works if you want plain HTTP but is no longer required. **Arg-order quirk:** `key=value` params must come *before* `-t`/other flags (`./cado-nfs.py <N> server.ssl=no -t 4`) — the `options` positional is `nargs="*"` and argparse won't fill it from a group after an optional-with-value. Verified: a 59-digit factorization completes in ~30 s over HTTPS with a clean log and the product checks out.

Detailed docs: `@README` (full build/run/distributed guide), `@README.dlp` (discrete log), `@README.Python` (orchestration internals).
