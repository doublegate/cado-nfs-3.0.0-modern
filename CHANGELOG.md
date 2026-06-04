# Changelog

All notable changes to this fork are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
loosely follows [Semantic Versioning](https://semver.org/).

This is a downstream **modernization fork** of upstream
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) 2.3.0. Only the changes
introduced by this fork are listed; for the upstream history up to 2.3.0 see
[`NEWS`](NEWS).

## [2.3.1-modern] — 2026-06-04

First release of the modernization fork. Upstream CADO-NFS 2.3.0 (2017) does
not build or run unmodified on a current toolchain. This release makes the
2.3.0 codebase build cleanly and factor numbers end-to-end on
**CMake 4.x, GCC 16, hwloc 2.x, OpenSSL 3.x, and Python 3.14**, with no change
to the underlying algorithms.

### Build system

- **CMake 4.x compatibility.** CMake 4 removed compatibility with policies
  below 3.5, but CADO-NFS declares `cmake_minimum_required(VERSION 2.8.11)`.
  `local.sh` now passes `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` via
  `CMAKE_EXTRA_ARGS` so configuration succeeds. (Harmless on older CMake.)
- **GCC 10+ `-fno-common`.** Modern GCC defaults to `-fno-common`, turning the
  project's tentative global definitions (e.g. `bw` in `linalg/bwc`) into
  multiple-definition link errors. `local.sh` now sets `CFLAGS="-O2 -fcommon"`.
- **gf2x ISO C90 generator.** `gf2x/lowlevel/gen_bb_mul_code.c` is compiled by
  gf2x's *build-system* compiler in ISO C90, where `//` comments are illegal.
  One dead-code `//` comment was converted to `/* … */`.
- **Version bumped** `2.3.0` → `2.3.1` (`CMakeLists.txt`).
- `local.sh` is now committed (it carries only portable, machine-independent
  build flags) so the tree builds out-of-the-box.

### C/C++

- **hwloc 1.x → 2.x port** in `linalg/bwc/cpubinding.cpp`. The removed
  `HWLOC_TOPOLOGY_FLAG_IO_DEVICES` / `HWLOC_TOPOLOGY_FLAG_IO_BRIDGES` topology
  flags are replaced by `hwloc_topology_set_io_types_filter(topology,
  HWLOC_TYPE_FILTER_KEEP_NONE)`, version-guarded with
  `#if HWLOC_API_VERSION >= 0x00020000` so hwloc 1.x still compiles. Behaviour
  is preserved: I/O devices are excluded from the topology, which is also the
  hwloc-2.x default.

### Python orchestration (`scripts/cadofactor/`, `cado-nfs-client.py`)

- **`fractions.gcd` → `math.gcd`** (`cadotask.py`); `fractions.gcd` was removed
  in Python 3.9. Guarded with a fallback import for very old Pythons.
- **`collections.*` ABCs → `collections.abc.*`** (`wudb.py`); the
  `collections.MutableMapping` / `Mapping` / `Container` aliases were removed in
  Python 3.10.
- **HTTPS work-unit server** (`wuserver.py`):
  - Server certificate key size raised 1024 → 2048 bits; OpenSSL 3.x rejects
    keys below 2048 (`EE_KEY_TOO_SMALL`).
  - `ssl.PROTOCOL_SSLv23` → `ssl.PROTOCOL_TLS_SERVER` (the former is deprecated).
  - Added `FixedHTTPServer.handle_error` to swallow the benign
    connection-teardown from the client's certificate-download probe, which
    previously printed an alarming (but harmless) `BrokenPipeError` traceback.
- **HTTPS work-unit client** (`cado-nfs-client.py`):
  - `urllib.request.urlopen(..., cafile=…)` → `urlopen(..., context=
    ssl.create_default_context(cafile=…))`; the `cafile`/`capath`/`cadefault`
    parameters were removed from `urlopen()` in Python 3.12. This was the
    failure that left clients unable to fetch work units and the run hung.
  - The `NO_CN_CHECK` setting (skip hostname verification) is now honoured on
    Python 3 instead of raising "not implemented".

### Verification

- Full build completes (100%, exit 0) on the reference machine
  (CachyOS, GCC 16.1.1, GMP 6.3.0, hwloc 2.13.0, OpenSSL 3.x, Python 3.14.5).
- A 59-digit demo factorization
  (`90377629292003121684002147101760858109247336549001090677693`) completes in
  ~30 s **over HTTPS**, producing four 15-digit primes whose product equals the
  input; the Linear-Algebra (Block-Wiedemann) phase exercises the ported
  `cpubinding.cpp`.
- Targeted unit-test subsets (`test_bitlinalg*`, `sievetest*`) pass.

### Not changed

- No algorithmic, numerical, or parameter changes — this fork is a portability
  layer only.
- Upstream source style is preserved; no bulk reformatting.
- Multi-machine/distributed mode is unchanged in spirit; only the shared SSL
  layer was modernized (now exercised successfully on localhost over HTTPS).

[2.3.1-modern]: https://github.com/doublegate/cado-nfs-2.3.1-modern/releases/tag/v2.3.1-modern
