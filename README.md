# CADO-NFS 2.3.1-modern

A **modernization fork** of [CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs)
2.3.0 — a complete implementation in C / C++ / Python of the **Number Field
Sieve (NFS)** algorithm for integer factorization and discrete logarithms in
finite fields.

> **What this fork is.** Upstream CADO-NFS 2.3.0 was released in 2017 and does
> not build or run unmodified on a current toolchain. This fork applies a thin
> portability layer so the 2.3.0 codebase builds cleanly and factors numbers
> end-to-end on **CMake 4.x, GCC 16, hwloc 2.x, OpenSSL 3.x, and Python 3.14** —
> with **no changes to the algorithms, numerics, or parameters**. See
> [`CHANGELOG.md`](CHANGELOG.md) for the exact patch list.
>
> This is **not** the official CADO-NFS. For releases, ongoing development, and
> support, use the upstream project (links below).

[![CI](https://github.com/doublegate/cado-nfs-2.3.1-modern/actions/workflows/ci.yml/badge.svg)](https://github.com/doublegate/cado-nfs-2.3.1-modern/actions/workflows/ci.yml)
[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL%202.1-blue.svg)](COPYING)

## Quick start

### Dependencies

- A C/C++ compiler (GCC ≥ 4.7 or compatible Clang; tested through GCC 16)
- [GMP](https://gmplib.org/) ≥ 5, **built with `--enable-shared`**
- CMake (any 2.8.11+; CMake 4.x works via the bundled `local.sh` flag)
- Python 3 with the `sqlite3` module
- Optional: `hwloc` (CPU pinning), MPI, `curl`

On Debian/Ubuntu:

```bash
sudo apt-get install -y build-essential cmake libgmp-dev libhwloc-dev python3
```

### Build

```bash
make            # out-of-tree build into build/<hostname>/
make check      # run the test suite (ctest); use ARGS="-R <regex>" to filter
```

The portable modern-toolchain flags live in the committed
[`local.sh`](local.sh) (`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for CMake 4 and
`-fcommon` for GCC 10+). Edit `local.sh` then `make cmake` to reconfigure.

### Factor a number

```bash
./cado-nfs.py 90377629292003121684002147101760858109247336549001090677693 -t 4
```

This runs the full pipeline (polynomial selection → sieving → filtering →
linear algebra → square root) over a local HTTPS work-unit server and prints
the prime factors. On a modern desktop the 59-digit demo finishes in ~30 s.
CADO-NFS targets numbers **> 85 digits**; **< 60 digits is unsupported**, and
you should strip small factors first.

> **CLI note:** `key=value` parameters must come *before* flags like `-t`
> (e.g. `./cado-nfs.py <N> server.ssl=no -t 4`). Pass `server.ssl=no` to use
> plain HTTP instead of TLS (optional).

For larger and distributed factorizations, discrete logarithms, and full
parameter documentation, see the upstream guides preserved in this tree:
[`README`](README), [`README.dlp`](README.dlp), and [`README.Python`](README.Python).

## What changed in this fork

| Area | Change | Why |
|------|--------|-----|
| Build | `local.sh`: `CMAKE_POLICY_VERSION_MINIMUM=3.5`, `-fcommon` | CMake 4.x dropped pre-3.5 policies; GCC 10+ defaults to `-fno-common` |
| Build | `gf2x/lowlevel/gen_bb_mul_code.c`: `//` → `/* */` | gf2x build compiler runs ISO C90 |
| C++ | `linalg/bwc/cpubinding.cpp`: hwloc 1.x → 2.x API | hwloc 2.0 removed the I/O topology flags |
| Python | `math.gcd`, `collections.abc.*` | removed from stdlib in 3.9 / 3.10 |
| Python | TLS server/client modernized (2048-bit cert, `PROTOCOL_TLS_SERVER`, `urlopen(context=)`) | OpenSSL 3.x + `urlopen(cafile=)` removed in 3.12 |

Full details with rationale: [`CHANGELOG.md`](CHANGELOG.md). The repository also
includes a [`CLAUDE.md`](CLAUDE.md) with build/test/run notes and the complete
modernization record.

## License

CADO-NFS is free software under the **GNU Lesser General Public License,
version 2.1** — see [`COPYING`](COPYING). This fork preserves that license and
all upstream copyright and authorship; the original authors are credited in
[`AUTHORS`](AUTHORS). The modifications described above are released under the
same LGPL-2.1 terms.

## Credits and attribution

- **Original work:** the CADO-NFS development team (INRIA / LORIA, Nancy,
  France) — Shi Bai, Cyril Bouvier, Pierrick Gaudry, Emmanuel Thomé,
  Paul Zimmermann, and many others (see [`AUTHORS`](AUTHORS)).
- **Upstream project:** <https://gitlab.inria.fr/cado-nfs/cado-nfs> ·
  homepage <http://cado-nfs.inria.fr/> · GitHub mirror
  <https://github.com/cado-nfs/cado-nfs>
- **Citation:** CADO-NFS, An Implementation of the Number Field Sieve Algorithm,
  Release 2.3.0 (2017). See [`AUTHORS`](AUTHORS) for the BibTeX entry.
- **This modernization fork:** maintained by [@doublegate](https://github.com/doublegate).

If you use CADO-NFS in academic work, please cite the **upstream** project, not
this fork.
