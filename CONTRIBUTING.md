# Contributing

Thanks for your interest in this project. Please read this first — it explains
what belongs **here** versus **upstream**.

## Scope of this fork

`doublegate/cado-nfs-2.3.1-modern` is a narrow **modernization fork** of
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) 2.3.0. Its only goal is to
let the 2.3.0 codebase build and run on current toolchains (CMake 4, GCC 16,
hwloc 2, OpenSSL 3, Python 3.14). It does **not** add features or change the
algorithms.

- **Algorithm work, new features, numerical improvements, performance work, or
  general bugfixes** belong **upstream**, where they benefit everyone and where
  the current maintainers can review them:
  <https://gitlab.inria.fr/cado-nfs/cado-nfs>.
- **This repo accepts** fixes that further the modernization goal: additional
  toolchain/compiler/OS-version compatibility, build fixes, Python-stdlib
  deprecation fixes, packaging, CI, and documentation of the above.

If you are unsure where a change belongs, open an issue here and ask.

## How to contribute here

1. Fork and create a feature branch (`git checkout -b fix/<short-desc>`).
2. Keep changes **minimal and surgical** — match the surrounding upstream style;
   do **not** bulk-reformat upstream source. There is intentionally no
   `clang-format` config.
3. Prefer version-guarded compatibility shims (e.g.
   `#if HWLOC_API_VERSION >= 0x00020000`, `try/except ImportError`) so older
   toolchains keep working.
4. Build and test:
   ```bash
   make
   make check ARGS="-R '<relevant-subsystem-regex>'"
   ./cado-nfs.py 90377629292003121684002147101760858109247336549001090677693 -t 4
   ```
   See [`CLAUDE.md`](CLAUDE.md) for the subsystem → test-pattern map.
5. Use [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `build:`, `docs:`, `chore:`…) and explain the *why*.
6. Update [`CHANGELOG.md`](CHANGELOG.md) under the unreleased/next section.
7. Open a pull request describing the toolchain/version the change targets and
   how you verified it.

## Reporting issues

Use the issue templates. For build or runtime failures, please include your OS,
compiler, CMake/Python/GMP/hwloc/OpenSSL versions, and the first error verbatim.

## License

By contributing, you agree that your contributions are licensed under the
project's **LGPL-2.1** license (see [`COPYING`](COPYING)), consistent with
upstream CADO-NFS.
