# bench/harness — measurement harnesses

Reusable driver scripts behind the numbers in [`../../BENCHMARKS.md`](../../BENCHMARKS.md),
preserved from the 3.1.0/3.2.0-modern development. Each rebuilds the relevant
bench(es) from `bench/*.cu` / `bench/*.c` (or drives `cado-nfs.py`) and prints
results; the per-kernel correctness validators live one level up in
[`../`](..) (`vpclmul-validate.sh`, `ifma-validate.sh`, `avx512-modinv-validate.sh`,
`las-microbench.sh`).

Each script derives the repo root from its own location (`$REPO`) and uses
`build/$(hostname)` for compiled CADO binaries; **scratch output defaults to
`/tmp/cado-nfs-bench/`** (created on demand — edit the path or `cd` there first).
Run them from anywhere; the venv (`scripts/setup-venv.sh`) is needed for the
`cado-nfs.py` ones.

| Script | What it measures | Backs | Needs |
|--------|------------------|-------|-------|
| `collect-v320.sh` | Rebuilds + runs the **3.2.0 GPU/SIMD additions** (A2 Edwards ECM, C1 SpMV, C3 batch-smooth, C4 sieve-scatter; B1/B2/B3 under SDE) into `results-v320.txt` | BENCHMARKS §5–6 | nvcc, GMP, Intel SDE |
| `gpu-standalone.sh` | Runs the standalone GPU + AVX-512 benches (GPU pre-factor ECM, the b64 SpMV scaling sweep, VPCLMULQDQ `mul1`, IFMA modmul) | BENCHMARKS §3–5 | nvcc, GMP, Intel SDE |
| `cpu-sweep.sh` | End-to-end **CPU factorization sweep** c60→c90 (the seeded inputs), wall + cpu/elapsed + per-phase grep | BENCHMARKS §1 | the venv; default build |
| `parse.sh` | Parses the `cpu-sweep.sh` logs into per-phase CPU-seconds (poly/sieve/filter/LA/sqrt) | BENCHMARKS §1 per-phase table | python3 |
| `iso.sh` | **Polyselect GPU-vs-CPU isolation** (huge `P`, tiny `admax` → root-finding dominates), WCT per path | C2 (`docs/gpu-polyselect.md`) | a `-DENABLE_GPU=ON` build |
| `hd.sh` | **Polyselect GPU-vs-CPU correctness + timing** — asserts the GPU path produces a *byte-identical* polynomial set | C2 (`docs/gpu-polyselect.md`) | a `-DENABLE_GPU=ON` build |
| `hdrtest.c` / `hdrtest_pclmul.c` | Compile the **integrated gf2x VPCLMULQDQ headers** (`gf2x_mul2/3/4.h`) — with `GF2X_HAVE_VPCLMUL_SUPPORT` set, and the PCLMUL `#else` fallback — against a scalar GF(2)[x] reference (B2) | B2 (`gf2x/already_tuned/x86_64_vpclmul/INTEGRATION.md`) | gcc, Intel SDE; a stub `gf2x.h`/`gf2x-small.h` on the include path (see `bench/vpclmul-muln.c` for the standalone-kernel equivalent) |

The standalone kernel sources these drive are the canonical, committed ones in
[`../`](..) (`gpu-ecm-edwards.cu`, `gpu-spmv-bench.cu`, `gpu-batch-smooth.cu`,
`gpu-sieve-scatter.cu`, `vpclmul-muln.c`, `ifma-gfp.c`, `avx512-modinv.c`, …).
Superseded development variants (e.g. the SpMV `spmv-vec`/`spmv-locality`
column-reordering experiments, whose conclusion is recorded in
`docs/gpu-linalg.md`) were intentionally not preserved — regenerate from the
committed canonical benches if needed.
