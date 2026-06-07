# bench/results — captured benchmark output

Raw results behind [`../../BENCHMARKS.md`](../../BENCHMARKS.md), kept for
reference/provenance. **Machine:** Intel Core i9-10850K (10C/20T) · 64 GiB DDR4 ·
NVIDIA RTX 3090 (sm_86) · CachyOS / Linux 7.0 · GCC 16.1.1 · CUDA 13.3 · GMP
6.3.0 · Intel SDE 10.8.0. Regenerate with the drivers in
[`../harness/`](../harness).

| File | What | Driver | Date |
|------|------|--------|------|
| `v3.2.0-benchmarks.txt` | The 3.2.0 GPU/SIMD additions — A2 GPU mixed-rep ECM (Edwards vs ladder, bit-exact), C1 adaptive SpMV, C3 batch-smoothness leaf, C4 sieve-scatter probe, and the B1/B2/B3 AVX-512 kernels under Intel SDE (all PASS) | `harness/collect-v320.sh` | 2026-06-07 |
| `cpu-factorization-sweep.txt` | End-to-end CPU factorization c60→c90 (seeded inputs) — wall time, cpu/elapsed, verified factors | `harness/cpu-sweep.sh` | 2026-06 |

These are point-in-time snapshots; NFS wall time has ~±15–20 % run-to-run variance
(randomized polynomial selection), and the GPU small-batch throughput numbers vary
with clock/scheduling — treat them as representative, not exact. The curated
tables in `BENCHMARKS.md` are the canonical reference.
