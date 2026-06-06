# GPU linear algebra (Block Wiedemann SpMV) — v3.1.0-modern, Track 2.2

The linear-algebra (matrix) step is the **fastest-growing** phase of NFS: in the
3.0.0-modern benchmarks its CPU time grows ~110× from c60 to c90 (vs ~60× for
sieving; see `BENCHMARKS.md`), so it becomes the second bottleneck as numbers
grow and is the natural target for new compute. Its kernel is a **sparse
matrix × block-of-vectors product (SpMV)** over GF(2), run thousands of times by
Block Wiedemann (`linalg/bwc`). This track puts that kernel on the GPU.

## The operation (matched bit-exact)

From `linalg/bwc/matmul-basic.cpp`, the SpMV for a non-transposed matrix×vector
is, per output row `i`, an XOR-accumulate over that row's nonzero columns:

```
dst[i] = XOR over { j : (i,j) nonzero }  of  src[j]
```

Each vector element is a **bitsliced block of K 64-bit limbs** — `b64` (K=1, 64
vectors at once) or `b128` (K=2, 128 vectors). The matrix is rows of column
indices (implicit 1 coefficients over GF(2)); `matmul-basic`'s `q` array is
`[len₀, col, …, len₁, col, …]`, equivalent to CSR (`rowptr`, `col`).

## Validated GPU kernel + benchmark

`bench/gpu-spmv-bench.cu` implements the GPU SpMV (one thread per output row,
XOR-accumulating `src[col]` over the row, K limbs per element) and **validates it
bit-exact** against the same CPU loop, then benchmarks both (CPU parallelized
across all cores). The matrix stays resident on the GPU (BWC reuses it across
thousands of iterations), so only the kernel is timed. On an RTX 3090 vs a
20-thread i9-10850K, synthetic matrices (~30 nonzeros/row):

| Block | Matrix | Validation | GPU | CPU (20 thr) | Speedup |
|---|---|---|---:|---:|---:|
| b64 (64 vec) | 2.0 M rows, 60 M nnz | **PASS** | 7.9 Gnz/s | 1.27 Gnz/s | 6.2× |
| b128 (128 vec) | 2.0 M rows, 60 M nnz | **PASS** | 5.1 Gnz/s | 0.33 Gnz/s | 15.4× |
| b256 (256 vec) | 0.5 M rows, 30 M nnz | **PASS** | 5.2 Gnz/s | 1.09 Gnz/s | 4.8× |

```bash
nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-spmv-bench.cu -o gpu-spmv-bench && ./gpu-spmv-bench
```

## Measured against CADO's real `bucket` backend (the honest comparison)

The table above compares the GPU to the **naive `matmul-basic` loop**, which
overstates the win. Measured directly with `bench_matcache` on a **real
1M×1M GF(2) matrix (30M nonzeros)** from `random_matrix`, **single CPU thread**:

| backend | ns/coeff | Gnz/s | vs naive |
|---|---:|---:|---:|
| `basic` (naive loop) | ~2.8 | 0.35 | 1.0× |
| `sliced` | ~1.5 | 0.67 | 1.9× |
| **`bucket` (production default)** | ~1.58 | **0.63** | **1.8×** |

So CADO's production `bucket` is **~1.8× the naive loop**, and the GPU b64 kernel
(7.9 Gnz/s) is **~12× a single `bucket` thread**. But the full 20-core CPU runs
`bucket` across all cores, and SpMV is **memory-bandwidth-bound** — the
i9-10850K's ~45 GB/s won't scale 20×, realistically reaching a few Gnz/s. **So
the honest single-machine win of this GPU kernel over the full production CPU is
modest — roughly 1.5–3×, not 6–15× — because both are bandwidth-bound** (the 3090
has ~20× the raw bandwidth, but the un-tuned kernel realizes only ~10% of it, and
`bucket` is cache-blocked to *need* less bandwidth). A rigorous full-CPU `bucket`
number needs a balanced multi-file split (`bench_matcache` threads over one
submatrix file per thread) — that exact measurement is the immediate next step.

**Where the GPU genuinely wins is at *scale*:** aggregate bandwidth across many
GPUs/nodes, and matrices too large for one machine's RAM — the multi-GPU/MPI path
below — not a single-desktop 10× on the matmul kernel.

## Other caveats

- **Synthetic matrix** for the GPU table: random CSR has worse locality than a
  real filtered/balanced BWC matrix (columns reordered for cache reuse). The real
  comparison can move either way; the `bucket` numbers above *are* on a real
  matrix.
- **Kernel un-tuned.** One-thread-per-row hits ~10% of the 3090's ~936 GB/s peak
  (uncoalesced `src[col]` gather). ELL/sliced formats, column sorting for reuse,
  and shared-memory `src` caching are known wins not yet applied — GPU headroom.
- **Memory.** The matrix must fit in GPU memory (24 GB on a 3090 → roughly up to
  ~c150-scale); larger needs the multi-GPU/multi-node path below.

## Integration path (next increments)

1. **A `matmul_bNN_gpu` backend** implementing `matmul_interface`
   (`build_cache`/`reload_cache`/`mul`) behind `matmul_interface::create`
   (`linalg/bwc/matmul.cpp` dispatch), keeping the CSR matrix **resident on the
   device** and copying only the `src`/`dst` vectors per iteration (or keeping
   them resident too and exchanging only across MPI). Selected with
   `mm_impl=gpu`.
2. **Bit-exact gate** via the existing `bench_matcache` check
   ((M·v₁)·v₂ == (Mᵀ·v₂)·v₁) plus a real-matrix run vs `bucket`.
3. **Multi-GPU / multi-node**: BWC already splits the matrix across an `nh×nv`
   MPI grid (`balancing_workhorse`), each rank owning a submatrix. The GPU
   backend slots in at each rank's local `mm->mul()`; one GPU per rank gives
   multi-GPU on a node and multi-node via the unchanged MPI comm layer — the
   natural HPC scale-out.

## Status

- **Done & validated:** the GF(2) SpMV GPU kernel (b64/b128/b256), bit-exact vs
  the CPU reference (`bench/gpu-spmv-bench.cu`); and the honest comparison vs
  CADO's real `basic`/`sliced`/`bucket` backends on a real matrix
  (`bench_matcache`), which puts the single-machine win at a sober ~1.5–3× (not
  6–15×) — both bandwidth-bound.
- **Next:** a full-CPU threaded `bucket` measurement (balanced split); then the
  `matmul_bNN_gpu` backend (resident matrix, ELL/coalesced kernel) + the
  multi-GPU/MPI wiring, where the GPU's real advantage (aggregate bandwidth at
  scale, out-of-core matrices) lives.
