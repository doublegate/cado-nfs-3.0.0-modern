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

So CADO's production `bucket` is ~1.8× the naive loop single-threaded. And it
**barely scales with cores**, because SpMV is memory-bandwidth-bound — measured
with `bench_matcache --nthreads N` (one ~6M-nnz submatrix per thread) on the
i9-10850K (10C/20T):

| threads | aggregate Gnz/s | vs 1 thread |
|---:|---:|---:|
| 1 | 0.85 | 1.0× |
| 2 | 1.26 | 1.5× |
| 4 | 1.51 | 1.8× |
| 8 | 1.59 | 1.9× |
| 16 | 1.81 | 2.1× |
| **20** | **1.81** | **2.1×** |

**Full-CPU `bucket` saturates at ~1.8 Gnz/s** (only ~2.1× the single thread —
bandwidth-bound by ~N=4). Against that, the GPU b64 kernel at **7.9 Gnz/s is a
measured ~4.4× the full production CPU** — a real single-machine win, well below
the inflated "6–15× vs naive" but solidly above bandwidth parity. And it is a
*floor*: the un-tuned kernel realizes only ~10% of the 3090's ~936 GB/s, so a
coalesced kernel should widen the gap.

**The GPU's biggest advantage is still at *scale*:** aggregate bandwidth across
many GPUs/nodes and matrices too large for one machine's RAM (the multi-GPU/MPI
path below) — but even on one desktop the kernel is ~4× the tuned CPU backend.

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

## The `matmul_bNN_gpu` backend (implemented + validated)

`linalg/bwc/matmul-gpu.cu` is a real GPU backend plugged into BWC's matmul
dispatch — selected with `mm_impl=gpu`, registered through the same
`CONFIGURE_MATMUL_LIB` / `COOKED_BWC_BACKENDS` machinery as `basic`/`bucket`,
built only with `-DENABLE_GPU=ON` (compiled by nvcc; `b64` and `b128`). It mirrors
`matmul-basic`'s cache format, keeps **both the matrix and its transpose resident
on the device as CSR** (so both BWC directions are fast one-thread-per-row
gathers), and copies only the `src`/`dst` vectors per call.

- **Bit-exact:** passes `bench_matcache`'s consistency check
  (`(M·v₁)·v₂ == (Mᵀ·v₂)·v₁`) — **all 4 checks pass** for both directions on the
  real 1M×1M matrix.
- **End-to-end: validated in a real factorization.** A full
  `cado-nfs.py … tasks.linalg.bwc.mm_impl=gpu` run (59-digit, `thr=2x2`) drives
  the entire BWC pipeline (krylov → lingen → mksol → gather, balanced matrix,
  multi-thread comm) through the GPU backend and returns the correct factors
  (**product == N**) — confirmed by the `mm_impl=gpu` bwc command line and the
  per-instance `matmul-gpu split` timing in the bwc logs. The backend is
  production-usable, not just a bench.
- **Measured (real backend, incl. per-call vector transfers):** BWC reuses the
  same host vector buffers every iteration, so the backend **page-locks (pins)
  each one the first time it sees it** — all subsequent H2D/D2H transfers then run
  at full PCIe bandwidth. That lifted it from 4.95 → **6.76 Gnz/s** on the RTX
  3090 — **~11× a single `bucket` thread** and **~3.75× the full-CPU `bucket`**
  (1.8 Gnz/s). The residual ~1 ms/iter is the unavoidable `src`+`dst` (~16 MB)
  still crossing PCIe each call; only **full vector residency** (vectors never
  leaving the GPU, exchanged only across MPI — an `mmt_vec`-layer change above the
  backend) removes it, toward the ~7.9 Gnz/s kernel-only ceiling.

## Where the time goes (measured), and the residency decision

Split-timed in the backend (`CADO_GPU_TIMING=1`, pinned, b64, 30M nnz, RTX 3090):

| segment | per-SpMV | share |
|---|---:|---:|
| H2D `src` | 1.81 ms | 36% |
| **kernel** | 2.63 ms | 52% |
| D2H `dst` | 0.64 ms | 12% |
| — transfers total | 2.45 ms | **48%** |

So even after pinning, **~half the time is the per-call `src`/`dst` transfers** —
full vector residency is a genuine **~2× opportunity** (~13 Gnz/s → ~7× the full
CPU). The catch is *what* it takes, established by reading the BWC vector layer:

- **Every SpMV is bracketed by host-memory work that can't be skipped at the
  backend level.** After each `matmul_top_mul_cpu` (the `mm->mul` seam,
  `matmul_top.cpp:762`) comes `mmt_vec_allreduce` / `matmul_top_mul_comm`
  (`matmul_top_comm.cpp`), whose reduce/broadcast call `MPI_Allreduce` /
  `MPI_Reduce_scatter_block` / `MPI_Allgather` **directly on `vec->v` host
  memory** and do thread-local `vec_add_and_reduce`/`vec_set` on host. And every
  inner iteration of `krylov.cpp` calls `x_dotprod(... ymy[0] ...)` which **reads
  the vector on the host** before the SpMV.
- Therefore a device-resident vector must, every iteration, be on the host for
  the dot-product and for the comm — so **transfers can only be removed by moving
  the dot-product, the thread reduction, the broadcast, and (for >1 node) the MPI
  exchange onto the GPU too.** That is a port of BWC's vector layer, not a backend
  tweak, and any missed sync silently corrupts the result.

**Decision:** this is worth ~2× but is a deliberate multi-step project, not a safe
single increment. The honest sequence:

1. **A device-resident `mmt_vec` shadow** + `matmul_interface` sync hooks
   (`to_device`/`from_device`), backend keeps the authoritative device copy.
2. **GPU `x_dotprod`** (sparse `x` · device vector) so the per-iteration dot
   product no longer pulls the vector to the host. **Kernel built + validated
   bit-exact** (`bench/gpu-xdotprod-bench.cu`, ALL PASS) — ready to wire in.
3. **GPU intra-node reduction/broadcast** for the single-node multi-thread comm
   (the common single-machine case), keeping vectors on-device across the inner
   loop; sync to host only per-interval (twist/save) and for real MPI. **Kernels
   built + validated bit-exact** (`bench/gpu-vecreduce-bench.cu`: GF(2) XOR-reduce
   + broadcast, ALL PASS, ~440 GB/s) — ready to wire in.
4. **Multi-node**: GPU-direct (CUDA-aware) MPI, or host-staged exchange.

Approach: each device-side primitive is built and validated **standalone first**
(no risk to real runs), then wired into the resident loop, with a full verified
factorization (`product == N`) as the gate at integration. **All three primitives
are ready and bit-exact**: the coalesced SpMV kernel, `x_dotprod`, and the
intra-node reduce/broadcast.

### Wiring — the device-residency control plane (done, `product == N`)

`matmul_interface` gains a `host_vector_modified(hostptr)` hook (no-op for CPU
backends). The GPU backend keeps a **persistent per-host-vector device buffer
pool** with a `current` flag; with `CADO_GPU_VECRESIDENT=1` a current buffer lets
`mul()` skip the H2D upload, and `host_vector_modified` clears the flag.
`matmul_top` calls the hook at every host write — after the comm
(`mmt_vec_allreduce` / `matmul_top_mul_comm`) and after `mmt_vec_twist/untwist`.
**Both modes verified by a real factorization** (`tasks.linalg.bwc.mm_impl=gpu`,
59-digit): default mode is bit-identical, and **resident-skip mode also returns
`product == N`** — proving the invalidation coverage is complete for the krylov
path (no missed host write).

But the skip is **correct yet inert**: the comm runs after *every* SpMV and
invalidates the src, so it is always re-uploaded — the H2D share does not drop.
This empirically confirms the analysis: **the win requires moving the comm itself
onto the device** (using the validated reduce/broadcast kernel), so the vector
stays device-authoritative through the comm and the (already-proven-correct) skip
finally triggers. That is the next, now-localized step. The remaining pieces are
then the per-interval host syncs (twist/save) and GPU `x_dotprod`, all on a
control plane whose correctness is already gated by `product == N`.

Each step is correctness-gated by a full verified factorization. Until then, the
backend captures the bounded win (pinning) and is bit-exact.

## Kernel tuning — done (coalesced warp-per-row)

The one-thread-per-row kernel was uncoalesced. A **warp-per-row** kernel — lanes
stride the row's nonzeros (so `col[]` reads coalesce), `src` gathered through the
read-only cache (`__ldg`), then the K-limb accumulator warp-reduced — is
**bit-exact** (validated in `bench/gpu-spmv-bench.cu`: warp PASS, 0 words; and the
backend still passes `bench_matcache` 4/4) and **1.8–3.1× faster** standalone
(b64 6.9→12.6 Gnz/s, b256 3.9→12.3). In the backend it cut the kernel from
2.63 → **0.97 ms** (2.7×), lifting the end-to-end SpMV to **8.96 Gnz/s (~5× the
full-CPU `bucket`)**.

Crucially this **flips the bottleneck**: the split is now H2D 1.81 ms / kernel
0.97 ms / D2H 0.62 ms — **72% transfers**. So the residency port above is now the
clearly dominant remaining single-machine win (~3× more headroom), and further
kernel tuning (ELL, column sorting, shared-mem `src`) is secondary.
3. **Multi-GPU / multi-node**: BWC already splits the matrix across an `nh×nv`
   MPI grid (`balancing_workhorse`), each rank owning a submatrix; the GPU backend
   slots in at each rank's local `mm->mul()`. One GPU per rank → multi-GPU on a
   node and multi-node via the unchanged MPI comm layer — the natural HPC
   scale-out and where the GPU's aggregate-bandwidth advantage compounds.

## Status

- **Done & validated:**
  - the GF(2) SpMV GPU kernel (b64/b128/b256), bit-exact vs the CPU reference
    (`bench/gpu-spmv-bench.cu`);
  - the honest full-CPU baseline — `bucket` saturates at ~1.8 Gnz/s (20 threads,
    bandwidth-bound), so the kernel-only GPU win is ~4.4×;
  - **a real `matmul_bNN_gpu` backend** (`linalg/bwc/matmul-gpu.cu`) plugged into
    BWC's dispatch (`mm_impl=gpu`), passing `bench_matcache`'s bit-exact check
    (4/4, both directions), with **pinned host-vector transfers** and a
    **coalesced warp-per-row kernel** — **8.96 Gnz/s, ~5× the full-CPU `bucket`**
    (kernel cut 2.7×; the split is now 72% transfers / 28% kernel).
- **Next:** full vector residency (the now-dominant 72% transfer share — a
  multi-step vector-layer port, scoped above), then multi-GPU/MPI wiring where the
  GPU's aggregate-bandwidth advantage at scale lives.
