# Changelog

All notable changes to this fork are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
loosely follows [Semantic Versioning](https://semver.org/).

This is a downstream **modernization + performance fork** of upstream
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs). The `3.0.x-modern` line is
rebased onto upstream **3.0.0**; only the changes introduced by this fork are
listed. For the upstream history see [`NEWS`](NEWS). The earlier `2.3.1-modern`
release (rebased on upstream 2.3.0) is preserved under the `v2.3.1-modern` tag;
`main` now tracks the latest release (`3.1.0-modern`).

## [3.2.0-modern] — Unreleased

Development cycle opened after `3.1.0-modern`. Plan in
[`docs/ROADMAP-v3.2.0-modern.md`](docs/ROADMAP-v3.2.0-modern.md), grounded in
current literature and the strategic reframe that **sieving (~91 % of RSA-250's
cost) and polynomial selection** — not linear algebra (~9 %) — are where the
leverage is. Targets: a faster GPU SpMV kernel (ELL/hybrid), GPU polynomial
selection, 3-D lattice sieving, real multi-GPU/multi-node linear algebra
(`CADO_GPU_NPART` + NVSHMEM), AVX-512 block/bucket sieving, mixed-representation
ECM, and an autotuner + planner + web dashboard. Same ethos: `product == N` /
bit-exact gate on every change, honest negatives, HW-gated work shipped as
validated-at-degenerate-path code + design.

- **Version bumped to `3.2.0-modern`** (`CMakeLists.txt` `CADO_VERSION_MINOR
  1 → 2`); roadmap added.

### GPU (C1) — adaptive SpMV kernel (sub-warp CSR-vector)

- **Faster GPU SpMV in the cache-resident regime.** The 3.1.0 warp-per-row kernel
  uses a full 32-lane warp per output row, but BWC rows have only ~30 nonzeros, so
  each lane does ~1 gather then a 5-step warp-reduce — reduce-bound. A new sub-warp
  **CSR-vector** kernel (`spmv_vec<K,VEC>`, VEC=16) puts two rows per warp, halving
  the reduce and raising occupancy. `launch_spmv` now **dispatches adaptively**:
  vec16 when the source vector is L2-resident (`nrows·K·8 ≤ 12 MB`), else the
  latency-hiding warp32 kernel (override with `CADO_GPU_SPMV={vec,warp}`).
- **Measured (RTX 3090, `bench/gpu-spmv-bench.cu`, bit-exact at every size):** in
  the cache-resident regime vec16 is **1.26–1.8× warp** (e.g. 49.5 vs 39.4 Gnz/s
  at 0.8 M rows; up to ~27 vs ~15 at 1 M); at large N (≥ ~2 M rows / wider blocks)
  it is a wash or worse, so the adaptive dispatch keeps warp32 there — **no
  regression**. End-to-end `product == N` on the c59 with both the adaptive (→vec)
  default and a forced `CADO_GPU_SPMV=warp` run.
- **Column reordering for the large-N gather — investigated and rejected
  (honest negative).** Measured the locality headroom directly (8 M rows, RTX
  3090): random columns 8.1 Gnz/s, a loose band only **1.1×**, a tight band 1.7×,
  within-row sorting **~2 %**. NFS matrices' skewed degree distribution (dense
  small-prime columns) can't be packed into tight bands, so the realistic gain
  (~1.1×) doesn't justify a global symmetric reorder + the consistent `mmt_vec`
  permutation it would need in the validated matmul layer (and CADO balancing
  already reorders). The large-N SpMV is gather-latency-bound; the better levers
  are the adaptive vec kernel, residency, and multi-GPU (`docs/gpu-linalg.md`).

### GPU (C2) — polynomial selection (profile + foundation kernel)

- **Profiled CADO polyselect stage-1** (the GPU target per the v3.2.0 reframe;
  proven GPU-friendly in msieve since 2009, absent in CADO). `perf` puts the hot
  self-time in **per-prime modular root finding** — `modredcul_intinv` (16 %, the
  modular inverse), `modul_poly_div_r`/`xpowmod` (root finding of `f mod p`,
  ~11 %), plus `L2_skewness`/`double_poly_compute_roots` (size scoring, ~15 %) and
  the collision hash (~10 %). Root finding over thousands of independent primes is
  the GPU sweet spot.
- **Foundation kernels, bit-exact:**
  - *modular inverse* (`bench/gpu-polyselect-modinv.cu`): GPU batched single-word
    modular inverse (the 16 % hottest leaf), bit-exact vs GMP over 200 000 (a, p)
    pairs (0 wrong; 469 M inv/s on an RTX 3090).
  - *per-prime root finding* (`bench/gpu-polyselect-roots.cu`): roots of a degree-d
    polynomial mod a batch of primes, one thread/prime by direct Horner evaluation
    over F_p — exactly correct by construction, validated bit-exact vs a CPU
    reference + a self-check (f(r) ≡ 0): 0 mismatch over 5133 primes (deg 6),
    **GPU 45.9 ms vs CPU 20-thread 277.6 ms = 6.0×**. Direct eval is O(p), a win
    only in the small-prime regime.
  - *gcd-based root finding* (`bench/gpu-polyselect-roots-gcd.cu`): the
    asymptotically-better, p-magnitude-independent method — `h = x^p mod f` by
    binary exponentiation (poly multiply mod f), `g = gcd(h − x, f)`, then
    Cantor–Zassenhaus split into linear factors, all over F_p reusing the validated
    modular arithmetic (`__host__ __device__`, identical GPU/CPU code). Validated
    **bit-exact vs direct-eval** (full root multiset): 0 mismatch / 0 self-check-bad
    over 3245 primes (deg 6, p < 30 000); and **5000 primes near 10⁹ in 27.8 ms**
    (0 self-check-bad) where direct-eval's O(p) is infeasible. The production
    root-finder for the full prime range.
- **Collision-search integration — exact target validated + designed.** CADO's
  per-prime step is `roots_mod_uint64(Ñ mod p, d, p)` = solve `x^d ≡ Ñ mod p` (the
  d-th roots, not an arbitrary polynomial). Re-validated the gcd kernel on that
  exact case (`f = x^d − a`): 0 mismatch / 0 self-check-bad over 3245 primes
  (p < 30 000) + 5000 near 10⁹, so its root set matches `roots_mod_uint64` across
  the full range. `docs/gpu-polyselect.md` now specifies the precise wiring: batch
  `(p, Ñ mod p)` → one GPU launch → `roots_lift` (CPU) + `polyselect_proots_add`,
  leaving the `shash` collision search byte-identical; a `polyselect-gpu.cu` under
  `-DENABLE_GPU=ON` reached via the `matmul-gpu-hooks` pattern; `--gpu-polyselect`
  (default off) gated on matching Murphy-E + `product == N`. The full live wiring
  into the multithreaded proots subtask is the remaining (large) step, scoped
  honestly; the kernels and the exact target are validated.
- **Design + plan** in `docs/gpu-polyselect.md`: the batched per-prime
  root-finding kernel (`gcd(x^p − x, f) mod p`, reusing the validated modinv) →
  feed the `shash` collision search → GPU size scoring → a `--gpu-polyselect`
  flag, gated on matching polynomial quality (Murphy-E) + an end-to-end
  `product == N`. The full module lands incrementally; this is the foundation.

### GPU (C2) — live `--gpu-polyselect` wiring (DONE; correct, gated, honest negative)

- **Full live integration shipped behind a default-off flag.** The validated
  gcd + Cantor–Zassenhaus root-finder is now wired into CADO's real polyselect:
  - `polyselect/polyselect-gpu.cu` — device backend; one thread per prime builds
    `f = x^d − a_i` and solves it mod `p_i`, the host entry batches a thread's
    whole prime range into **one launch** with **persistent per-thread device +
    pinned-host buffers** (no per-ad-value `cudaMalloc` churn).
  - `polyselect/polyselect-gpu-hooks.{h,cpp}` + `polyselect-gpu-stub.cpp` — the
    hook ABI (the `matmul-gpu-hooks` pattern): the `cado_gpu_polyselect_roots`
    pointer lives in `polyselect_common` (dependency-free leaf, no circular
    static-link), `cado_gpu_polyselect_init()` is defined in the `.cu` (GPU build)
    *or* the stub (CPU build) — exactly one is linked.
  - `polyselect/polyselect_proots.cpp` — the injection, gated on
    `CADO_GPU_POLYSELECT` + a non-null hook; gathers `(p_i, Ñ mod p_i)` for the
    thread's range, one device call, then per-prime `roots_lift` +
    `polyselect_proots_add` (byte-identical downstream). Falls back to the
    unchanged per-prime CPU loop on any failure / non-GPU build.
  - `cado-nfs.py --gpu-polyselect` (`toplevel.py`) — sets `CADO_GPU_POLYSELECT=1`
    for every spawned polyselect worker; documented as experimental.
- **Correctness gate — passed.** GPU vs CPU produce a **bit-identical polynomial
  set** (sorted-poly-line diff = 0) at three scales (198 / 136 / 7 kept polys;
  `d=4,5`, `P` up to 250000); device-absent → clean CPU fallback; the 32 ctest
  `polyselect` tests pass; **end-to-end `product == N`** on the 59-digit smoke with
  `--gpu-polyselect`.
- **Performance gate — honest negative.** Measured (i9-10850K + RTX 3090, `d=5`
  `P=50000` `admax=60000` `-t 4`): **CPU 3.2 s vs GPU 4.1 s** — a net slowdown.
  Two fundamental causes, not bugs: (1) **Amdahl** — root-finding is a minority of
  stage-1 (collision search + size-opt stay on CPU), capping the best case at
  ~1.5×; (2) CADO's CPU `roots_mod_uint64` is a *specialised d-th-root* algorithm,
  already so fast that there is no compute to amortise the kernel-launch + PCIe
  round-trip against. Mirrors the 3.0.0 GPU-cofactorization Amdahl finding. The
  win requires offloading the **collision search** (the bulk of stage-1) at much
  larger `N` — the documented next step, for which this validated root-finder and
  hook ABI are the foundation. Full analysis in `docs/gpu-polyselect.md`.

### GPU (C2) — live collision-search offload (DONE; correct, size-gated)

- **The collision search now runs on the GPU behind `--gpu-polyselect`.** A second
  hook (`cado_gpu_polyselect_collisions`, same leaf-TU ABI) installed by
  `polyselect-gpu.cu` runs the whole-range generate → `thrust::sort_by_key` → detect
  pipeline on the device (shared, mutex-serialized, persistent buffers) and returns
  the colliding `(u, p₁, p₂)`. Wired into the **computational** collision pass (`CCS`)
  in `polyselect_collisions.cpp`: one thread issues the device search over the full
  prime range and pushes the *same* `polyselect_match_info` jobs as the CPU `shash2`,
  so `match` and everything downstream are byte-identical. The cheap **decisional**
  pass (`DCS`, run per special-q) stays on CPU.
- **Size-gated, so no regression.** The GPU path engages only when the estimated
  u-count (`Σ nrᵢ·2·umax/pᵢ²`) clears ~4 M; below that the unchanged CPU `shash`
  runs. At the c59–c90 sizes testable here (a few thousand u) it is a no-op; it
  engages at large N where the collision search dominates stage-1. The measured-
  negative GPU root-finder was **split off** to a separate opt-in
  (`CADO_GPU_POLYSELECT_ROOTS`) so it no longer rides along with `--gpu-polyselect`.
  (`CADO_GPU_POLYSELECT_FORCE` overrides the gate for testing.)
- **Validated.** Forcing the gate on at c59 exercised the device path **203×, 0
  fallbacks**, polynomial set **byte-identical** to CPU (66 = 66); **end-to-end
  `product == N`** with GPU collisions forced; all 32 `polyselect` ctests pass on the
  default CPU path. Enabled-but-gated-off cost is a one-time ~4 s CUDA init (constant
  vs work size; negligible for a real run). The fixed bug worth noting: the per-prime
  offset scan over the `uint8_t` `nr` array must accumulate in 32-bit (an explicit
  `0u` init) — without it the scan overflows at 256, scrambling the expansion into a
  6.4-trillion-entry garbage workload (OOM / `ppl==0` infinite loop). Full design +
  results in `docs/gpu-polyselect.md`.

### GPU (C2) — collision-search offload, foundation kernel (the real win; bit-exact)

- **GPU collision search, validated bit-exact** (`bench/gpu-polyselect-collision.cu`).
  The collision search is the memory-bound *bulk* of polyselect stage-1 and the part
  that **grows with N** (so, unlike root-finding, no Amdahl ceiling). For each prime
  `p` with lifted root `r` CADO emits every `u ≡ r (mod p²)` in `[−umax, umax)` and a
  collision is two equal `u` from different primes. The GPU-friendly reformulation —
  **generate → `thrust::sort_by_key` → detect adjacent-equal**, with the whole
  `u`-multiset **resident on the device** (only the small `(p,r)` table in, only the
  few collisions out) — sidesteps the PCIe round-trip that sank the root-finding
  offload. Gate = bit-exact vs a CPU `std::sort` reference on **44.9 M `u`-values**
  (4459 primes in `[50000,100000]`, `umax = 2.5·10¹³`, 539 MB resident): **sorted
  multiset identical (0 mismatches)** and **collision set identical** (31 — 8
  injected + 23 natural birthday collisions). **GPU 40.3 ms vs CPU `std::sort`
  5270 ms (≈130×, 1.1 Gu/s).** Honest: the 130× is vs `std::sort`; CADO's `shash` is
  an O(n) hash (faster than `std::sort`), so the real in-situ speedup vs `shash` is
  smaller and will be measured during live integration (same discipline as the
  root-finder). What this proves: the entire collision multiset can be built, sorted
  and de-duplicated on the GPU, bit-exactly, with no PCIe traffic for the bulk data —
  the foundation for the msieve-style "stage-1 resident on the GPU" endgame. Design +
  the live-integration plan in `docs/gpu-polyselect.md`.

### Roadmap (A1) — 3-D lattice sieving re-scoped (honest correction)

- **A1 closed for the factorization track.** The roadmap listed 3-D lattice sieving
  (arXiv 2001.10860) as a locally-testable `c100` factorization win. Research showed
  this is a **misread**: that paper's 3-D/higher-D enumeration is for the **Tower NFS
  / extension-field discrete log** (its result is a 𝔽_{p⁶} DLP), where elements have
  degree ≥ 3; **integer-factorization relations are (a,b) pairs — inherently 2-D**,
  and CADO's siever is 2-D because of that, so the stated "seeded c100" gate is not
  achievable. Recorded as a negative (like PGO / column-reorder); any 3-D pursuit
  belongs under DLP/exTNFS (A4), not the factorization track. `docs/ROADMAP-v3.2.0-modern.md`
  updated.

### Orchestration/UX (E2/E3) — factor planner + per-host autotuner (DONE)

- **`--plan` / `--plan-json` (E3, the "factor planner").** Given `N` and the
  host's thread count, `cado-nfs.py --plan` prints a feasibility verdict, a
  **wall-time envelope** (central + ±20 % NFS-variance band), the rough per-phase
  split, a single-machine-vs-cluster recommendation, and **GPU triage**
  (`--gpu-prefactor` first; GPU BWC linalg at large N) — then exits without
  running anything. The model is anchored on the **measured** `BENCHMARKS.md`
  numbers (c60 18.5 s, c70 27.2 s, c80 74.4 s, c90 197.9 s on the i9-10850K at 20
  threads) plus the documented order-of-magnitude envelope (c100 ~10 min, c110
  ~1 hr), interpolated log-linearly in digit count and Amdahl-scaled to the host's
  thread count (parallel fraction 0.7). It reproduces the c90 anchor to the second
  (3.3 min) and scales correctly with `-t` (55.8 s at c70/-t 8 vs 27 s at -t 20).
  `--host-speed FACTOR` lets a user fold in a measured per-core speed; `<60` digits
  is flagged TOO SMALL (use ECM/P±1), `≥130` flags distributed mode. Honest
  throughout: clearly labelled an estimate, with the variance / unmeasured-per-core
  caveats printed.
- **`--autotune` (E2, the per-host tuner).** Calibrates **only the safe scheduling
  knobs** to the detected host — the local client/thread layout
  (`slaves.nrclients`, e.g. 2 clients × 2 threads on 4 cores) and the work-unit
  *granularity* (`tasks.sieve.qrange`, `tasks.polyselect.adrange`, scaled by a
  bounded √(threads/20) factor clamped to [0.5, 4×]). It **never touches the
  number-theoretic bounds** (`lim*`/`lpb*`/`mfb*`/`I`), which determine relation
  yield and matrix structure — so only the *chunking* of identical work changes and
  `product == N` is preserved by construction. Verified end-to-end: the c59 smoke
  with `--autotune -t 4` set nrclients=2, qrange 2000→1000, adrange 5000→2500,
  logged "bounds unchanged", and returned the correct four prime factors in 15.2 s.
- **New module `scripts/cadofactor/planner.py`** (no third-party deps,
  side-effect-free): the estimator, the duration/feasibility formatting, host
  detection (threads + GPU-present + GPU-build-present), and the autotune override
  computation — all pure and **doctested** (registered as `test_python_planner`).
  Wired into `toplevel.py` via the same print-and-exit pattern as `--suggest-params`
  (plan handled before the parameter file is even resolved, since it needs only N);
  `--autotune` applies after the parameter file is read.

Post-`3.1.0-modern` housekeeping carried in this cycle (no code/behaviour change):

- **Project renamed to `cado-nfs-modern`.** Both the local checkout and the
  GitHub repository were renamed `cado-nfs-3.0.0-modern` → `cado-nfs-modern`
  (GitHub auto-redirects the old URL); the repo's About blurb was reworded.
  `main` was fast-forwarded to the `3.1.0-modern` release tip and is the default
  branch; the interim `v3.1.0-dev` / `v3-modern` branches were removed (all
  contained in `main`; release tags retained).
- **README** rebannered to 3.1.0-modern: badges/links point at `cado-nfs-modern`
  (+ an AVX-512-validation CI badge and a release badge), a "New in 3.1.0-modern"
  section (GPU linalg, GPU pre-factoring, AVX-512 VPCLMULQDQ/IFMA,
  orchestration/UX), a GPU performance note, and the new docs in the index.
- **`.gitignore`** hardened: standalone bench binaries built into the repo root
  (`gpu-prefactor`, `ifma-modmul`, `gpu-spmv-bench`, …), PGO/profiling artifacts
  (`*.gcda`/`*.gcno`/`*.profraw`/`perf.data`), `--json-status` snapshots,
  `autom4te.cache/`, and editor/OS cruft.
- **`BENCHMARKS.md` re-run and reorganized for 3.1.0-modern** (all numbers
  re-measured 2026-06-06 on the i9-10850K + RTX 3090). Restructured into six
  consistent sections — CPU factorization (seeded c60-c90 sweep + per-phase + the
  2.3.x comparison + projections, all `product == N`), the deterministic siever
  microbench (11.67 s, confirming no 3.1.0 CPU-path change), GPU pre-factoring ECM
  (48.7×/25.4×/10.5×), GPU linear algebra (SpMV scaling sweep 30.6→8.1 Gnz/s
  c100→c120 vs CPU 2.25→0.23, + the end-to-end c90 residency anchor, bwc 8.18 s),
  AVX-512 kernels (VPCLMUL `mul1` + IFMA modmul bit-exact under SDE), and a unified
  Reproducing section. README Performance table + GPU note synced to the same
  figures. `CLAUDE.md` updated for the rename + current 3.1.0 state.

## [3.1.0-modern] — 2026-06-06

From 3.1.0-modern the fork carries its own minor line (still upstream 3.0.0's NFS
algorithms; the bump reflects substantial original work beyond a pure mirror).
This release adds GPU linear algebra (SpMV + full vector residency), a GPU
pre-NFS ECM front-end, AVX-512 VPCLMULQDQ/IFMA kernels (SDE-validated), and a
batch of orchestration/UX features. Every change is gated by a verified
`product == N` factorization or a bit-exact validation; hardware-blocked items
(multi-GPU/multi-node, DLP/product-tree) are documented as designs rather than
shipped as unvalidated code.

### GPU (Track 2.2) — linear-algebra SpMV (foundation)

- **Bit-exact GF(2) SpMV on the GPU** (`bench/gpu-spmv-bench.cu`), the kernel
  Block Wiedemann (`linalg/bwc`) runs thousands of times and the
  fastest-growing NFS phase (~110× c60→c90). Mirrors `matmul-basic` exactly
  (`dst[i] = XOR_{j in row i} src[j]`, bitsliced K-limb blocks b64/b128/b256);
  validated bit-exact vs the CPU loop (0 wrong words). With the matrix resident
  on the device (as BWC reuses it), on an RTX 3090 vs a 20-thread i9-10850K it
  reaches 6–15× the *naive* CPU SpMV (7.9 Gnz/s b64, 5.1 b128).
- **Honest comparison vs the real `bucket` backend** (measured with
  `bench_matcache` on real `random_matrix` matrices). Single-thread on a 1M×1M
  30M-nnz matrix: `basic` ~0.35 Gnz/s, `bucket`/`sliced` ~0.63–0.67 (~1.8× the
  naive loop). A **full-CPU threaded scan** (one ~6M-nnz submatrix per thread)
  shows `bucket` is **memory-bandwidth-bound — it saturates at ~1.8 Gnz/s** (only
  ~2.1× the single thread across all 20 threads, flat by ~N=4). So vs the *full
  production CPU*, the GPU b64 kernel (7.9 Gnz/s) is a **measured ~4.4×** — well
  below the inflated "6–15× vs naive" but a real single-machine win, and a floor
  (the kernel realizes only ~10% of GPU bandwidth). The GPU's largest advantage
  remains at *scale* (aggregate multi-GPU/multi-node bandwidth, out-of-core
  matrices). Full design + measured numbers: `docs/gpu-linalg.md`.
- **A real `matmul_bNN_gpu` BWC backend** (`linalg/bwc/matmul-gpu.cu`) plugged
  into CADO's matmul dispatch — `mm_impl=gpu`, registered through the same
  `CONFIGURE_MATMUL_LIB`/`COOKED_BWC_BACKENDS` machinery as `basic`/`bucket`,
  built (b64+b128) only with `-DENABLE_GPU=ON` (nvcc compiles it with all of
  CADO's C++20 arith headers). Mirrors `matmul-basic`'s cache format and keeps
  **both M and Mᵀ resident on-device as CSR** so both BWC directions are fast
  gathers. **Passes `bench_matcache`'s `(M·v₁)·v₂ == (Mᵀ·v₂)·v₁` check — all 4,
  both directions — on a real matrix — and **end-to-end in a real factorization**:
  `cado-nfs.py … tasks.linalg.bwc.mm_impl=gpu` (59-digit, thr=2x2) drives the whole
  BWC pipeline (krylov/lingen/mksol/gather, balanced matrix, multi-thread comm)
  through the GPU backend and returns the correct factors (product == N). The
  backend **pins each reused BWC host
  vector once** (page-locked memory), so all H2D/D2H transfers run at full PCIe
  speed: **6.76 Gnz/s** (up from 4.95 pageable) — **~11× a single `bucket`
  thread, ~3.75× the full-CPU `bucket`** (1.8 Gnz/s). The residual ~1 ms/iter is
  `src`+`dst` still crossing PCIe.
- **Coalesced warp-per-row kernel** (`bench/gpu-spmv-bench.cu` + the backend): lanes
  stride the row's nonzeros (coalesced `col[]`), `src` gathered via `__ldg`,
  K-limb accumulator warp-reduced. Bit-exact (warp PASS / `bench_matcache` 4/4)
  and **1.8–3.1× faster** standalone; in the backend it cut the kernel
  2.63→0.97 ms, lifting end-to-end SpMV to **8.96 Gnz/s — ~5× the full-CPU
  `bucket`**. This flips the bottleneck to **72% transfers / 28% kernel**, so full
  vector residency (the scoped multi-step vector-layer port in
  `docs/gpu-linalg.md`) is now the dominant remaining single-machine win.
- **Comm-on-device foundation** (toward full vector residency — the 72% transfer
  share). Three validated pieces, each gated by a verified factorization
  (`product == N`): (1) a **process-global device-vector registry** keyed by host
  pointer (so the BWC comm can reach *sibling threads'* device buffers — all
  threads share the CUDA context); (2) a **`bwc_base`↔GPU hook ABI**
  (`matmul-gpu-hooks.h`, function pointers the GPU backend installs at init, so
  `bwc_base` keeps no hard CUDA dependency); (3) a **bit-exact device GF(2)
  reduce/broadcast** wired into `mmt_vec_allreduce` behind `CADO_GPU_DEVCOMM`.
  `product == N` on the c60 across thread grids `-t 2/4/6/8` and on a c70, in
  default, `DEVCOMM`, and `VECRESIDENT+DEVCOMM` modes; regression-guarded by a new
  `test_gpu_vecreduce` ctest (GPU builds only).
- **The per-iteration hot comm on device — the 2D "shuffled-product" transpose.**
  A 1-matrix factorization's hot comm is `matmul_top_mul_comm` =
  `mmt_vec_reduce`+`mmt_vec_broadcast`, which reduce-scatters along one grid axis
  and all-gathers along the perpendicular one (coupling all threads' data) — much
  harder than the 1D `allreduce` above. `matmul_top_mul_comm_gpu`
  (`matmul_top_comm.cpp`) runs it on the device-resident buffers by **mirroring
  the host algorithm op-for-op at identical byte offsets** (so it is bit-for-bit
  the host comm by construction): five barriered phases per thread over new
  low-level device-op hooks (`xor_block`/`copy_block`/`upload`/`download`/`ensure`
  in `matmul-gpu-hooks.h`), with the broadcast collapsing to a no-op for the
  common `THREAD_SHARED` source vector. `product == N` on the c60 across `-t 4`
  (2×2 square) and `-t 8` (2×4 rectangular), 10/10 each, in default, `DEVCOMM`,
  and `VECRESIDENT+DEVCOMM` modes; `compute-sanitizer` memcheck on `prep` reports
  0 errors. A flaky data-dependent fault found en route (compute-sanitizer):
  `mul()`'s host-buffer pinning (`cudaHostRegister`) is smaller than the comm's
  full-vector copies and CUDA enforces the registered region — `g_pin` now skips
  under `DEVCOMM` (copies go pageable; the default path keeps pinning).
- **Full vector residency — the steady krylov loop runs entirely on device
  buffers (H2D/D2H eliminated).** With `CADO_GPU_VECRESIDENT` + `CADO_GPU_DEVCOMM`
  the BWC vectors stay device-resident across `mul → comm → mul`: `mul()` skips its
  H2D (device src current) and D2H (dst left on device, host stale); the 2D comm
  skips its host upload of `w` and marks its output `v` device-resident instead of
  writing it back; `matmul_top_mul` no longer invalidates the comm's device result
  in residency mode. Scoped to the krylov inner loop (`cado_gpu_residency_active`,
  set/cleared in `krylov.cpp`) so prep/secure/twist stay host-authoritative; the
  one per-iteration host read (`x_dotprod`) and the loop boundary materialise the
  vector via `cado_gpu_sync_to_host`, and twist/untwist already invalidate the
  device copy so each block re-seeds. **Validated:** `product == N` 45/45 across
  default, `DEVCOMM`, and `VECRESIDENT+DEVCOMM` × `-t 2/4/8`; `compute-sanitizer`
  memcheck on krylov in residency mode reports 0 errors; transfer counters
  (`CADO_GPU_STATS`) confirm the steady loop skips **D2H 100% / H2D ~99%**. This
  eliminates the per-iteration PCIe transfers (the measured ~60% of SpMV time at
  c70/c80, growing with N), realising the ~2.6× SpMV-hot-loop win where linalg
  dominates (large N / DLP). Default and DEVCOMM-only paths are unchanged and
  bit-exact.
- **GPU `x_dotprod` — krylov's last per-iteration D2H removed.** The BW-sequence
  gather (`xdotprod.cpp`) now runs on the device-resident vector
  (`xdot_kernel`, GF(2), x-indices uploaded once and cached) instead of pulling
  the whole vector back to host; `x_dotprod` is residency-aware (device gather, or
  sync + host path if the device copy isn't current). `cado_gpu_residency_available`
  (set by the backend iff both flags are on) gates the residency code paths so they
  engage only in genuine residency runs. `product == N` 18/18 across modes × grids;
  compute-sanitizer clean. With this the steady loop has **zero** per-iteration
  host transfers.
- **Multi-node MPI + per-rank multi-GPU.** The GPU backend builds and runs under
  MPI (build fix: `--mpi` compile marker applied only to non-CUDA languages so
  `matmul-gpu.cu` compiles under nvcc). `gpu_select_device()` binds each process to
  a GPU by its node-local MPI rank (the standard one-rank-per-GPU model;
  `CADO_GPU_DEVICE` overrides). Validated `product == N` with GPU SpMV under 2 MPI
  ranks (`mpi=1x2`/`2x1`) in default, `DEVCOMM`, and `VECRESIDENT+DEVCOMM` modes —
  the device comm safely falls back to the host comm for `njobs>1` (residency is
  single-rank), so multi-rank runs stay correct. On a single GPU the device
  selection is a no-op; it round-robins across GPUs on multi-GPU hardware.
- **mksol full vector residency — transfer-free accumulator.** Extends residency
  from krylov to mksol, whose inner loop does a per-iteration host `addmul_tiny`
  (`ymy[0].own += Σ vi[i].own × ff`, GF(2)) that otherwise forces the accumulator
  back to the CPU each iteration. A bit-exact GPU `addmul_tiny`
  (`bench/gpu-addmul-bench.cu`, validated standalone) runs it on the device, so
  the accumulator stays device-resident across the whole block: mksol seeds the
  constant-per-block `vi[i]` + zeroed accumulator on the device, runs the addmul
  on the GPU (the matmul reuses the krylov residency path), and materialises
  `ymy[0]` at block end for untwist/save. Works because mksol's `ymy[0]` is a
  shared vector, so the standalone `mmt_vec_broadcast` is a no-op. `product == N`
  18/18 across default, `DEVCOMM`, `VECRESIDENT+DEVCOMM` × `-t 4/8`;
  compute-sanitizer clean; mksol's matmul runs fully transfer-free (H2D 100% +
  D2H 100% skipped — even cleaner than krylov, the accumulator is never
  invalidated mid-block). GF(2) only; falls back to the host addmul otherwise.
- **secure full vector residency + single-node residency gate.** `secure`'s
  check-vector accumulator (the same GF(2) `addmul_tiny` as mksol, on the shared
  `dvec`) now uses the device path too, so the last BWC stage that touched the
  accumulator host-side each iteration stays device-resident. With this, residency
  is **gated to the single-node case** in all three drivers (krylov/mksol/secure):
  `pi->wr[0]->njobs == 1 && pi->wr[1]->njobs == 1`. The device comm only handles
  `njobs==1`, and the sync-based MPI fallback yields no transfer win, so under MPI
  residency now cleanly disables and the run takes the validated GPU-SpMV +
  host-comm path (multi-node residency with a real transfer win needs a
  local-device/MPI-data comm split — future work, needs multi-GPU HW to validate).
  Also hardened `g_invalidate`: a host-buffer write (twist, or the MPI host comm)
  now clears `host_dirty` as well as `current`, making the host fully authoritative
  so a later `sync_to_host` cannot D2H stale device data over it. `product == N`:
  single-rank resident `-t 4/8` PASS, `mpi=1x2`/`2x1` resident (disabled) PASS,
  default/`DEVCOMM`/resident 12/12 PASS; compute-sanitizer clean.
- **Build fix — GPU backend links into standalone matmul tools.** The
  `cado_gpu_*` comm-on-device hook pointers were *defined* in `bwc_base`
  (`matmul_top_comm.cpp`), which the GPU matmul backend references but
  `bench_matcache` does not fully pull in — a circular static-link dependency left
  the symbols unresolved (`bench_matcache --impl gpu` failed to link). The
  definitions moved to a new dependency-free leaf TU `matmul-gpu-hooks.cpp`
  compiled into `matmul_common` (the leaf every backend already links), so both
  `bwc_base` (which calls the hooks) and the GPU backend (which installs them)
  resolve them with no cycle and no CUDA dependency in CPU-only builds. Behaviour
  is byte-identical (null pointers ⇒ host comm). Re-validated end-to-end after a
  full relink: c59 GPU `product == N`.
- **Intra-node multi-GPU matrix partition (`CADO_GPU_NPART`).** The GPU SpMV
  backend can split each direction's CSR into `nparts` contiguous output-row
  chunks placed round-robin across the visible GPUs; `mul()` runs one partial
  SpMV per chunk (src replicated to each device) and gathers the dst chunks.
  Default `nparts=1` is exactly the prior single-device path (zero overhead). On
  a single GPU every chunk maps to device 0, which **exercises the
  split/multi-launch/gather logic bit-exactly** — `product == N` validated on the
  c59 at `CADO_GPU_NPART=1/2/3` (identical factors). Genuine cross-physical-device
  execution (2+ GPUs, and the per-device-stream overlap that would make it a
  throughput win) is **unverified here — this box has one RTX 3090** — and is the
  documented next step (`docs/gpu-linalg.md`). The path is independent of vector
  residency (alternative strategies); `nparts>1` takes the plain
  upload/compute/writeback path.
- **Multi-node residency — design documented (not code).** The transfer-win
  version of residency-under-MPI is specified in `docs/gpu-linalg.md` (the
  local-device-reduce / MPI-boundary-only-exchange split) rather than implemented,
  for a concrete reason: transfer accounting shows any version that routes the
  comm through host MPI just moves the two per-iteration transfers from the mul to
  the comm (**no net win** — confirming why residency is single-node-gated), and
  the only winning version needs ≥2 GPUs (ideally CUDA-aware MPI) to realise *and*
  to validate for correctness (an earlier attempt hit a reduce-scatter
  direction/offset bug and was reverted rather than ship a comm returning wrong
  results). The doc gives the full algorithm + buffer choreography so it drops
  into `matmul_top_mul_comm_gpu`'s `njobs>1` branch when multi-GPU HW is available.
- **GPU linalg at scale — measured (Track 2.2 headline).** A scaling sweep
  (`bench/gpu-spmv-bench.cu`, b64, bit-exact at every size) shows the GPU SpMV
  win **grows with N**: GPU warp kernel 28.9→7.9 Gnz/s as the matrix grows
  1M→8M rows (240M nnz, c100→c120 scale), while the CPU reference loop collapses
  1.58→0.19 Gnz/s (random-CSR cache thrash) — 18×→41× over the reference loop, a
  steadier ~4.4× over the tuned `bucket`'s saturated ~1.8 Gnz/s. End-to-end anchor:
  a 90-digit GNFS with `mm_impl=gpu` + residency returns `product == N` (bwc 8.18s
  real — small at c90, sieving-dominated). Recorded in `BENCHMARKS.md`.

### CPU/SIMD (Track 1.1) — AVX-512 VPCLMULQDQ gf2x auto-detection

- **gf2x VPCLMULQDQ backend auto-detection.** `gf2x/config/features.m4` adds
  `CHECK_VPCLMUL_SUPPORT` (+ a `VPCLMUL_EXAMPLE` using `_mm512_clmulepi64_epi128`)
  and `configure.ac` prefers the `x86_64_vpclmul` backend over `x86_64_pclmul`
  when supported. Because the probe uses `AC_RUN_IFELSE`, it only says "yes" when
  the test program actually runs — so a non-AVX-512 host (the Comet Lake reference
  box) reports "no" and keeps the pclmul backend: **no mis-selection, no
  regression** (verified: regenerated `configure` selects `x86_64_pclmul` here and
  gf2x's own `make check` passes). On real AVX-512 silicon the VPCLMULQDQ backend
  (hot `mul1` accelerated; `mul2`–`mul9` are the proven PCLMUL code, valid there)
  is selected automatically.
- **`mul1` bit-exact under Intel SDE** (`bench/vpclmul-validate.sh`, now
  auto-finding `/opt/intel-sde/sde64`): PASS, 200 000 trials vs the scalar
  reference — the primary Drucker–Gueron base-case win.
- **CI validation** (`.github/workflows/avx512-validate.yml`): regenerates
  `configure` (autoreconf, in CI's clean autotools env — so the source commit
  doesn't churn this box's vendored libtool/automake), checks the auto-detection,
  validates `mul1` under SDE (objdump fallback if SDE is unavailable), and — when
  SDE is present — force-builds the `x86_64_vpclmul` backend and runs gf2x's tests
  under `sde64 -future`. Correctness-only; the ~39% perf gain is measurable only on
  AVX-512 hardware. Remaining (perf-gated): port `mul2`–`mul9` to VPCLMULQDQ +
  threshold retune (see `gf2x/already_tuned/x86_64_vpclmul/INTEGRATION.md`).

### CPU/SIMD (Track 1.4) — AVX-512 IFMA GF(p) modmul kernel (foundation)

- **Bit-exact AVX-512-IFMA batched Montgomery modular multiplication**
  (`bench/ifma-modmul.c`): the foundation kernel for an mpfq GF(p) IFMA backend
  (DLP linear algebra). It does **8 independent modmuls per instruction stream**
  (one per 512-bit lane) in radix 2^52 — the natural IFMA radix, so partial
  products land exactly in the lo/hi halves `_mm512_madd52lo/hi_epu64` produce —
  via a lane-parallel CIOS Montgomery. Validated **bit-exact vs GMP under Intel
  SDE** (`bench/ifma-validate.sh`, `-future`): **0/32000 wrong, 260-bit, 8-way.**
  Same method as the gf2x VPCLMULQDQ work (Comet Lake has no IFMA; SDE emulates
  it). CI-gated in `.github/workflows/avx512-validate.yml` (objdump `vpmadd52`
  fallback if SDE is unavailable). Full mpfq integration (wiring it into the
  generated GF(p) arithmetic + the BWC GF(p) SpMV, and a perf number on real
  IFMA silicon) is follow-up; this proves the arithmetic primitive.
- **Cofactor scale-out + GPU product-tree — design documented** (Tracks 2.3/2.4,
  `docs/gpu-cofactorization.md`). The multi-GPU curve-batching *mechanism* already
  exists and is validated at N=1 (`misc/gpu_prefactor` splits across
  `cudaGetDeviceCount()` devices); what remains — distributing across ≥2 GPUs,
  MPI-awareness, and DLP tuning — is HW/regime-gated, not algorithmic, so it is
  specified rather than shipped. GPU batch product-tree smoothness (Bernstein
  remainder tree, flag-gated) is a new algorithm requiring a bit-exact
  relation-set harness; designed to reuse the validated multi-precision device
  arithmetic (`bench/gpu-ecm-mp.cu`, `bench/ifma-modmul.c`).

### CPU (Track 1.2) — PGO retry (honest negative)

- **Siever-trained PGO retry — rejected again.** v3.0.0 rejected whole-program PGO
  (+2.8% slower). This retry instrumented only the `las` objects
  (`-fprofile-generate`), trained on a multi-seed c120 sampled-special-q corpus,
  then rebuilt with `-fprofile-use` (gcc; `-fprofile-partial-training` so the rest
  stays `-O3`). Measured on `bench/las-microbench.sh` (deterministic, <1%
  variance): **12.04s vs 11.69s baseline = +3.0% slower** — no win, consistent
  with v3.0.0. The `-O3 -march=native` host-ISA codegen already captures the gain;
  PGO's layout/inlining choices do not help this siever. Recorded, not adopted.

### CPU (Track 1.3) — hot-scalar micro-opt (profiled; no safe win)

- **Profiled the siever; the hot loops are already optimally tuned.** A `perf`
  profile of the c120 microbench workload puts the self-time in
  `fill_in_buckets` (~12%), `plattice_info` ctor (~11%), `sieve_small_bucket_region`
  (~10%), ECM `stage2_one_w` (~10%), `invmod_redc_32` (~9.5%), and
  `apply_buckets_inner` (~7%). The bucket-apply scatter — the obvious software-
  prefetch candidate — is already 16×-unrolled with batched 64-bit reads, SIMD,
  and an update-stream prefetch (`las-apply-buckets.hpp`), and its scatter target
  `S[x]` is cache-resident *by design* (bucket regions are sized to fit cache).
  The remaining candidate, batched modular inversion (Montgomery's trick) in
  `invmod_redc_32`/`plattice_info`, needs an invasive restructuring of per-prime
  lattice setup — high correctness risk on a tight 9.5% routine, with expected
  nil-to-negative payoff given the two PGO negatives and how hand-tuned the code
  already is. **No micro-opt adopted**; the `-O3 -march=native` codegen win
  already captures what is safely available. Recorded per the measure-and-record
  ethos.

### UI/UX (Track 3.1) — run-status reporting

- **`--json-status FILE`** writes a machine-readable status snapshot (schema
  `cado-nfs-status/1`: state, phase + index/total, percent, ETA, work-unit
  counts, factors, timestamps), rewritten atomically on every update — for
  dashboards/tooling and the forthcoming `/status` endpoint.
- **`--progress`** shows a compact single-line progress indicator (phase ·
  percent · work-units · ETA) on stderr (pair with `--screenlog WARNING` for a
  clean line).
- New dependency-free `scripts/cadofactor/status.py` singleton, fed by the
  existing phase loop (`CompleteFactorization.run`) and per-work-unit
  `verification()` (which already computed achievement + ETA). Off by default; no
  behaviour change unless a flag is given. Validated on a 59-digit factorization.

### UI/UX (Track 3.2) — /status endpoint + dashboard

- **`GET /status` on both servers.** The Flask work-unit server
  (`api_server.py`) serves the live `cado-nfs-status/1` snapshot (state, phase +
  index/total, percent, ETA, work-unit counts, factors) straight from the
  in-process status singleton, and the Rust `cado-wu-server` serves a
  `cado-nfs-wu-status/1` snapshot (work-units total/available/assigned/ok/error/
  done + percent + serving flag) computed from the wudb it already owns. Both
  reuse data the servers already hold — no new bookkeeping.
- **`GET /dashboard`** (Flask) is a dependency-free single-page view that polls
  `/status` every 2 s and renders phase / progress bar / ETA / work-units /
  factors. Inline HTML, no static-asset plumbing.
- The Track 3.1 status reporter (`status.py`) now **tracks state in memory
  unconditionally** (the file/stderr *output* stays gated on
  `--json-status`/`--progress`), so `/status` is live even when no status flag was
  passed. Validated live: a running c90 factorization reported phase "Polynomial
  Selection (root optimized)" [2/12] via `GET /status`; the Rust endpoint reported
  correct per-status work-unit counts against a seeded wudb.

### UI/UX (Track 3.3) — parameter interpolation

- **Parameter interpolation instead of a hard error.** When no preset parameter
  file matches the input size (after the existing ±3/±5-digit search window),
  `find_default_parameter_file` (`toplevel.py`) now **interpolates** a file from
  the two nearest presets that bracket the size — linearly scaling every numeric
  key present in both (`lim*`, `lpb*`, `mfb*`, `I`, `qmin`, `ncurves*`,
  `target_density`, …) by the fractional digit position, rounding integer keys,
  and preserving the nearer preset's structure/comments. If the size is outside
  the available preset range it clamps to the nearest single preset. Either way it
  emits a clear "heuristic; override with `-p`" warning, so off-preset sizes
  (e.g. a cofactor returned by `--gpu-prefactor`, or any gap such as c45 between
  the c30 and c60 presets) get usable parameters instead of aborting the run.
- **`--suggest-params`** resolves the parameter file for `N` (interpolating if
  needed), prints its path and contents, and exits without factoring — a quick way
  to inspect or capture a starting point to hand-tune.
- Validated: `product == N` on a real 45-digit factorization driven entirely by
  auto-interpolated c30↔c60 parameters; all 299 `toplevel.py` doctests pass
  (the three that asserted the old "no parameter file found" error now assert the
  interpolated-file behaviour). No change when an exact/near preset exists.

### UI/UX (Track 3.4) — clap CLIs + cluster launcher

- **clap argument parsing for both Rust binaries.** `cado-wu-server-rs` and
  `cado-nfs-client-rs` replace their hand-rolled `parse_args` loops with `clap`
  derive parsers, gaining real `--help`/`--version`, value validation, and clear
  errors. Every flag name and semantic is preserved (server `--db` still required,
  `--whitelist` still comma-separated/repeatable; client `--server` still
  repeatable for failover, the TLS flags `--insecure`/`--cafile`/`--certsha1`
  still set the `CADO_NFS_*` env vars the client reads). Validated end-to-end:
  `rust/server-interop-test.sh` and `rust/server-swap-test.sh` PASS — the latter
  drives a full c59 `product == N` with the Rust server swapped in via the Python
  shim, launched with the exact `--db/--addr/--port/--uploaddir/--whitelist`
  arguments the new parser handles.
- **`scripts/cluster-launch.sh`** fans the static Rust client across a cluster:
  SSH host list (`--hosts`/`--hostfile`, `--clients-per-host`) or Slurm
  (`--slurm --ntasks`), all pointed at one `--server` with the same `--certsha1`
  pinning; `--stop` kills the clients, `--dry-run` previews. Auto-finds the client
  binary in the build/Rust target dirs.
- **`cado-nfs-monitor-rs`** — a ratatui terminal dashboard that polls a server's
  `/status` and renders a live progress gauge, phase/state, work-unit counts, ETA,
  and discovered factors. Understands both schemas (`cado-nfs-status/1` from the
  Flask driver and `cado-nfs-wu-status/1` from the Rust server). `--once` prints a
  plain-text summary and exits (scriptable / no-TTY). Validated `--once` against
  both a live Rust work-unit server and a `cado-nfs-status/1` document
  (c120 `[5/12] Lattice Sieving` 63.4%, wu 820/1300, ETA rendered).

### GPU (Track 2.1) — pre-NFS factoring front-end (validated)

- **Multi-precision (K-limb) GPU ECM.** The existing GPU ECM
  (`sieve/ecm/gpu_ecm.cu`) only handles moduli < 2^126 — useless for stripping a
  factor from a real NFS-sized N, since ECM runs *modulo N*. `bench/gpu-ecm-mp.cu`
  generalizes the validated 2-limb CIOS Montgomery and the Montgomery-curve XZ
  ladder to K 64-bit limbs, so ECM stage-1 runs modulo a multi-hundred-bit N. The
  same `__host__ __device__` code runs on CPU and GPU; **validated bit-exact**:
  `montmul` 0/20000 wrong vs an independent binary-mulmod reference and ECM
  0/512 GPU lanes differing from CPU, for 128/256/512-bit widths. Functionally it
  **strips a planted ~20-bit factor from near-full-width composites** (~520k /
  135k / 30k curves/s for 128/256/512-bit on an RTX 3090). This is the math
  foundation for the standalone `--gpu-prefactor` stage.
- **Standalone `gpu-prefactor` tool** (`misc/gpu_prefactor/`): a working CLI that
  strips factors from a decimal `N` on the GPU before NFS (which CADO already
  tells users to do by hand). Width `K ∈ {2,4,8,16}` (up to ~307-digit `N`) is
  chosen automatically; GMP handles parsing, the Montgomery setup, and `gcd`.
  Demonstrated stripping a 12-digit factor from a 103-digit `N` (cofactor
  correctly identified prime). Unlike in-sieve GPU cofactorization (a documented
  Amdahl no-win), pre-factoring is a *separate* stage, so the GPU throughput is
  pure upside.
- **Stage-2 BSGS + Suyama-σ curves** (generalizing `bench/gpu-ecm-stage2.cu` to
  K limbs). The Suyama setup's one modular inverse per curve is done on the host
  with GMP (non-invertible denominators are themselves free factors); the device
  runs stage-1 + a baby-step/giant-step stage-2 over the primes in (B1,B2]. Each
  run does a bit-exact GPU-vs-CPU self-check of the new BSGS composition
  (`# selfcheck: PASS`). This lifts the reach from ~12-digit to **15-digit
  factors** (e.g. stripped a 15-digit factor from a 102-digit N; a 14-digit from
  a 95-digit N) at the same `B1`. CLI: `gpu-prefactor <N> [B1] [curves] [B2]`.
- **Multi-GPU** batching: the curve batch is split across all visible devices and
  launched asynchronously (concurrent), degenerating to a single launch on a
  one-GPU box. Plus a **CMake target** (`misc/CMakeLists.txt`, built when
  `-DENABLE_GPU=ON`): `make gpu-prefactor`, installed to `<libdir>/misc`. The
  target forces device `-O3` (CADO sets no `CMAKE_BUILD_TYPE`, so default CUDA
  flags are unoptimized — a ~30× trap) and defaults to `sm_86`
  (`-DCADO_GPU_ARCH=` to override). The CMake build matches the standalone nvcc
  throughput (~550 curves/s, stage1+2, 512-bit on an RTX 3090).
- **`cado-nfs.py --gpu-prefactor`** integration (`scripts/cadofactor/gpu_prefactor.py`):
  runs the GPU stage before NFS, with `--gpu-b1/--gpu-b2/--gpu-curves`. If it
  fully factors N (cofactor 1 or prime), it prints the factorization and **skips
  NFS entirely**; if a composite cofactor remains, it finishes with a fresh
  `cado-nfs.py` on the cofactor (which selects parameters for the cofactor's
  size); if nothing is stripped or the GPU binary is absent, it falls through to
  a normal run. Only prime gcd-divisors are trusted (verified by Miller–Rabin).
  Validated: a 90-digit N (14-digit × 76-digit prime) was factored entirely by
  the GPU pre-stage in seconds, **product == N**, NFS skipped; normal runs
  without the flag are unaffected.
- **Staged-`B1` schedule** (`gpu-prefactor <N> staged [maxdigits] [scale]`):
  escalating `B1` (2000 → 11000 → 50000 → 250000 → 1e6 → 3e6) so small factors
  are found cheaply before spending curves at high `B1`; stops as soon as the
  cofactor is 1/prime. Reaches ~20–30-digit factors.
- **CPU-vs-GPU benchmark** (`bench/gpu-prefactor-bench.cu`): the *same* `ecm_run2`
  on the GPU vs the full 20-thread CPU (`std::thread`) — **49× / 25× / 11×** at
  128/256/512-bit on an RTX 3090 vs i9-10850K (the advantage shrinks at wider
  moduli as register/local-memory pressure lowers GPU occupancy — honest caveat).
  Recorded in `BENCHMARKS.md`. (Caught and fixed a dead-code-elimination trap: the
  CPU side needs an observable sink or the optimizer removes the whole ECM.)
- **Canonical deep-dive doc** (`docs/gpu-prefactor.md`): the why (separate stage,
  no Amdahl ceiling — contrast with `docs/gpu-cofactorization.md`), the
  multi-precision Montgomery ECM, the correctness gates, build/run, the integrated
  `--gpu-prefactor` path, and the measured throughput table. Benchmark re-confirmed
  on the RTX 3090 (48.6× / 25.9× / 12.2× at 128/256/512-bit, within noise of the
  recorded figures) and an end-to-end `staged` run on a 90-digit N verified
  `product == N` (every stripped factor divides N; cofactor × stripped-product == N;
  self-check PASS throughout).

## [3.0.0-modern] — 2026-06-05

Rebases the modernization effort onto upstream CADO-NFS **3.0.0** and adds four
tracks of rigorously-measured performance / robustness work. **No NFS
mathematics is changed** — every track is an implementation, build, or
orchestration optimization, and correctness (verified `product == N`
factorizations + `make check`) is the gate after each phase. Results are
reported honestly, including the parts that did *not* pay off.

Reference box for all measurements: **i9-10850K (10C/20T, Comet Lake → no
AVX-512) + RTX 3090 (Ampere, `sm_86`), 64 GiB, CachyOS.**

### Phase 0 — rebase onto upstream 3.0.0

- Builds cleanly on a current box (**CMake 4.x, GCC 16, C++20, Python 3.14**).
  3.0.0 already subsumes the 2.3.x portability fixes and brings, for free, the
  Bouvier–Imbert batch cofactorization (eprint 2018/669) and `I>16` sieving.
- The Flask/`requests` orchestration deps live in a venv
  (`scripts/setup-venv.sh`, pointed at by `local.sh`) rather than the system.
- Baseline established with a verified smoke factorization.

### Phase 1 — build & compiler (measured, not guessed)

- `bench/las-microbench.sh`: a deterministic siever microbenchmark (pinned c120
  polynomial + `las --random-sample N --seed S`, <1% CPU variance) that makes
  few-percent changes measurable despite the ~15–20% end-to-end polyselect noise.
- Result on the c120 siever workload (mean of 3, <1% variance):

  | flags | time | vs stock |
  |---|---|---|
  | stock `-O2` | 12.57 s | — |
  | `-O2 -march=native` | 11.79 s | −6.2% |
  | **`-O3 -march=native`** | **11.66 s** | **−7.2% (adopted)** |
  | `+LTO` | — | 0% (and breaks `-Werror`) → rejected |
  | `+PGO` | — | +2.8% (regresses) → rejected |

  `-march=native` (AVX2/BMI2/FMA host codegen) is the only real win; LTO and PGO
  are rejected on evidence. Final flags adopted in `local.sh`.

### Phase 2 — SIMD modernization

- **AVX2 on the siever: ruled out by profiling.** The hot path is scatter
  (`fill_in_buckets`, small-sieve) plus scalar modular arithmetic
  (`invmod_redc_32`, `plattice_info`), not vectorizable loops; the explicitly
  SSE2 survivor code is cold. Widening it would not move the needle.
- **AVX-512 VPCLMULQDQ gf2x base-case multiply** (the GF(2)[x] hot path used by
  `lingen`): `bench/vpclmul-mul1n.c` implements `mul_1_n`/`addmul_1_n` at four
  carryless multiplies per instruction (vs one per 128-bit `pclmul`), validated
  **bit-exact over 200 000 random trials under Intel SDE**
  (`bench/vpclmul-validate.sh`). `gf2x/already_tuned/x86_64_vpclmul/` carries the
  gf2x-integrated backend plus an `INTEGRATION.md` completion guide. Performance
  is gated on real AVX-512 silicon — the reference box is Comet Lake, so this is
  validated correct-only here.

### Phase 3 — GPU ECM cofactorization (RTX 3090) — validated, honest negative

- `bench/gpu-modmul-bench.cu`: batched 64-bit Montgomery modmul (ECM's inner
  primitive) — **~279 G modmul/s on the 3090 vs ~7.2 G across 20 CPU cores
  (~39×)**, correctness-checked.
- `bench/gpu-ecm.cu`: a complete batched ECM stage-1 (Montgomery-curve XZ ladder,
  on-device binary gcd). The same `__host__ __device__` code runs on CPU and GPU,
  so the GPU path is **bit-exact**; it cracks 256/256 test composites at
  ~2.75 M curves/s (B1=2000). Extended with Suyama-σ curves, a 128-bit Montgomery
  modulus (CIOS REDC), and a stage-2 BSGS continuation (+64% yield/curve).
- Integrated into CADO behind `facul`/`las-cofactor` with a CMake CUDA build and
  a live batched-drain hook (relations validated identical to the CPU path).
- **Honest conclusion (`docs/gpu-cofactorization.md`):** cofactorization is only
  ~8% of siever time (Amdahl ceiling), and a mid-effort "throughput win" turned
  out to be a correctness bug (a single-factor batch emitting relations with
  primes > `lpb`; fixed in `c675f82` with a complete-factorization check). With
  correctness enforced there is **no net single-machine speedup** from the GPU at
  these sizes — the value would be a cheaper-cofactorization *parameter-regime
  shift* on much larger inputs. Documented as a measured negative result rather
  than an unsubstantiated win.

### Phase 4 — Rust orchestration (robustness / scalability)

Ports the **network/DB substrate** of the distributed layer to async Rust while
keeping the **exact existing HTTP/JSON work-unit protocol**, so Rust binaries
interoperate with an unmodified `cado-nfs.py` during migration. This buys
robustness (no GIL, connection pooling, a single static client binary), not
single-machine factoring speed. Full details in `docs/rust-orchestration.md`.

- **`cado-nfs-client-rs`** (`rust/cado-nfs-client`, reqwest + rustls, no OpenSSL):
  the complete client loop — fetch work-unit, download + checksum-verify
  (sha1/256/3-256) inputs, run commands (argv split, no shell, exactly as the
  Python client), upload results. Plus failover across multiple `--server` URLs,
  TLS cert-pinning (`--certsha1`), `--niceness`, download `flock`, and `STDIN<n>`
  redirection (which the stock Python client never implemented).
- **`cado-wu-server-rs`** (`rust/cado-wu-server`, axum + tokio + bundled
  rusqlite): the five endpoints over the same `wudb` SQLite schema and status
  codes, replicating the `WuAccess` assign/result lifecycle, with stale-work
  timeout reassignment, `410`-after-finish, an r2d2/WAL connection pool, TLS
  (`--cert/--key`), and IP-whitelist enforcement (`--whitelist`, IPv4/IPv6
  exact + CIDR, matching `api_server.py`'s `api_limit_remote_addr`).
- **In-process swap:** `scripts/cadofactor/external_api_server.py` (`ExternalApiServer`)
  implements the `ApiServer` interface the driver uses by launching the Rust
  server over the same database; `cadotask.py` switches to it when
  `CADO_RUST_WU_SERVER` is set. TLS is wired (the shim regenerates the same
  self-signed cert and reports its SHA1 so clients pin it) and the whitelist is
  passed through.
- **`cadotask.py` is deliberately left in Python.** Phase 4's scope is the
  substrate, not the scheduler DAG; porting ~7.5K LOC of orchestration logic has
  no robustness/throughput payoff and a large correctness surface, so it stays
  behind the stable `ApiServer` seam.
- **Validation:** `rust/interop-test.sh` (Rust client ↔ stock Python server),
  `rust/server-interop-test.sh` (full lifecycle, Rust server + client),
  `rust/robustness-test.sh` (**13/13**), `rust/deploy-test.sh` (a real 59-digit
  factorization run entirely by external Rust clients), `rust/server-swap-test.sh`
  (Rust server swapped into a live `cado-nfs.py` run, plain HTTP) and
  `rust/server-swap-tls-test.sh` (the same over TLS with cert-pinning + whitelist).

### Net finding

CPU-side tuning is nearly tapped out on this hand-optimized codebase (Phase 1
~7%, Phase 2 AVX2 nil). The structural headroom is in *new compute resources* —
the GPU (validated ~39× on the modmul primitive, though Amdahl-bounded at these
sizes) and, where available, AVX-512 VPCLMULQDQ (validated correct) — and in the
Rust orchestration substrate for multi-client robustness.

### Versioning

- `CADO_VERSION_STRING` is `3.0.0-modern` (`CMakeLists.txt`); the numeric
  components stay at upstream 3.0.0, only the display string is suffixed. No test
  depends on the version string.

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
