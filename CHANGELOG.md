# Changelog

All notable changes to this fork are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
loosely follows [Semantic Versioning](https://semver.org/).

This is a downstream **modernization + performance fork** of upstream
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs). The `3.0.x-modern` line is
rebased onto upstream **3.0.0**; only the changes introduced by this fork are
listed. For the upstream history see [`NEWS`](NEWS). The earlier `2.3.1-modern`
release (rebased on upstream 2.3.0) lives on the `main` branch.

## [Unreleased] — 3.1.0-modern

From 3.1.0-modern the fork carries its own minor line (still upstream 3.0.0's NFS
algorithms; the bump reflects substantial original work beyond a pure mirror).
Work in progress — see the v3.1.0 roadmap. Landed so far:

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
