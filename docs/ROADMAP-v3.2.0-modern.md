# CADO-NFS v3.2.0-modern — roadmap

Forward plan for the next minor of this modernization + performance fork, after
`3.1.0-modern` shipped GPU linear algebra (SpMV + full vector residency), a GPU
pre-NFS ECM front-end, AVX-512 VPCLMULQDQ/IFMA kernels, and the Track-3
orchestration/UX layer. This roadmap is **research-grounded** (sources at the
bottom) and **honest about hardware**: the reference box is Comet Lake
(i9-10850K, **no AVX-512**) + a single **RTX 3090**, so AVX-512 work is
SDE/CI-gated and multi-GPU/multi-node work is validated at the degenerate path
locally + on cloud/CI multi-GPU, with perf claims only from real hardware.

## The strategic reframe

The single most important datapoint for sequencing:

> **RSA-250 cost (Zimmermann/PRACE): sieving ≈ 2450 core-years (91 %), linear
> algebra ≈ 250 core-years (9 %).**

`3.1.0-modern` put its GPU effort into the linear-algebra phase — the **9 %**. At
scale the cost lives in **sieving and polynomial selection**, so v3.2.0 aims the
algorithm + GPU + CPU effort there, while finishing the linalg work (better SpMV
kernel, real multi-GPU) it started.

## Standing ethos (unchanged)

- **No NFS-math change that alters results.** Gate every change on `make check` +
  verified `product == N` (or bit-exact vs CPU/GMP; AVX-512 under Intel SDE).
- **Honest negatives recorded.** v3.1.0 rejected PGO (×2) and hot-scalar
  micro-opt; those avenues stay closed unless new evidence appears.
- **Hardware-gated work ships as code validated at the degenerate path + a design
  doc**, never as an unvalidated claim.

---

## Track A — Algorithm / new techniques

- **A1. 3-D lattice sieving** ("A New Angle on Lattice Sieving for the NFS",
  arXiv 2001.10860). **RE-SCOPED — original framing was a misread (honest
  correction).** The paper's 3-D (and higher-D) enumeration is for the **Tower NFS
  / discrete log in extension fields** (its headline is a 133-bit subgroup DLP in
  𝔽_{p⁶}, a 423-bit field, ≈3× the prior record); the extra dimensions exist
  because TNFS sieves over higher-degree elements `a + b·x + c·x² …`. For ordinary
  **integer factorization the relations are (a,b) pairs — inherently 2-D** — and
  CADO's factorization siever is 2-D *because of that*, not as a limitation (the
  q-lattice has exactly two basis vectors; survivors map (i,j)→(a,b)). So **there
  is no "seeded c100" 3-D win**: the stated gate is not achievable, and 3-D sieving
  would only help the fork's extension-field **DLP** side — research-grade, a large
  effort, and *not* "locally testable on a c100." This avenue is therefore closed
  for the factorization track (recorded like the PGO / column-reorder negatives);
  any future pursuit belongs under **A4 (DLP/exTNFS)**. The freed sequencing slot
  goes to the genuinely high-value sieving + polyselect work (C2 collision-search
  offload, B1 AVX-512 sieving, D3 sieve fan-out).
- **A2. Mixed-representation ECM cofactorization** (Springer 2020, "Faster
  Cofactorization with ECM Using Mixed Representations"). Upgrade the in-sieve
  `facul` ECM and the v3.1.0 GPU pre-factor/cofactor ECM to mixed
  Edwards/Montgomery representations → fewer modmuls per curve. Bit-exact
  relation-set gate; feeds C3.
- **A3. Parallel structured Gaussian elimination** for filtering/merge (the
  algorithm used for RSA-240/250). The 3.1.0 benchmark re-run showed **merge is
  the high-variance phase** (35–71 CPU-s at c70–c90); parallelizing it cuts that.
  CPU, testable.
- **A4. (stretch) DLP: exTNFS / Tower-NFS exploration** for the medium-
  characteristic discrete-log side the fork supports. Research-grade,
  correctness-only; document feasibility rather than commit speculative math.

## Track B — CPU performance (the honest envelope)

CPU tuning is nearly tapped (PGO/micro-opt rejected in 3.1.0). The real items are
AVX-512, which this box can only validate (SDE) not perf-measure:

- **B1. AVX-512 block + bucket sieving** (SciTePress/SECRYPT 2021): vectorize the
  sieve-index calculation, sieve-array updates, and bucket insertion. A
  *different, published AVX-512 approach* from the AVX2-on-siever path 3.1.0 ruled
  out — and it targets the 90 % phase. Correctness-only under Intel SDE + the CI
  gate; perf on an AVX-512 CI runner.
- **B2. Complete the AVX-512 VPCLMULQDQ gf2x port** (`mul2`–`mul9` + threshold
  retune; the Drucker–Gueron bigger gains are at 256/512-bit operands). 3.1.0
  shipped `mul1` + detection. SDE-validated, perf CI-gated.
- **B3. Finish the IFMA GF(p) backend → mpfq integration.** 3.1.0 has the
  validated standalone IFMA modmul kernel; wire it into mpfq's generated GF(p)
  arithmetic + the DLP BWC SpMV. SDE-validated.

## Track C — GPU, single machine

- **C1. Faster SpMV kernel** (ELL / hybrid / sliced-ELL + column reordering +
  shared-memory `src` caching). The 3.1.0 warp kernel realizes only ~10–15 % of
  the 3090's ~936 GB/s; Schmidt et al. reach 4–8× single-GPU with a hybrid sparse
  format. Bit-exact gate; **measurable now** with the existing SpMV sweep harness.
  *Sequenced #1.*
- **C2. GPU polynomial selection.** Proven in **msieve** (GPU Kleinjung stage-1
  since 2009); CADO lacks it. Polyselect grows with N and is embarrassingly
  parallel — a clean single-machine GPU win, reusing the 3.1.0 multi-precision
  device arithmetic. Gate: same polynomial quality (Murphy-E) as the CPU path.
- **C3. GPU batch-smoothness product tree** (the 3.1.0 documented design;
  Bernstein remainder tree). Flag-gated alternative to per-cofactor ECM in heavy-
  `mfb` regimes; reuses the validated device arithmetic (`bench/gpu-ecm-mp.cu`,
  `bench/ifma-modmul.c`). Bit-exact relation-set gate.
- **C4. (research) GPU sieving feasibility study.** Honest: NFS lattice sieving is
  memory-scatter-bound and largely unsolved on GPU (the "GPU tensor-core lattice
  sieving" work is for lattice *cryptanalysis*, a different problem). Scope as a
  *measured feasibility study*, not a promised feature.

## Track D — GPU at scale (multi-GPU / HPC / multi-machine)

The 3.1.0 designs, implementable where HW allows; correctness validated at the
degenerate path locally + on a multi-GPU cloud/CI runner; perf only from real HW.

- **D1. Intra-node multi-GPU matrix partition → real validation.** 3.1.0 shipped
  the `CADO_GPU_NPART` partition validated at N=1; finalize cross-device staging +
  per-device CUDA streams for overlap, validate on ≥2 GPUs. Schmidt et al. show
  GPU clusters beating larger CPU clusters on kilobit-SNFS matrices — the target.
- **D2. NVSHMEM / GPUDirect multi-node residency** (the 3.1.0 local-device-reduce
  / MPI-boundary-exchange split design). Use NVSHMEM (GPU-initiated PGAS comm over
  NVLink / InfiniBand-RDMA) to keep BWC vectors device-resident and exchange only
  across the MPI boundary, overlapping comm with the SpMV — the only path to a
  multi-node win per the 3.1.0 transfer accounting.
- **D3. Cluster / HPC orchestration.** Promote `scripts/cluster-launch.sh` to a
  real distributed driver: Slurm `sbatch` job arrays, GPU-aware client placement
  (one rank per GPU), and fan-out of the *sieving* stage (the 90 %) through the
  Rust work-unit server — the biggest practical lever.

## Track E — UI/UX & orchestration

Building on 3.1.0's `--json-status`, `/status`, `/dashboard`, clap CLIs, and
`cado-nfs-monitor-rs`:

- **E1. Real web dashboard** — phase timeline, per-phase ETA, live throughput +
  GPU/CPU utilization graphs, work-unit/client map (extend `/dashboard` into an
  SPA over the `/status` history).
- **E2. Per-machine parameter auto-tuner** ✓ **DONE.** `--autotune` calibrates the
  **safe scheduling knobs** to the host — local client/thread layout
  (`slaves.nrclients`) and work-unit granularity (`tasks.sieve.qrange`,
  `tasks.polyselect.adrange`, bounded √(threads/20) scaling). It **deliberately
  does not touch `lim*/lpb*/mfb*/I`** (the original framing): those set relation
  yield + matrix structure, so retuning them risks altering/breaking the
  factorization. Tuning only the *chunking* of identical work keeps `product == N`
  by construction (verified on the c59 smoke). `scripts/cadofactor/planner.py`.
- **E3. "Factor planner"** ✓ **DONE.** `--plan` / `--plan-json`: given `N` (+ the
  host thread count, `--host-speed`, GPU presence/build), prints feasibility, a
  wall-time envelope, the per-phase split, single-machine-vs-cluster strategy, and
  `--gpu-prefactor` / GPU-linalg triage, then exits. Model anchored on the measured
  BENCHMARKS c60–c90 sweep + the documented c100/c110 envelope, log-linear in
  digits, Amdahl-scaled to thread count. Doctested (`test_python_planner`).
- **E4. Checkpoint/resume + observability hardening** — robust mid-phase resume,
  structured JSON logs, optional OpenTelemetry metrics; container images
  (CUDA + CPU) and reproducible release artifacts.

---

## Cross-cutting

- **Correctness gate** after every change: seeded c60–c120 `product == N`;
  GPU/SIMD bit-exact; AVX-512 under SDE 10.8.0 (`/opt/intel-sde/sde64`).
- **CI:** extend `avx512-validate.yml`; add a multi-GPU cloud CI job (D1/D2) and an
  AVX-512 perf runner (B1/B2/B3).
- **Bench:** extend `bench/` + `BENCHMARKS.md`; record honest negatives.
- **Version/docs:** `CMakeLists.txt` `3.1.0 → 3.2.0-modern`; new
  `## [3.2.0-modern]` CHANGELOG; new docs (`docs/gpu-polyselect.md`,
  `docs/avx512-sieving.md`, a multi-GPU deep-dive).

## Sequencing & honest expected payoff

| # | Item | Effort | Risk | Honest payoff (this box → at scale) |
|---|------|--------|------|--------------------------------------|
| 1 | **C1** better SpMV kernel | Med | Low | measurable now; bigger at scale |
| 2 | **C2** GPU polyselect | Med | Low–Med | **real single-machine win** (proven in msieve) |
| 3 | ~~**A1** 3-D lattice sieving~~ → **C2 collision-search offload** | High | Med | A1 closed (TNFS/DLP only, not c100 — honest correction); the collision-search GPU offload is the real polyselect win |
| 4 | **E2/E3** autotuner + planner ✓ DONE | Med | Low | UX + cuts variance; no raw speed |
| 5 | **A2** mixed-rep ECM | Med | Med | feeds C3 + CPU cofactor |
| 6 | **A3** parallel merge | Med | Med | cuts the high-variance filtering phase |
| 7 | **D1** multi-GPU partition (real) | High | High | **the large-N / HPC win** (needs ≥2 GPUs) |
| 8 | **B1/B2/B3** AVX-512 sieving + gf2x + IFMA | Med–High | Low | AVX-512-HW-only (SDE-correct, CI-perf) |
| 9 | **D2** NVSHMEM multi-node residency | High | High | cluster win (needs CUDA-aware MPI + multi-GPU) |
| 10 | **C3 / C4 / A4** product-tree · GPU-sieve study · exTNFS | High | High | research / regime-specific |

**Net north star:** put the GPU + algorithm effort where the cost actually is —
**polynomial selection (C2) and sieving (A1, B1, D3)** — plus a **faster SpMV
kernel (C1)** and finally realising the **multi-GPU / multi-node linalg win
(D1/D2)** on real hardware, wrapped in a genuinely usable **autotuner + planner +
dashboard (E)**.

---

## Sources

- New angle on lattice sieving (3-D enumeration): arXiv:2001.10860
- AVX-512 block/bucket sieving: SciTePress/SECRYPT 2021 (105152)
- Multi-GPU Block Wiedemann over GF(2), hybrid sparse format: Schmidt et al.,
  Concurrency & Computation 2013 (DOI 10.1002/cpe.2896)
- RSA-250 cost breakdown: Zimmermann, *Factoring RSA-250 with PRACE*
- Parallel structured Gaussian elimination: *Mathematical Cryptology* 0(1)
- Faster cofactorization with ECM (mixed representations): Springer 2020
  (10.1007/978-3-030-45388-6_17); GPU cofactorization: eprint 2014/397
- GPU polynomial selection (Kleinjung stage-1): msieve (`Readme.nfs`)
- NVSHMEM / GPUDirect: NVIDIA developer docs; RDMA SpMV on GPUs (arXiv:2311.18141)
- Surveys: *Advancements and Prospects in Large Integer Factorization* (2024);
  *Integer Factorization: Another perspective* (arXiv:2507.07055, 2025)
