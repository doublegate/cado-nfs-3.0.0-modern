# CADO-NFS 3.4.0-modern

A **modernization + performance fork** of
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) — a complete
implementation in C / C++ / Python of the **Number Field Sieve (NFS)**, the
fastest known algorithm for factoring large integers and computing discrete
logarithms in finite fields.

> **What this fork is.** It rebases the upstream 3.0.0 codebase onto a current
> bleeding-edge toolchain (**CMake 4.x, GCC 16, CUDA 13, Python 3.14**) and adds
> several tracks of *rigorously measured* performance and orchestration work. From
> **3.1.0-modern** the fork carries its own minor line — still upstream 3.0.0's NFS
> algorithms, numerics, and parameters (**unchanged**), the bump reflecting
> substantial original work. The per-cycle arc:
>
> - **3.2.0-modern** aims the GPU + algorithm effort where the cost actually is
>   (sieving and polynomial selection, ~91 % of an RSA-250-scale run): a
>   **validated GPU mixed-representation (twisted-Edwards) ECM** (~1.5–2.9× the
>   Montgomery ladder), **GPU polynomial-selection collision offload**, an
>   **adaptive GPU SpMV** kernel, **AVX-512 VPCLMULQDQ `mul2/3/4` + IFMA GF(p) +
>   batched modular-inverse** kernels (all SDE-validated), real **multi-GPU
>   partition** (per-device streams, `product == N` at scale) with a **multi-node
>   NVSHMEM/GPUDirect design**, **cluster sieving orchestration** (Slurm job arrays,
>   GPU-aware placement), and a **factor planner + per-host autotuner** — on top of
>   the 3.1.0 GPU-linalg / GPU-prefactor / AVX-512 / orchestration tracks and the
>   3.0.0-modern build/SIMD/GPU-cofactor/Rust base.
> - **3.3.0-modern** opens on the honest premise the fork has now confirmed three
>   times — single-machine NFS *speed* is essentially tapped out on this hardware —
>   and pivots accordingly: a shippable **operator-experience core** (a live
>   dashboard + trend-ETA, a **`--doctor`** preflight, shell completions + man
>   pages, checkpoint/resume clarity, **Slurm/PBS** integration), the fork's
>   **first measured-on-silicon SIMD** kernel (**AVX2 batched modular inverse,
>   ~4.6×**, bit-exact), **Galois-automorphism auto-detection** (a genuine,
>   measurable algo win), plus an honestly-gated GPU/DLP research track (GPU
>   root-sieve, GPU GF(p) lingen NTT, IFMA→arith-modp routing, an exTNFS skeleton)
>   — each a validated, measured kernel reported with its honest non-win where it
>   is one.
> - **3.4.0-modern** keeps that shape and lands the one materially new opportunity
>   the codebase exploration surfaced: the GPU pre-NFS factoring front-end is the
>   **one stage with no Amdahl ceiling**, so it gets the cycle's **headline measured
>   win** — GPU **Pollard P-1 / Williams P+1** beside the batched ECM, on the
>   bit-exact Montgomery core, with an adaptive escalating-B1 schedule (**14/14
>   bit-exact vs GMP, ~3.3× faster time-to-strip** on a P±1-smooth factor). Around
>   it ships a real **observability/usability core** — completion **notifications**
>   (ntfy / Slack / Discord / webhook / email / desktop), a structured **JSON event
>   log** + a Prometheus **`/metrics`** endpoint on both servers, a **multi-run
>   history DB** (`--list-runs` / `--compare-runs`), **per-phase + all-phases ETA**,
>   a **`--wizard`** parameter TUI — plus a **data-driven autotuner** (`--calibrate`
>   + a regression cost model on the run history that sharpens `--plan`,
>   number-theoretic bounds untouched), and the carry-forward research track (GPU
>   root-sieve launch threshold C5+, GPU GF(p) lingen NTT multi-modular CRT C6+).
>
> **How it's gated.** Every change is gated on `make check` + verified
> `product == N` (or a bit-exact / SDE validation); hardware-blocked items
> (multi-GPU/multi-node, AVX-512 perf) ship as documented designs or
> correctness-validated kernels, not unvalidated claims. Results are reported
> honestly, including the parts that did **not** pay off and the work that turned
> out **already optimal upstream**.
>
> **Not the official CADO-NFS.** For releases, ongoing development, and support, use
> the upstream project (links at the bottom). Earlier releases are preserved under
> their tags: `2.3.1-modern`, `v3.0.0-modern`, `v3.1.0-modern`, `v3.2.0-modern`,
> `v3.3.0-modern`.

[![CI](https://github.com/doublegate/cado-nfs-modern/actions/workflows/ci.yml/badge.svg)](https://github.com/doublegate/cado-nfs-modern/actions/workflows/ci.yml)
[![AVX-512 validation](https://github.com/doublegate/cado-nfs-modern/actions/workflows/avx512-validate.yml/badge.svg)](https://github.com/doublegate/cado-nfs-modern/actions/workflows/avx512-validate.yml)
[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL%202.1-blue.svg)](COPYING)
[![Upstream: 3.0.0](https://img.shields.io/badge/upstream-CADO--NFS%203.0.0-informational.svg)](https://gitlab.inria.fr/cado-nfs/cado-nfs)
[![Release: 3.4.0-modern](https://img.shields.io/badge/release-3.4.0--modern-success.svg)](https://github.com/doublegate/cado-nfs-modern/releases/tag/v3.4.0-modern)

**Jump to:** [Quick start](#quick-start) · [Performance](#performance) ·
[What this fork changes](#what-this-fork-changes-vs-upstream-300) ·
[How it works](#how-it-works) · [Documentation](#documentation) ·
[License](#license) · [Credits](#credits-and-attribution)

## Quick start

### Dependencies

- A C/C++ compiler conforming to **C99 + C++20** — GCC ≥ 10, LLVM Clang ≥ 12,
  Apple Clang ≥ 16, or Intel ICX ≥ 2023 (tested through GCC 16)
- [GMP](https://gmplib.org/) ≥ 5, **built with `--enable-shared`**
- CMake ≥ 3.18 (CMake 4.x works)
- Python 3 with **Flask** and **requests** (the work-unit server is Flask in
  3.0.0) — install them into a venv with the helper below
- Optional: `hwloc` (CPU pinning), MPI (clusters), `curl`, CUDA (the GPU backend)

On Debian/Ubuntu:

```bash
sudo apt-get install -y build-essential cmake libgmp-dev libhwloc-dev \
                        python3 python3-venv python3-pip
```

### Build

```bash
bash scripts/setup-venv.sh   # ONE-TIME: create cado-nfs.venv (Flask + requests)
make                         # out-of-tree build into build/<hostname>/
make check                   # run the test suite (ctest); ARGS="-R <regex>" to filter
```

> **Why the venv first?** CADO-NFS 3.0.0 needs Flask/requests at *configure*
> time, and the committed [`local.sh`](local.sh) points cmake at
> `cado-nfs.venv/bin/python3`. `local.sh` also carries this fork's performance
> flags (`-O3 -march=native -mtune=native`). Edit it and run `make cmake` to
> reconfigure; `-march=native` is host-specific, so override it for a
> distributable build.

### Factor a number

```bash
cado-nfs.venv/bin/python3 ./cado-nfs.py \
  90377629292003121684002147101760858109247336549001090677693 -t 4
```

This runs the full pipeline (polynomial selection → sieving → filtering →
linear algebra → square root) over a local work-unit server and prints the
prime factors; the 59-digit demo finishes in ~15 s on a modern desktop.
CADO-NFS targets numbers **> 85 digits**; **< 60 digits is unsupported**, and
you should strip small factors first.

> **CLI note:** `key=value` parameters must come *before* flags like `-t`
> (e.g. `./cado-nfs.py <N> server.ssl=no -t 4`). Pass `server.ssl=no` for plain
> HTTP instead of TLS.

For larger and distributed factorizations, discrete logarithms, and the full
parameter reference, see the preserved upstream guide
([`README.upstream.md`](README.upstream.md)),
[`README.dlp`](README.dlp), and [`README.Python`](README.Python).

## Performance

Reference timings factoring balanced (RSA-like) semiprimes on an
**Intel i9-10850K (10 cores / 20 threads), 64 GiB RAM, CachyOS**, GCC 16.1.1
`-O3 -march=native`, GMP 6.3.0, all 20 threads. Same seeded inputs as the prior
`2.3.1-modern` benchmarks, so the comparison is apples-to-apples. Every result
was verified (factors multiply back to the input and are prime).

| Digits | Bits | Wall time | CADO CPU | Parallel | vs 2.3.1-modern (CPU) |
|-------:|-----:|----------:|---------:|---------:|----------------------:|
| 60 | ~199 | 18.5 s | 31.1 s | 1.7× | −46 % |
| 70 | ~232 | 27.2 s | 108.5 s | 4.1× | −11 % |
| 80 | ~265 | 74.4 s | 406.0 s | 5.5× | −27 % |
| 90 | ~299 | 197.9 s | 1604.2 s | 8.1× | −17 % |

(Re-confirmed 2026-06-07 on 3.2.0-modern; like 3.1.0, 3.2.0 adds **no CPU-path
change** — all new work is GPU / AVX-512 (SDE-validated) / orchestration — so these
match the 3.0.0/3.1.0-modern line within run-to-run variance.)

**Key findings**

- **Total CPU work is down (~25-32 % typical; −11 to −46 % this run)** versus the
  2.3.0-based fork on identical inputs — mostly from upstream 3.0.0's
  Bouvier–Imbert batch cofactorization (eprint 2018/669) and `I>16` sieving,
  compounded by this fork's `-O3 -march=native` (~7 % on the siever; the
  deterministic siever microbench is **11.67 s**). The per-run spread is
  polyselect/merge variance; the CPU reduction is the robust, repeatable signal.
- **Parallel efficiency *drops* (e.g. c90 11.1×→8.1×) as a consequence of the
  CPU reduction**, not a regression: with less embarrassingly-parallel sieve
  work, the sequential phases (linear algebra, square root, orchestration)
  become a larger fraction of the run.
- **Sieving dominates** (45-74 % of CPU) and is the parallel phase; **linear
  algebra grows the fastest** (~110× from c60 to c90) and is the emerging second
  bottleneck — the classic NFS trade-off. Wall-time roughly doubles per +10
  digits, matching the sub-exponential `L(1/3)` complexity of NFS.
- **Practical envelope on this desktop:** ≤c75 interactive · c80-c95 a few
  minutes · ~c100 ≈ 10 min · ~c110 ≈ 1 hr · ≥c130 wants distributed mode.

**GPU (RTX 3090).** The GPU pre-factoring ECM front-end runs **48.7× / 25.4× /
10.5×** the full 20-thread CPU at 128/256/512-bit moduli; the GPU BWC SpMV holds
**8.1 Gnz/s at c120-scale (240 M nonzeros, bit-exact)** — ~4.5× the tuned CPU
`bucket` backend (saturated ~1.8 Gnz/s), widening as the matrix grows. Full vector
residency removes the per-iteration PCIe transfers (end-to-end c90 `product == N`).
**New in 3.2.0:** an **adaptive sub-warp SpMV** kernel adds **1.3–1.8×** in the
cache-resident regime (**43.5 Gnz/s** at c100-scale); a **twisted-Edwards
mixed-representation ECM** stage-1 is **~1.5–2.9×** the Montgomery ladder
(bit-exact, growing with modulus width); GPU **polynomial-selection collision
offload** and a **batch-smoothness leaf** kernel are validated bit-exact; the
**multi-GPU matrix partition** now uses per-device streams and was validated
`product == N` on a full c90 run at `CADO_GPU_NPART=2`. AVX-512 VPCLMULQDQ
`mul2/3/4`, IFMA GF(p), and a batched modular-inverse kernel are all bit-exact
under Intel SDE. GPU results need `-DENABLE_GPU=ON`; the CPU table above is the
default build.

Full methodology, per-phase breakdown, the 2.3.1→3.0.0 comparison, the GPU
sweeps, projections, and seeded reproducible inputs:
[**`BENCHMARKS.md`**](BENCHMARKS.md).

## What this fork changes (vs upstream 3.0.0)

Upstream 3.0.0 already subsumes the toolchain-portability fixes the prior
`2.3.1-modern` fork carried (hwloc 2.x, Python-3 stdlib moves, OpenSSL 3.x) and
brings, for free, the batch cofactorization and `I>16` sieving above. On top of
that, this fork adds four independently-validated tracks:

| Track | What | Result |
|-------|------|--------|
| **1 · Build / compiler** | `local.sh`: `-O3 -march=native -mtune=native`; LTO and PGO evaluated | **~7 % on the siever** (host-ISA codegen). LTO/PGO were *measured and rejected* (0 % / −2.8 %). |
| **2 · SIMD** | AVX2 on the siever profiled; an AVX-512 **VPCLMULQDQ** gf2x base-case kernel added | AVX2 ruled out (the hot path is scatter + scalar modarith, not vectorizable). The gf2x kernel is **validated bit-exact** under Intel SDE; perf is gated on real AVX-512 silicon (the reference box is Comet Lake). |
| **3 · GPU ECM cofactorization** | A batched CUDA ECM backend (`sieve/ecm/gpu/`) behind `facul`/`las-cofactor`, validated bit-exact vs the CPU path | The GPU modmul primitive is ~39× a 20-core CPU — but cofactorization is only ~8 % of siever time, so **honestly, no net single-machine speedup at these sizes** (Amdahl-bounded). Documented as a measured negative, not an unsubstantiated win. |
| **4 · Rust orchestration** | The `rust/` workspace — a static-binary work-unit **client** and an async **server**, same HTTP/JSON protocol + `wudb` SQLite schema | Interoperates with an unmodified `cado-nfs.py`; the Rust server can be **swapped in** for the Flask server live (`CADO_RUST_WU_SERVER=…`), with TLS, IP-whitelist, and cert-pinning. For multi-client robustness, not single-machine speed. |

### New in 3.4.0-modern

3.4.0 keeps the v3.3.0 shape — a shippable, measurable core plus an honestly-gated
research track — and adds the one materially new opportunity the codebase
exploration surfaced: **the GPU pre-NFS factoring front-end is the only pipeline
stage with no Amdahl ceiling**, so it gets the cycle's headline *measured* win.

| Track | What | Result |
|-------|------|--------|
| **C7 — GPU prefactor P-1/P+1 (headline, measured)** | Pollard **P-1** + Williams **P+1** (stage-1 + stage-2 BSGS) beside the batched ECM, on the bit-exact Montgomery core (`misc/gpu_prefactor/gpu_pm1_pp1.cuh`), interleaved under an **adaptive escalating-B1** schedule that stops once the cofactor is prime/1 | **14/14 bit-exact** vs CPU **and** a GMP reference (RTX 3090); every factor re-verified `f∣N`; **~3.3× faster time-to-strip** on a P-1-smooth factor, ~30 ms when they find nothing. Honest: one sequence = one lane (coverage + cheaper strip, not GPU throughput — that stays ECM). |
| **Usability / observability core (E9–E12)** | Completion/failure **notifications** (ntfy/Slack/Discord/webhook/email/desktop); a structured NDJSON **event log** (`--json-log`) + a Prometheus **`/metrics`** endpoint on both servers; a **multi-run history DB** (`--list-runs`/`--compare-runs`); **per-phase + all-phases ETA** in the monitor + `/dashboard`; a **`--wizard`** parameter TUI; path-aware completions + man EXAMPLES | All runs here, all default-off and additive (no NFS-math change). `notify.py`/`runs.py`/`wizard.py` doctested; secrets stay out of the parameter snapshot; notification/recording failures are isolated from the run. |
| **A7 — Data-driven autotuner** | **`--calibrate`** backs out this host's per-core speed from `~/.cado-nfs/runs.db`; a **log-linear regression cost model** over the run history folds an empirical estimate into `--plan` | Seeded validation: recovers **1.500×** and R²=0.978; sharpens the plan as the history grows. **Number-theoretic bounds (`lim*`/`lpb*`/`mfb*`/`I`) are never touched** — prediction only. |
| **Research track (C5+/C6+)** | A **conditional-launch threshold** that unlocks the v3.3.0 GPU root-sieve at large N with no small-N regression; a **multi-modular CRT wrapper** around the GPU GF(p) lingen NTT (the piece C6 said real coefficients need) | C5+ bit-exact at every size, heuristic routes to the measured-faster path (crossover ~16 M-cell lines, **~3.9×** at large N); C6+ CRT **== `__int128` convolution, 0/2999 wrong**. Honestly scoped: large-N / cluster-DLP, not a desktop win. |

### New in 3.3.0-modern

3.3.0 opens on the conclusion the prior three revisions measured repeatedly:
**single-machine NFS *speed* is essentially tapped out on this hardware** (Comet
Lake i9-10850K — AVX2, no AVX-512; one RTX 3090). So it splits, transparently, into
a **shippable operator-experience core** (the real, here-and-now value) and an
**honestly-gated research track** — each item validated and *measured*, reported
with its honest non-win where it is one.

| Track | What | Result |
|-------|------|--------|
| **Usability core (E4–E8)** | A live TUI dashboard with **trailing-window ETA** + throughput + host CPU/GPU; a **`--doctor`** preflight (build/GPU/CPU/RAM/disk/env → GO·NO-GO); **shell completions** (bash/zsh/fish) + a **man page**; **`--checkpoint-interval`**; **Slurm/PBS** integration + **`--suggest-{slurm,pbs}-config`** | All runs on this box today; the genuine high-ROI center. `doctor.py` doctested; completions generated from the argparse spec; PBS mirrors the existing Slurm `sbatch`. |
| **AVX2 modinv (B4)** | The siever's per-prime modular inverse as an **AVX2 8-way** masked binary-GCD (`bench/avx2-modinv.c`), ported from the SDE-only AVX-512 B1 kernel | The fork's **first measured-on-silicon SIMD**: **~4.6× scalar**, bit-exact vs GMP (0/320000), **native** (no SDE). Honest: Amdahl-bounded whole-siever (the byte-scatter majority stays scalar). |
| **Galois auto-detect (A5)** | An exact automorphism detector (`galois.py`) — Möbius-invariance in integer arithmetic with the orbit guard — exposed as **`--galois-detect`** | Cross-validated against CADO's own `tests/sieve/galois.poly` (`autom2.2`) and a cyclic cubic (`autom3.1`); correct no-op on generic GNFS. The matrix/sieve reduction is CADO's upstream `--galois`. |
| **Research track (C5/C6/B5/A6)** | GPU stage-2 root-sieve; GPU GF(p) lingen **NTT**; the **IFMA→arith-modp** routing bridge; an **exTNFS** feasibility skeleton | All validated bit-exact (C5 0-wrong vs int16; C6 0/1199 vs schoolbook; B5 0/32000 under SDE) and **honestly scoped**: C5 is a per-rotation wash at testable sizes, C6/B5 are multi-GPU/DLP/HW-gated, A6 is documented design only. |

### New in 3.2.0-modern

3.2.0 aims the GPU + algorithm effort where an RSA-scale run's cost actually lives
— **sieving and polynomial selection (~91 %)**, not just linear algebra (~9 %) —
and finishes the AVX-512 and multi-GPU tracks. Every entry is gated on `product ==
N`, a bit-exact check, or Intel-SDE validation; several investigations honestly
concluded **"already optimal upstream"** or **"measured negative."**

| Track | What | Result |
|-------|------|--------|
| **GPU mixed-rep ECM (A2)** | A twisted-Edwards `a=−1` extended-coordinate stage-1 (`bench/gpu-ecm-edwards.cu`) for the GPU pre-factor/cofactor ECM, with double-and-add and wNAF | **Bit-exact vs the Montgomery ladder** through the birational map (0/8192 at 128/256/512-bit); wNAF is **~1.5–2.9×** the ladder, the win **growing with modulus width**. (The CPU `facul` path already does mixed-rep — the upstream "mishmash" bytecode.) |
| **GPU polyselect + SpMV (C1/C2)** | An **adaptive sub-warp SpMV** (vec16) kernel; **GPU collision-search offload** for Kleinjung stage-1, size-gated behind `--gpu-polyselect` | SpMV **1.3–1.8×** the warp kernel cache-resident (43.5 Gnz/s); polyselect collisions produce a **byte-identical** polynomial set, `product == N`. |
| **AVX-512 (B1/B2/B3)** | VPCLMULQDQ gf2x `mul2/3/4`; IFMA GF(p) plain-representation `plain_mul`/`vec_add_dotprod`; a 16-way **batched modular inverse** for the siever's per-prime lattice setup | All **bit-exact under Intel SDE** (gf2x 0/200000; IFMA 0/32000; modinv 0/640000), CI-gated. Honest: the siever's byte-scatter majority does **not** vectorize on AVX-512 (no 8-bit scatter); perf gated on real AVX-512 silicon. |
| **Multi-GPU / HPC (D1/D2/D3)** | Per-device CUDA streams for the `CADO_GPU_NPART` partition; a multi-node **NVSHMEM/GPUDirect** residency design; **cluster sieving** orchestration (Slurm `sbatch` job arrays, GPU-aware one-client-per-GPU placement) | Partition validated **`product == N` on a full c90** at `NPART=2`; multi-node is a HW-gated design (single-rank degenerate path validated); cluster driver validated across SSH/srun/sbatch. |
| **Algorithm + UX (A3/C3/E2/E3)** | Parallel structured Gaussian elimination (merge); GPU batch-smoothness leaf; a **factor planner** (`--plan`) + **per-host autotuner** (`--autotune`) | Merge is **already the parallel B–Z code** (verified ~3.3× at 8 threads); the batch-smooth leaf is bit-exact vs GMP (the tree stays CPU/GMP); `--plan` estimates feasibility/wall-time/strategy; `--autotune` tunes only safe scheduling knobs (`product == N` preserved). |

The honest through-line continues: **CPU-side and several algorithm tracks are
already optimal upstream** (the parallel merge and the CPU mixed-rep ECM are the
RSA-record code), so the fork's measured wins concentrate in the **GPU fixed-width
tracks** and **orchestration**. exTNFS-DLP and GPU-lattice-sieving were studied and
**documented as research-grade / measured-negative**, not pursued.

### New in 3.1.0-modern

Building on that base, 3.1.0 aims the GPU at the *linear-algebra* phase (where
the structural headroom actually is), adds a separate GPU factoring stage, and
rounds out the orchestration/UX layer:

| Track | What | Result |
|-------|------|--------|
| **GPU linear algebra** | A real `mm_impl=gpu` BWC SpMV backend (`linalg/bwc/matmul-gpu.cu`) with M+Mᵀ resident as CSR, a coalesced warp-per-row kernel, and **full vector residency** (the steady krylov/mksol/secure loop runs entirely on device buffers — 2D comm, `x_dotprod`, and `addmul_tiny` all ported) | Bit-exact (`bench_matcache` 4/4) and end-to-end `product == N`. The win **grows with N**: GPU warp SpMV holds **7.9 Gnz/s at c120-scale** while the CPU reference loop collapses — ~4.4× the tuned `bucket` backend, 18×→41× the reference loop across 1M→8M rows. Intra-node multi-GPU partition (`CADO_GPU_NPART`); multi-node residency is a documented design (HW-gated). |
| **GPU pre-NFS factoring** | `cado-nfs.py --gpu-prefactor`: batched multi-precision GPU ECM (stage-1 + stage-2 BSGS + Suyama, multi-GPU, escalating-`B1`) strips factors *before* NFS — a separate stage with no Amdahl ceiling | **~49× / 26× / 12×** the full CPU at 128/256/512-bit on an RTX 3090; end-to-end `product == N`. Unlike in-sieve cofactorization (Track 3), this is a real single-machine win when `N` has a findable factor. |
| **AVX-512 (VPCLMULQDQ + IFMA)** | gf2x VPCLMULQDQ auto-detection (run-test-gated, safe on non-AVX-512 hosts) + an IFMA GF(p) Montgomery modmul kernel | Both **bit-exact under Intel SDE** (`mul1`; IFMA modmul 0/32000 vs GMP), CI-gated (`avx512-validate.yml`). Correctness-only; the ~39 % perf gain is AVX-512-hardware-gated. |
| **Orchestration / UX** | Parameter interpolation + `--suggest-params`; a `/status` endpoint + `/dashboard` on both servers; `clap` CLIs, a `cluster-launch` helper (SSH/Slurm), and a `ratatui` terminal monitor (`cado-nfs-monitor-rs`) | Off-preset sizes get interpolated params instead of an error (`product == N` on a real c45); live phase/progress/ETA/factors; one-command cluster fan-out. |

The honest through-line: **CPU-side tuning is nearly tapped out** on this
hand-optimized codebase (PGO was re-measured at +3 % and rejected again; the hot
loops are already SIMD/unrolled/prefetched). The structural headroom is in new
compute resources — and 3.1.0 confirms it: the **GPU linear-algebra win grows
with N** (the fastest-growing NFS phase), and the **GPU pre-factoring front-end**
is a clean single-machine win, while AVX-512 is correctness-validated and
perf-gated on hardware. Full rationale and measurement methodology are in
[`CHANGELOG.md`](CHANGELOG.md); the GPU, GPU-linalg, GPU-prefactor, and Rust work
each have a dedicated deep-dive (see [Documentation](#documentation)).

## How it works

New to the Number Field Sieve? Two companion explainers, by background:

- **Plain English, no math** —
  [`docs/number-field-sieve-plain-english.md`](docs/number-field-sieve-plain-english.md):
  a friendly, analogy-driven tour for any curious reader (why factoring is hard,
  what the program actually does, and why it matters for online security).
- **The mathematics** —
  [`docs/number-field-sieve.md`](docs/number-field-sieve.md): the rigorous
  version — congruence of squares, the two-polynomial number-field
  construction, smoothness and lattice sieving, the $\mathbb{F}_2$ linear
  algebra, the algebraic square root, complexity, and how each phase maps to the
  directories in this tree.

The pipeline, stage by stage (each maps to a top-level directory):
**polynomial selection** (`polyselect/`) → **lattice sieving** (`sieve/`) →
**filtering** (`filter/`) → **linear algebra / Block Wiedemann** (`linalg/`) →
**algebraic square root** (`sqrt/`), orchestrated by `cado-nfs.py` /
`scripts/cadofactor/` (or the Rust layer in `rust/`).

## Documentation

| Document | What it covers |
|----------|----------------|
| [`README.upstream.md`](README.upstream.md) | Upstream 3.0.0's full build / usage / distributed / troubleshooting guide |
| [`docs/number-field-sieve-plain-english.md`](docs/number-field-sieve-plain-english.md) | The NFS, in layman's terms |
| [`docs/number-field-sieve.md`](docs/number-field-sieve.md) | The NFS mathematics, in depth |
| [`docs/gpu-linalg.md`](docs/gpu-linalg.md) | GPU BWC SpMV backend + full vector residency: kernel, transfer analysis, at-scale sweep, multi-GPU partition (per-device streams) |
| [`docs/gpu-ecm-mixedrep.md`](docs/gpu-ecm-mixedrep.md) | **(3.2.0)** GPU twisted-Edwards mixed-representation ECM: bit-exact vs the ladder, ~1.5–2.9× (CPU already does it upstream) |
| [`docs/gpu-prefactor.md`](docs/gpu-prefactor.md) | GPU pre-NFS ECM factoring front-end: why it sidesteps Amdahl, the multi-precision Montgomery ECM, measured CPU-vs-GPU |
| [`docs/gpu-prefactor-pm1pp1-c7.md`](docs/gpu-prefactor-pm1pp1-c7.md) | **(3.4.0, C7 — headline)** GPU Pollard P-1 / Williams P+1 + adaptive escalating-B1 on the prefactor front-end: 14/14 bit-exact vs GMP, ~3.3× faster time-to-strip |
| [`docs/usability-v340.md`](docs/usability-v340.md) | **(3.4.0, E9–E12 + A7)** notifications, NDJSON event log + Prometheus `/metrics`, run-history DB, per-phase ETA + `--wizard`, the data-driven autotuner |
| [`docs/ROADMAP-v3.4.0-modern.md`](docs/ROADMAP-v3.4.0-modern.md) | **(3.4.0)** the cycle anchor: the "speed tapped out" premise, the one new opportunity (the no-Amdahl prefactor stage), the track map + gates |
| [`docs/gpu-cofactorization.md`](docs/gpu-cofactorization.md) | GPU ECM cofactorization: measured results + honest Amdahl analysis (+ cofactor scale-out / product-tree designs) |
| [`docs/gpu-batch-smooth-c3.md`](docs/gpu-batch-smooth-c3.md) · [`docs/gpu-sieving-c4.md`](docs/gpu-sieving-c4.md) | **(3.2.0)** GPU batch-smoothness leaf (validated); GPU lattice-sieving feasibility (measured negative) |
| [`docs/avx512-sieving-b1.md`](docs/avx512-sieving-b1.md) · [`docs/ifma-gfp-b3.md`](docs/ifma-gfp-b3.md) | **(3.2.0)** AVX-512 batched modular inverse for the siever; IFMA GF(p) for the BWC backend (both SDE-validated) |
| [`docs/parallel-merge-a3.md`](docs/parallel-merge-a3.md) · [`docs/extnfs-a4.md`](docs/extnfs-a4.md) · [`docs/multinode-residency-d2.md`](docs/multinode-residency-d2.md) | **(3.2.0)** Parallel merge (already upstream, verified); exTNFS-DLP feasibility; multi-node NVSHMEM/GPUDirect residency design |
| [`docs/rust-orchestration.md`](docs/rust-orchestration.md) | The Rust client/server, the work-unit protocol, the in-process swap, and `cluster-launch.sh` (Slurm/SSH/GPU-aware) |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Performance sweep, per-phase breakdown, methodology, projections (incl. GPU pre-factoring + GPU linalg + the 3.2.0 GPU/AVX-512 additions) |
| [`CHANGELOG.md`](CHANGELOG.md) | Everything this fork changed, with rationale (3.4.0-modern · 3.3.0-modern · 3.2.0-modern · 3.1.0-modern · 3.0.0-modern · 2.3.1-modern) |
| [`CLAUDE.md`](CLAUDE.md) | Build/test/run notes and the fork's internal map |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) · [`SECURITY.md`](SECURITY.md) | How to contribute here vs upstream; security-reporting policy |
| [`README.dlp`](README.dlp) · [`README.Python`](README.Python) | Discrete logarithms; Python orchestration internals (upstream) |

## License

CADO-NFS is free software under the **GNU Lesser General Public License,
version 2.1** — see [`COPYING`](COPYING). This fork preserves that license and
all upstream copyright and authorship; the original authors are credited in
[`AUTHORS`](AUTHORS). The modifications described above are released under the
same LGPL-2.1 terms.

## Credits and attribution

- **Original work:** the CADO-NFS development team (INRIA / LORIA, Nancy,
  France) — Shi Bai, Cyril Bouvier, Pierrick Gaudry, Alexander Kruppa,
  Emmanuel Thomé, Paul Zimmermann, and many others (see [`AUTHORS`](AUTHORS)).
- **Upstream project:** <https://gitlab.inria.fr/cado-nfs/cado-nfs> ·
  homepage <http://cado-nfs.inria.fr/> · GitHub mirror
  <https://github.com/cado-nfs/cado-nfs>
- **Citation:** *CADO-NFS, An Implementation of the Number Field Sieve
  Algorithm.* See [`AUTHORS`](AUTHORS) for the BibTeX entry; per upstream's own
  note, set the release number to the version you actually used (here, 3.0.0).
- **This modernization + performance fork:** maintained by
  [@doublegate](https://github.com/doublegate).

If you use CADO-NFS in academic work, please cite the **upstream** project, not
this fork.
