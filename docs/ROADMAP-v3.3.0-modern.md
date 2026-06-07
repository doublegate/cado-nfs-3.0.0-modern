# Roadmap — v3.3.0-modern

This is the planning anchor for the `3.3.0-modern` development cycle, the successor
to [`ROADMAP-v3.2.0-modern.md`](ROADMAP-v3.2.0-modern.md). It records *why* this
cycle is shaped the way it is, the per-track scope, and the honesty gates each item
must pass. Per-track deep-dives are linked as they land.

## The honest premise

CADO-NFS factors a large integer `N` in five stages — **polynomial selection**
(pick two polynomials sharing a root mod `N`), **lattice sieving** (collect
"relations": values smooth over a factor base on both sides), **filtering**
(merge/purge the relations into a sparse matrix), **linear algebra** (find a
dependency in that matrix with Block Wiedemann), and the **algebraic square root**
(turn the dependency into a factor). The wall-clock is dominated by sieving (~91 %
at RSA-250 scale) and, far behind it, linear algebra (~9 %).

Across three revisions this fork has measured, and re-measured, the same wall:
**on the reference box — an Intel i9-10850K (Comet Lake: AVX2, *no* AVX-512) and a
single RTX 3090 — single-machine NFS *speed* is essentially tapped out.**

- CPU tuning is saturated: `-O3 -march=native` was the only real win; LTO/PGO and
  AVX2-on-the-siever were measured and rejected.
- GPU **cofactorization** is **Amdahl-capped**: it is ~8 % of sieve time, so even a
  10× kernel speedup is <1 % of the whole run. (This is why GPU ECM *stage-2*
  completion was deliberately **not** chosen for this cycle.)
- GPU **lattice sieving** is a **measured negative** (C4): GPU scatter beats a CPU
  socket on the apply step alone, but byte-atomics + on-GPU generation + capacity
  are unsolved.
- The AVX-512 **B-series** (B1/B2/B3) is **bit-exact under Intel SDE but
  silicon-gated** — there is no AVX-512 on this CPU to measure a speedup on.
- **Multi-GPU** (D1) and **multi-node** (D2) are correctness-validated only at the
  degenerate single-device path; true throughput needs hardware the box lacks.

A 2026 internet survey (upstream CADO on INRIA GitLab; msieve, YAFU, GGNFS, FLINT;
GMP-ECM/CGBN; the RSA-250 and DLP-240 record papers; 2021–2026 eprint/arxiv)
confirms two things: **(1)** no published technique since ~2010 delivers a >5 %
single-machine NFS speedup, and **(2)** this fork is already *ahead of every public
implementation* on GPU and SIMD modernization (upstream has no GPU polyselect, only
2017-era GPU linalg, no GPU sieving; msieve has GPU polyselect stage-1 but it only
helps beyond ~140 digits; YAFU is AVX-512 but CPU-only).

## The shape this dictates

Rather than promise a single-machine speed win the evidence won't support, this
cycle splits — transparently — into two halves:

1. **A shippable, *measurable* operator-experience core (Track E).** This is where
   the real, here-and-now value is, and all of it runs on the reference box.
2. **An honestly-gated experimental/research track (A/B/C)**, explicitly requested.
   Each item is attempted under the standing gate (`product == N` / bit-exact /
   Intel SDE) and **documented even when the result is a wash or a HW-gated
   design**. Honest negatives are a *valid deliverable*, not a failure.

The cycle covers **both integer factorization and discrete logarithm (DLP)** work.

## Track map

The fork's letter tracks group by theme across revisions: **A** = number-field /
cofactor math, **B** = SIMD, **C** = GPU, **D** = multi-GPU/HPC, **E** = UX /
orchestration. v3.3.0 continues each.

| Code | Item | Class | Honest payoff on this hardware | Doc |
|------|------|-------|--------------------------------|-----|
| E4 | Live TUI dashboard + ETA + per-phase + GPU/CPU util | Usability | Real, here-and-now (UX, not speed) | [usability-v330](usability-v330.md) |
| E5 | `--doctor` preflight (env + resource feasibility) | Usability | Real, here-and-now | [usability-v330](usability-v330.md) |
| E6 | Shell completions (bash/zsh/fish) + man pages | Usability | Real, here-and-now | [usability-v330](usability-v330.md) |
| E7 | Checkpoint/resume robustness + clarity | Reliability | Real (clarity now; deeper ckpt deferred) | [usability-v330](usability-v330.md) |
| E8 | Slurm/PBS integration + auto-gen job script | Systems | Real capability unlock | [usability-v330](usability-v330.md) |
| A5 | Galois automorphism auto-detect + recommend | Algorithm | **Real & measurable** (modest matrix/sieve shrink) | [galois-auto-a5](galois-auto-a5.md) |
| A6 | exTNFS / Tower-NFS feasibility skeleton | Research | Research-grade (DLP); documented, gated | [extnfs-a4](extnfs-a4.md) |
| B4 | **AVX2** batched modular inverse (real silicon) | SIMD | **First measured SIMD number on this box** | [avx2-simd-b4](avx2-simd-b4.md) |
| B5 | IFMA GF(p) → `arith-modp` wiring (DLP linalg) | SIMD | HW-gated (AVX-512-IFMA); DLP-only | [ifma-gfp-b3](ifma-gfp-b3.md) |
| C5 | GPU polyselect **stage-2** root-sieve | GPU | Real at large N; Amdahl-limited at testable sizes | [gpu-polyselect-ropt-c5](gpu-polyselect-ropt-c5.md) |
| C6 | GPU GF(p) **lingen NTT** (BWC linear-generator) | GPU | Tiny single-machine net; multi-GPU/DLP play | [gpu-lingen-ntt-c6](gpu-lingen-ntt-c6.md) |

## Sequencing

1. **E6** (completions + man), **E5** (`--doctor`) — immediate UX, runs here.
2. **E4** (dashboard + ETA), **E8** (scheduler) — high UX / capability unlock.
3. **A5** (Galois auto-detect) — the measurable modest algorithmic win.
4. **B4** (AVX2 modinv) — first measured SIMD on this box (may be a wash — measure).
5. **E7** (checkpoint/resume) — reliability + clarity.
6. **C5** (GPU root-sieve) — real at >c100; wash at testable sizes.
7. **B5**, **C6**, **A6** — DLP/cluster + research; HW-gated, documented honestly.

## Gates (unchanged fork ethos)

- No changes to the core NFS algorithms or their parameters.
- After every change: full `make check` + seeded c60–c100 factorizations verified
  `product == N`; GPU/SIMD kernels re-verified bit-exact vs the CPU path (under
  Intel SDE for any AVX-512 piece).
- Measured results reported honestly — **including negatives and HW-gated
  designs** — in `docs/` and `BENCHMARKS.md`.
- Do not bulk-reformat upstream C/C++. Commit / push / release only when asked.

## Net target

The headline *shippable, measured* value is a markedly better operator experience
(Track E) plus the first **real AVX2 measured SIMD numbers** (B4) and a genuine,
measurable **Galois** algorithmic win (A5). The GPU/DLP research (C5, C6, B5, A6)
is attempted under the validation gate and reported honestly. **No dishonest
single-machine "speed win" is promised.**
