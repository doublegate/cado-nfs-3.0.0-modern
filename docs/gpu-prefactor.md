# GPU ECM pre-factoring front-end (v3.1.0-modern, Track 2.1)

Strip small/medium prime factors from a large `N` on the GPU with batched ECM,
**before** handing the reduced cofactor to NFS. CADO-NFS already advises this by
hand — the README says to *"strip small prime factors (trial division / P-1 /
P+1 / ECM) before using it"* — so this is a stage users are expected to run
anyway; the contribution here is doing it on the GPU, where ECM's thousands of
independent curves are the ideal workload.

The implementation lives in `misc/gpu_prefactor/` (`gpu-prefactor.cu`, device math
in `gpu_ecm_mp.cuh`); the orchestration hook is `scripts/cadofactor/gpu_prefactor.py`
wired to `cado-nfs.py --gpu-prefactor`.

## Why this is a real win (unlike in-sieve GPU cofactorization)

`docs/gpu-cofactorization.md` records the honest negative result of Phase 3:
offloading the *siever's* cofactorization to the GPU is a no-win, because that
step is only ~8% of siever time (Amdahl-bound) and the CPU `facul` already finds
every valid relation — a 39× kernel speedup on 8% of the work cannot move the
single-machine wall.

**Pre-factoring is a different, separate stage.** It runs once, up front, on the
whole `N` — not inside the siever's inner loop — so there is no Amdahl ceiling to
fight. Whenever `N` happens to have a findable factor, the GPU's large ECM
throughput is pure upside: a factor stripped here shrinks the number NFS has to
factor (or removes the need for NFS entirely), and a factor *not* found costs
only the ECM time, exactly as the manual ECM step would.

## The hard part: ECM *modulo a multi-hundred-bit N*

ECM finds a factor `p` of `N` by working **modulo N** (the big number), not modulo
`p` — you do not know `p` yet. The existing GPU ECM (`sieve/ecm/gpu_ecm.cu`) only
handles moduli `< 2^126`, which is fine for sieve cofactors but useless for a
multi-hundred-digit `N`. So this tool carries a **multi-precision (K-limb)
Montgomery ECM** that generalizes the bit-exact-validated 2-limb CIOS `montmul`
(`bench/gpu-mont128.cu`) and the Montgomery-curve XZ ladder (`bench/gpu-ecm.cu`)
to `K` 64-bit limbs. Widths `K ∈ {2,4,8,16}` cover `N` up to **1022 bits (~307
digits)**; the width is chosen automatically from `N`'s size.

GMP handles the host-side pieces: parsing `N`, the per-modulus Montgomery setup
(`n^{-1} mod 2^64`, `R mod n`, `R^2 mod n`), the Suyama-σ curve construction
(`σ → (X0, Z0, a24)` mod N, which can itself drop a free factor if a denominator
is non-invertible), and the final `gcd(Z, N)` that turns a stage output into a
factor. Stage-2 is the standard baby-step/giant-step (BSGS) continuation over the
primes in `(B1, B2]`.

## Correctness — validated bit-exact, and `product == N`

The fork's gate ("every GPU result re-verified against the CPU path") is enforced
two ways:

- **Device math, bit-exact vs CPU** (`bench/gpu-ecm-mp.cu`): `montmul` 0/20000
  wrong vs an independent binary-mulmod reference, and ECM 0/512 GPU lanes
  differing from the CPU `ecm_run2` for 128/256/512-bit moduli. The *same*
  `__host__ __device__ ecm_run2` runs on both sides, so it is the identical
  algorithm, not a comparison against a different ECM.
- **Per-run self-check**: every pass re-runs a 32-lane subset on the CPU and
  compares `z1`/`g2`; the tool prints `# selfcheck: PASS` (and aborts the pass on
  any mismatch). Stage-2 BSGS is a new composition of the validated primitives, so
  this guards it on every invocation — cheap, and in keeping with the fork ethos.
- **End-to-end `product == N`** (this release): a 90-digit `N` with several
  factors ≤ 30 digits was run through `staged` mode; every stripped factor divides
  `N`, the reported cofactor matches the recomputed quotient, and
  `(stripped product) × cofactor == N` exactly. Self-check PASS throughout.

Any output `gcd` equal to `1` or `N` is discarded (no factor), so a degenerate
curve can never produce a false factor.

## Build & use

Via the project (built only with `-DENABLE_GPU=ON`):

```bash
# in local.sh:  CMAKE_EXTRA_ARGS="... -DENABLE_GPU=ON"   (optionally -DCADO_GPU_ARCH=86)
make cmake && (cd build/$(hostname) && make gpu-prefactor)
build/$(hostname)/misc/gpu-prefactor <N> [B1=50000] [curves=4096] [B2=100*B1]
```

Or standalone:

```bash
nvcc -arch=sm_86 -O3 misc/gpu_prefactor/gpu-prefactor.cu -lgmp -o gpu-prefactor
./gpu-prefactor <N> [B1=50000] [curves=4096] [B2=100*B1]   # single stage
./gpu-prefactor <N> staged [maxdigits=30] [curve_scale=1.0] # escalating-B1 schedule
```

Stage-2 (BSGS) + Suyama-σ curves are on by default. The curve batch is split
across all visible GPUs (multi-GPU; a single launch on a one-GPU box). The
**staged** mode escalates `B1` (2000 → 11000 → 50000 → 250000 → 1e6 → 3e6),
finding small factors cheaply at low `B1` before spending curves at high `B1`, and
stops once the cofactor is prime or 1. Exit code `0` if at least one factor was
stripped, `1` if none was found (try a larger `B1`/`B2`/more curves), `2` on a
usage/size error.

### Integrated use (`cado-nfs.py`)

```bash
cado-nfs.py <N> --gpu-prefactor [--gpu-b1 50000] [--gpu-b2 5000000] [--gpu-curves 8192]
```

Runs this stage before NFS. If it fully factors `N` (cofactor 1 or prime), it
prints the factorization and **skips NFS**; if a composite cofactor remains, it
continues with a fresh `cado-nfs.py` on the cofactor; if nothing is stripped (or
the binary isn't built), it falls through to a normal NFS run with no change in
behaviour.

## Measured throughput (RTX 3090 vs the full i9-10850K)

The CPU baseline is the **same** `ecm_run2` (stage-1 + stage-2 BSGS) run across
all 20 hardware threads — an apples-to-apples comparison (identical algorithm and
code), not a different CPU ECM. At `B1 = 50000`, `B2 = 5e6` (a representative
pre-factoring stage), `bench/gpu-prefactor-bench.cu`:

| Modulus width | GPU (curves/s) | CPU, 20 threads (curves/s) | GPU speedup |
|---|---:|---:|---:|
| 128-bit (≤ ~38-digit N) | ~16 500 | ~340 | **~49×** |
| 256-bit (≤ ~77-digit N) | ~3 100 | ~120 | **~26×** |
| 512-bit (≤ ~154-digit N) | ~320 | ~26 | **~12×** |

The GPU edge shrinks at wider moduli (more registers/limbs per lane, lower
occupancy), but stays a solid order-of-magnitude even at 512 bits. After partial
stripping the surviving cofactor is smaller, so later stages run at the wider
widths' rates. These numbers reproduce within run-to-run noise; full methodology
and the build lines are in `BENCHMARKS.md`.

## Status & next increments

- **Done & validated:** the multi-precision GPU ECM math (bit-exact, K ∈
  {2,4,8,16}); stage-1 + stage-2 BSGS + Suyama-σ curves (per-run GPU-vs-CPU
  self-check); multi-GPU curve batching; the CMake target (`-DENABLE_GPU=ON`,
  device `-O3`, `sm_86`); the `cado-nfs.py --gpu-prefactor` integration
  (fast-path skip + cofactor continuation); the escalating-`B1` staged schedule;
  and the CPU-vs-GPU benchmark. End-to-end `product == N` confirmed.
- **Next (Track 2.1, optional):** per-cofactor adaptive curve counts; exposing the
  staged schedule directly through `cado-nfs.py --gpu-prefactor`; and (Track 2.3)
  distributing the curve batch across cluster GPUs for a DLP-scale front-end.

Reach today: ~20–30-digit factors via `staged`; raising `B1`/`B2`/`curves`
extends it, at the usual ECM time/probability trade-off.
