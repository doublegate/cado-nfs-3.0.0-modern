# GPU ECM pre-factoring front-end (v3.1.0-modern, Track 2.1)

Strip small/medium prime factors from a large `N` on the GPU with batched ECM,
**before** handing the reduced cofactor to NFS. CADO-NFS already advises this by
hand (the README: *"strip small prime factors … before using it"*); this does it
on the GPU.

## Why this is a real win (unlike in-sieve GPU cofactorization)

`docs/gpu-cofactorization.md` shows that offloading the siever's *cofactorization*
to the GPU is a no-win: it is only ~8 % of siever time (Amdahl-bound), and `facul`
already finds every valid relation. **Pre-factoring is a different, separate
stage**, so the GPU's large ECM throughput is pure upside whenever `N` happens to
have a findable factor — there is no Amdahl ceiling to fight.

## The hard part: ECM *modulo a multi-hundred-bit N*

ECM finds a factor `p` of `N` by working **modulo N** (the big number), not modulo
`p`. The existing GPU ECM (`sieve/ecm/gpu_ecm.cu`) only handles moduli `< 2^126`
— fine for sieve cofactors, useless here. So this tool uses a **multi-precision
(K-limb) Montgomery ECM** that generalizes the bit-exact-validated 2-limb CIOS
montmul (`bench/gpu-mont128.cu`) and the Montgomery-curve XZ ladder
(`bench/gpu-ecm.cu`) to `K` 64-bit limbs. Widths `K ∈ {2,4,8,16}` cover `N` up to
**1022 bits (~307 digits)**; the width is chosen automatically from `N`.

The device math is **validated bit-exact** (`bench/gpu-ecm-mp.cu`): `montmul`
0/20000 wrong vs an independent binary-mulmod reference, and ECM 0/512 GPU lanes
differing from CPU, for 128/256/512-bit. GMP handles `N` parsing, the per-modulus
Montgomery setup (`n^{-1} mod 2^64`, `R mod n`, `R^2 mod n`) and `gcd(Z, n)`.

## Build & use

Via the project (built only with `-DENABLE_GPU=ON`):

```bash
# in local.sh: CMAKE_EXTRA_ARGS="... -DENABLE_GPU=ON"   (optionally -DCADO_GPU_ARCH=86)
make cmake && (cd build/$(hostname) && make gpu-prefactor)
build/$(hostname)/misc/gpu-prefactor <N> [B1=50000] [curves=4096] [B2=100*B1]
```

Or standalone:

```bash
nvcc -arch=sm_86 -O3 misc/gpu_prefactor/gpu-prefactor.cu -lgmp -o gpu-prefactor
./gpu-prefactor <N> [B1=50000] [curves=4096] [B2=100*B1]
```

Stage-2 (BSGS) + Suyama-σ curves are on by default; each run self-checks a
32-lane subset GPU-vs-CPU (`# selfcheck: PASS`). The curve batch is split across
all visible GPUs (multi-GPU; one launch on a single-GPU box).

Example (a 12-digit factor inside a 103-digit `N`):

```
$ ./gpu-prefactor 7531562789977494832502692671807766352173951462749583994637719958325130456544186728842558728590304445
13  50000 16384
# GPU ECM pre-factor: 339-bit N (~103 digits), width K=8 (512-bit), B1=50000, curves=16384, 5133 multipliers
factors stripped: 889476218401
remaining cofactor: 846741333…725265224027430410085313  (prime)
```

Exit code `0` if at least one factor was stripped, `1` if none was found (try a
larger `B1` / more curves), `2` on a usage/size error.

## Status & next increments

- **Done & validated:** the multi-precision GPU ECM math (bit-exact); stage-1 +
  stage-2 BSGS + Suyama-σ curves (per-run GPU-vs-CPU self-check); multi-GPU
  batching; and the CMake target (`-DENABLE_GPU=ON`, device `-O3`, `sm_86`).
- **Next (Track 2.1):** a `cado-nfs.py --gpu-prefactor` pre-stage (run this,
  substitute the reduced cofactor, recombine the stripped factors at the end);
  and a benchmark vs CPU GMP-ECM across factor sizes.

Reach today: ~15-digit factors at `B1=50000`; raising `B1`/`B2`/`curves` extends
it (the usual ECM trade-off). Larger factors want the standard staged-`B1`
schedule, a follow-up.
