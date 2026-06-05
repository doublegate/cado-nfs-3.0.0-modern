# GPU ECM cofactorization (Phase 3)

CADO-NFS factors the small "cofactors" of sieve survivors with ECM
(`sieve/ecm/`, entered via `facul_all(std::vector<cxx_mpz>, strategies)` — a
**batch** of small moduli, each run through ~11–14 ECM curves). ECM is
**modular-arithmetic-bound** on small (≤ mfb ≈ 52–96 bit) moduli, and the batch
is thousands of independent curves — the ideal GPU workload. Offloading it frees
CPU cores to sieve (Bos & Kleinjung; eprint 2014/397).

## Measured on this box (RTX 3090, sm_86, vs i9-10850K)

`bench/gpu-modmul-bench.cu` — batched 64-bit Montgomery modmul (ECM's inner
primitive), 1 M independent moduli, **correctness PASS vs CPU**:

| | throughput |
|---|---|
| RTX 3090 | **279 G modmul/s** |
| CPU, 1 core | 0.36 G/s |
| CPU, 20 cores (est.) | 7.2 G/s |
| **GPU ÷ 20-core CPU** | **~39×** |

So the GPU does ECM's core op ~39× faster than the whole CPU.

## What that does and doesn't mean (honest Amdahl analysis)

- The **cofactorization phase** can be ~39× cheaper — but in the c120 siever
  profile, cofactorization (`factor`) is only **~8 %** of `las` CPU time
  (`sieving` dominates at ~70 %). So *naively* offloading the current
  cofactorization frees ~8 % of CPU for more sieving — a real but modest
  end-to-end gain.
- **The bigger win is a parameter-regime shift.** CADO uses modest `ncurves`
  (11–14) and `mfb` partly *because* cofactorization is expensive on CPU. With a
  ~39×-cheaper GPU cofactorization you can afford **many more curves / larger
  `mfb` / an extra large prime**, which raises the relations-per-special-q yield
  and rebalances sieve-vs-cofactor — potentially a larger net throughput gain,
  and the main reason GPU cofactorization matters in practice (and more so for
  DLP, where cofactorization is a larger fraction).

## Working GPU ECM stage-1 (`bench/gpu-ecm.cu`)

A complete batched ECM stage-1 — Montgomery-curve XZ ladder for `[∏ p^e]P`
(prime powers ≤ B1), on-device binary gcd — runs one curve per thread. The same
`ecm_stage1` compiles for host and device, so the GPU is checked bit-exact:

```
validation   : PASS (0/2048 GPU lanes differ from CPU)
factors found: 256/256 composites cracked by >=1 curve   (B1=2000, 64-bit n)
throughput   : ~2.75M curves/s on the RTX 3090
```

At CADO's ~13 curves/cofactor that is ~210 k cofactors/s. This is the core of the
backend; the remaining work is integration + refinement.

## Integration roadmap (remaining)

1. ~~GPU ECM stage-1 kernel~~ ✅ done (`bench/gpu-ecm.cu`): validated bit-exact,
   finds factors, benchmarked. Refinements:
   - ✅ **Suyama-σ parametrization** (`bench/gpu-ecm-suyama.cu`) — curves with
     guaranteed torsion 12; device modular inverse (with lucky-factor-in-setup);
     ~1.08× more composites cracked at a fixed low curve budget vs the POC curve.
   - ✅ **128-bit (2-limb) Montgomery modmul** (`bench/gpu-mont128.cu`) — CIOS
     REDC, validated bit-exact on CPU and GPU over 200 000 trials; the core for
     larger-`mfb` cofactors. Full 128-bit ECM = the validated 64-bit ladder/curve
     structure over this modmul.
   - ✅ **stage 2 (BSGS continuation)** (`bench/gpu-ecm-stage2.cu`) — baby steps
     `[r]Q`, giant steps `[mW]Q`, accumulated cross-differences with one final
     gcd; validated bit-exact GPU vs CPU, and finds **+64% more factors per
     curve** (1011 → 1663 / 2048 composites at B1=2000, B2=50000).
2. ✅ **Library module** `sieve/ecm/gpu_ecm.{hpp,cu}` — packages the validated
   stage-1 + stage-2 kernel behind a plain-C++ batch API
   (`gpu_ecm::factor_batch(moduli, ncurves, B1, B2, factor)` +
   `gpu_ecm::available()`). Standalone-tested: **1000/1000 moduli cracked, every
   returned factor divides its modulus.** `gpu_ecm_stub.cpp` is the non-CUDA
   fallback so the CPU path is unaffected when CUDA is off.
3. ✅ **CMake CUDA build** — `config/cuda.cmake` (`option(ENABLE_GPU)`,
   `check_language(CUDA)`, `CUDAToolkit`, default `sm_86`) wired into the `facul`
   library (`sieve/ecm/CMakeLists.txt`). **Configures cleanly with
   `-DENABLE_GPU=ON`** (CUDA 13.3 detected; pass `-DCMAKE_CUDA_ARCHITECTURES=86`
   for the 3090). CPU build is unchanged when the flag is off.
4. **Survivor-batch hook.** Architectural finding: `facul_all(N, …)` receives the
   **2 cofactors of one relation**, so the GPU must hook one level up, where
   survivors already accumulate. CADO 3.0.0 collects all survivors in
   **`las.survivors.L`** (a `list<(special_q, list<cofac_candidate>)>`, each
   `cofac_candidate` = `{a, b, vector<cxx_mpz> cofactor}`), drained for
   cofactoring at `sieve/las.cpp:~997` (batch mode).
   - ✅ **Bridge done** — `sieve/ecm/gpu_cofac.{hpp,cpp}`: a `cxx_mpz`↔`uint64`
     survivor-batch flush (`gpu_ecm::cofac_batch(cofactors, ncurves, B1, B2)`)
     that gathers the eligible single-word cofactors, runs ONE GPU launch, and
     returns a found factor per cofactor (1 = none → CPU path). Plain C++ (GMP),
     calls the validated `gpu_ecm::factor_batch`; **syntax-checks clean against
     CADO headers**; built into the `facul` library.
   - ⏳ **Drain wiring** — the deferred-cofactoring flow: survivors are created
     in `sieve/las-process-bucket-region.cpp:~567`, transferred to
     `ws.cofac_candidates` (`:~779`), then dispatched as per-survivor
     `detached_cofac` tasks (`sieve/las-detached-cofac.cpp`). A correct, batched
     hook accumulates the queued `cofac_candidates`, calls `cofac_batch` once,
     pre-divides each cofactor by its GPU factor, then lets the existing
     per-survivor cofactoring finish the (smaller) remainder — so relation
     emission is byte-for-byte unchanged. This restructures a multithreaded hot
     path and MUST be validated relations-identical in a live run; it is the
     remaining focused follow-on (a per-survivor hook would be a regression and
     is not worth wiring).
5. ✅ **Full in-CADO CUDA build works.** With `-DENABLE_GPU=ON`
   `-DCMAKE_CUDA_ARCHITECTURES=86`, `gpu_ecm.cu` compiles under `nvcc` inside
   CADO's C++20/`-Werror` build (no flag conflicts), `gpu_cofac.cpp` compiles,
   and `las` links `libcudart` with the `gpu_ecm::{available,factor_batch,
   cofac_batch}` symbols present. The CUDA-enabled `las` runs correctly
   (microbench unchanged at ~11.7 s). ⏳ Remaining: the drain wiring (item 4) +
   live-factorization validation + `ncurves`/`mfb`/B2 retuning.

## Status

- ✅ CUDA 13.3 + RTX 3090 verified end-to-end; CADO had no prior GPU code.
- ✅ **Full GPU ECM engine built & validated**: stage-1, Suyama curves, 128-bit
  modmul, stage-2 BSGS — all bit-exact vs CPU; ~39× modmul throughput vs the
  20-core CPU.
- ✅ **Packaged as `sieve/ecm/gpu_ecm` + CMake `ENABLE_GPU` build** that
  configures cleanly; standalone batch test passes (1000/1000).
- ⏳ Remaining: hook `factor_batch` into the survivor-batch layer (item 4),
  full in-CADO CUDA build, and in-factorization validation/retuning — all fully
  testable on this hardware.
