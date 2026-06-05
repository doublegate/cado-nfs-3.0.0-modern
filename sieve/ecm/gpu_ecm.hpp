#ifndef CADO_GPU_ECM_HPP
#define CADO_GPU_ECM_HPP

/* Batched ECM cofactorization on a CUDA GPU, for use as an optional backend of
 * facul_all() (see sieve/ecm/facul.cpp). factor_batch handles the common small-
 * cofactor case (odd modulus < 2^62, one machine word); factor_batch_128 handles
 * odd moduli < 2^126 (two words) for the larger cofactors that arise when
 * mfb > 62. Validated kernels: bench/gpu-ecm*.cu (stage 1 + stage-2 BSGS) and
 * bench/gpu-mont128.cu (2-limb montmul), bit-exact vs CPU. See
 * docs/gpu-cofactorization.md.
 *
 * This header is plain C++ (no CUDA types) so facul.cpp can include it whether
 * or not the build has CUDA; gpu_ecm::available() returns false when built
 * without CUDA or when no device is present.
 */

#include <cstdint>
#include <vector>

namespace gpu_ecm {

/* True iff this binary was built with CUDA and a usable device is present. */
bool available();

/* For each modulus, try `ncurves` ECM curves (stage 1 to B1, stage-2 BSGS to
 * B2). On return, factor[i] is a nontrivial factor of moduli[i], or 0 if none
 * was found / the modulus was skipped. A modulus is processed only if it is
 * odd and < 2^62; otherwise factor[i] = 0 (leave it to the CPU path).
 *
 * `moduli` and `factor` have the same length; `factor` is resized by the call.
 */
void factor_batch(std::vector<uint64_t> const & moduli,
                  int ncurves,
                  unsigned long B1,
                  unsigned long B2,
                  std::vector<uint64_t> & factor);

/* 128-bit variant for cofactors that overflow one word (odd, < 2^126; used when
 * mfb > 62, e.g. c175's mfb1=90). Moduli and factors are passed as lo/hi 64-bit
 * limb pairs to keep this header free of compiler-specific 128-bit types.
 * fac_lo[i]/fac_hi[i] is a nontrivial factor of moduli[i], or 0 if none. The
 * device math is the bit-exact-validated 2-limb CIOS montmul (bench/gpu-mont128).
 */
void factor_batch_128(std::vector<uint64_t> const & mod_lo,
                      std::vector<uint64_t> const & mod_hi,
                      int ncurves,
                      unsigned long B1,
                      unsigned long B2,
                      std::vector<uint64_t> & fac_lo,
                      std::vector<uint64_t> & fac_hi);

} // namespace gpu_ecm

#endif /* CADO_GPU_ECM_HPP */
