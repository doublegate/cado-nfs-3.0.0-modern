/* Non-CUDA fallback for the GPU ECM backend (gpu_ecm.hpp). Compiled in place of
 * gpu_ecm.cu when the build has no CUDA (ENABLE_GPU off or no nvcc), so that
 * facul.cpp / the batch cofactorization layer can call gpu_ecm::available()
 * unconditionally and simply get `false` (-> CPU path). */
#include "gpu_ecm.hpp"

namespace gpu_ecm {

bool available() { return false; }

void factor_batch(std::vector<uint64_t> const & moduli, int /*ncurves*/,
                  unsigned long /*B1*/, unsigned long /*B2*/,
                  std::vector<uint64_t> & factor)
{
    factor.assign(moduli.size(), 0);   /* no factors found; defer to CPU */
}

void factor_batch_128(std::vector<uint64_t> const & mod_lo,
                      std::vector<uint64_t> const & /*mod_hi*/,
                      int /*ncurves*/, unsigned long /*B1*/, unsigned long /*B2*/,
                      std::vector<uint64_t> & fac_lo,
                      std::vector<uint64_t> & fac_hi)
{
    fac_lo.assign(mod_lo.size(), 0);   /* no factors found; defer to CPU */
    fac_hi.assign(mod_lo.size(), 0);
}

} // namespace gpu_ecm
