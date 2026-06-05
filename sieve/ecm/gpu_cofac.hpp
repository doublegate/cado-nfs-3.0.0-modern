#ifndef CADO_GPU_COFAC_HPP
#define CADO_GPU_COFAC_HPP

/* cxx_mpz <-> uint64 bridge for the GPU ECM batch (gpu_ecm.hpp). This is the
 * survivor-batch flush: las accumulates survivors in las.survivors.L (see
 * sieve/las.cpp), each carrying word-sized cofactors (cofac_candidate.cofactor,
 * cxx_mpz). When enough have queued, the drain hands the whole batch here; we
 * extract the eligible single-word cofactors, run ONE GPU ECM launch over all
 * of them, and hand back a found factor per cofactor for the per-survivor
 * cofactorization to consume as a hint.
 *
 * Compiled by the normal C++ compiler (it touches GMP / cxx_mpz, which nvcc
 * cannot), and calls the plain-uint64 gpu_ecm::factor_batch in gpu_ecm.cu.
 */

#include <vector>
#include "cxx_mpz.hpp"

namespace gpu_ecm {

/* For each input cofactor, return a non-trivial factor found by GPU ECM, or 1
 * if none was found or the cofactor is ineligible (not a single odd word in
 * [3, 2^62), e.g. already prime / too large -> left to the CPU path). All
 * eligible cofactors are processed in a single batched GPU launch.
 *
 * `out.size() == cofactors.size()`. A return of `available()==false` yields all
 * ones (i.e. defer everything to the CPU path).
 */
std::vector<cxx_mpz> cofac_batch(std::vector<cxx_mpz> const & cofactors,
                                 int ncurves,
                                 unsigned long B1,
                                 unsigned long B2);

/* Fully factor each cofactor on the GPU as far as ECM can, via up to
 * `max_rounds` batched passes (each pass splits the still-composite parts). On
 * return, primes[i] holds the prime factors the GPU extracted (each verified
 * prime) and leftover[i] is the remaining unfactored part (1 if complete), with
 * the invariant product(primes[i]) * leftover[i] == cofactors[i]. This lets the
 * caller skip facul entirely when leftover[i] == 1 (smooth iff every prime is
 * <= 2^lpb), and reject early when a prime exceeds 2^lpb -- the win in the
 * heavy-cofactoring (large-mfb) regime. primes[i] empty / leftover[i] ==
 * cofactors[i] means the GPU found nothing (or the cofactor was ineligible). */
void cofac_batch_full(std::vector<cxx_mpz> const & cofactors,
                      int ncurves,
                      unsigned long B1,
                      unsigned long B2,
                      std::vector<std::vector<cxx_mpz>> & primes,
                      std::vector<cxx_mpz> & leftover,
                      int max_rounds = 4);

} // namespace gpu_ecm

#endif /* CADO_GPU_COFAC_HPP */
