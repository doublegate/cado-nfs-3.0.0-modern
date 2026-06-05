/* cxx_mpz <-> uint64 survivor-batch bridge for the GPU ECM backend; see
 * gpu_cofac.hpp. Normal C++ TU (uses GMP/cxx_mpz); calls the validated
 * gpu_ecm::factor_batch (gpu_ecm.cu / gpu_ecm_stub.cpp). */

#include "gpu_cofac.hpp"
#include "gpu_ecm.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <gmp.h>
#include "gmp_auxx.hpp"   /* mpz_set_uint64 / mpz_get_uint64 */

namespace gpu_ecm {

/* Cofactors are routed by size: odd values < 2^61 go to the single-word
 * factor_batch; odd values in [2^61, 2^125) go to the 2-limb factor_batch_128
 * (mfb > 62, e.g. c175's mfb1=90). Even values, <= 2, and >= 2^125 are left to
 * the CPU path (result stays 1). */
std::vector<cxx_mpz> cofac_batch(std::vector<cxx_mpz> const & cofactors,
                                 int ncurves, unsigned long B1, unsigned long B2)
{
    size_t const M = cofactors.size();
    std::vector<cxx_mpz> result(M);
    for (auto & r : result) mpz_set_ui(r.x, 1);        /* default: no factor */

    if (M == 0 || !available()) return result;

    /* gather, remembering each cofactor's position, split by word width */
    std::vector<uint64_t> mod64;   std::vector<size_t> idx64;
    std::vector<uint64_t> lo128, hi128; std::vector<size_t> idx128;
    for (size_t i = 0; i < M; i++) {
        mpz_srcptr c = cofactors[i].x;
        if (mpz_cmp_ui(c, 2) <= 0 || mpz_even_p(c)) continue;   /* >2 and odd */
        size_t const bits = mpz_sizeinbase(c, 2);
        if (bits <= 61) {
            mod64.push_back(mpz_get_uint64(c)); idx64.push_back(i);
        } else if (bits <= 125) {
            cxx_mpz hi, lo;
            mpz_fdiv_q_2exp(hi.x, c, 64);                       /* c >> 64 */
            mpz_fdiv_r_2exp(lo.x, c, 64);                       /* c & (2^64-1) */
            lo128.push_back(mpz_get_uint64(lo.x));
            hi128.push_back(mpz_get_uint64(hi.x));
            idx128.push_back(i);
        }
    }

    /* one GPU launch per width over the whole batch */
    if (!mod64.empty()) {
        std::vector<uint64_t> fac;
        factor_batch(mod64, ncurves, B1, B2, fac);
        for (size_t k = 0; k < idx64.size(); k++)
            if (fac[k] > 1) mpz_set_uint64(result[idx64[k]].x, fac[k]);
    }
    if (!lo128.empty()) {
        std::vector<uint64_t> flo, fhi;
        factor_batch_128(lo128, hi128, ncurves, B1, B2, flo, fhi);
        for (size_t k = 0; k < idx128.size(); k++) {
            if (flo[k] == 0 && fhi[k] == 0) continue;           /* no factor */
            cxx_mpz f, lo;
            mpz_set_uint64(f.x, fhi[k]);
            mpz_mul_2exp(f.x, f.x, 64);
            mpz_set_uint64(lo.x, flo[k]);
            mpz_add(f.x, f.x, lo.x);
            if (mpz_cmp_ui(f.x, 1) > 0) mpz_set(result[idx128[k]].x, f.x);
        }
    }

    return result;
}

void cofac_batch_full(std::vector<cxx_mpz> const & cofactors,
                      int ncurves, unsigned long B1, unsigned long B2,
                      std::vector<std::vector<cxx_mpz>> & primes,
                      std::vector<cxx_mpz> & leftover,
                      int max_rounds)
{
    size_t const M = cofactors.size();
    primes.assign(M, {});
    leftover.resize(M);
    for (size_t i = 0; i < M; i++) mpz_set(leftover[i].x, cofactors[i].x);
    if (M == 0 || !available()) return;

    /* per-cofactor list of still-composite parts to keep splitting; leftover[]
     * accumulates parts the GPU could not factor. Invariant throughout:
     * product(primes[i]) * product(pending[i]) * leftover[i] == cofactors[i]. */
    std::vector<std::vector<cxx_mpz>> pending(M);
    for (size_t i = 0; i < M; i++) {
        mpz_srcptr c = cofactors[i].x;
        if (mpz_cmp_ui(c, 1) <= 0) continue;                 /* leftover stays c (==1) */
        if (mpz_probab_prime_p(c, 25)) {                     /* already prime */
            primes[i].push_back(cofactors[i]); mpz_set_ui(leftover[i].x, 1);
        } else {
            pending[i].push_back(cofactors[i]); mpz_set_ui(leftover[i].x, 1);
        }
    }

    for (int round = 0; round < max_rounds; round++) {
        /* gather every still-composite part into one flat GPU batch */
        std::vector<cxx_mpz> flat; std::vector<size_t> ref;
        for (size_t i = 0; i < M; i++)
            for (auto const & p : pending[i]) { flat.push_back(p); ref.push_back(i); }
        if (flat.empty()) break;

        std::vector<cxx_mpz> const fac = cofac_batch(flat, ncurves, B1, B2);

        std::vector<std::vector<cxx_mpz>> next(M);
        bool progress = false;
        for (size_t k = 0; k < flat.size(); k++) {
            size_t const i = ref[k];
            mpz_srcptr p = flat[k].x;
            mpz_srcptr f = fac[k].x;
            if (mpz_cmp_ui(f, 1) > 0 && mpz_cmp(f, p) < 0 && mpz_divisible_p(p, f)) {
                progress = true;
                cxx_mpz g; mpz_divexact(g.x, p, f);
                /* classify the factor and the cofactor: prime -> done, else recurse */
                if (mpz_probab_prime_p(f, 25)) primes[i].push_back(fac[k]);
                else                           next[i].push_back(fac[k]);
                if (mpz_cmp_ui(g.x, 1) != 0) {
                    if (mpz_probab_prime_p(g.x, 25)) primes[i].push_back(g);
                    else                             next[i].push_back(g);
                }
            } else {
                /* GPU stuck on this composite -> leave it for the CPU path */
                mpz_mul(leftover[i].x, leftover[i].x, p);
            }
        }
        pending.swap(next);
        if (!progress) break;
    }

    /* anything still pending at the round limit goes to the CPU leftover */
    for (size_t i = 0; i < M; i++)
        for (auto const & p : pending[i])
            mpz_mul(leftover[i].x, leftover[i].x, p.x);

    /* optional invariant self-check: product(primes)*leftover == cofactor */
    if (getenv("CADO_GPU_DEBUG")) {
        long bad = 0, full = 0;
        for (size_t i = 0; i < M; i++) {
            cxx_mpz prod; mpz_set_ui(prod.x, 1);
            for (auto const & p : primes[i]) mpz_mul(prod.x, prod.x, p.x);
            mpz_mul(prod.x, prod.x, leftover[i].x);
            if (mpz_cmp(prod.x, cofactors[i].x) != 0) bad++;
            if (mpz_cmp_ui(leftover[i].x, 1) == 0) full++;
        }
        fprintf(stderr, "# cofac_batch_full: M=%zu full=%ld INVARIANT-BAD=%ld\n",
                M, full, bad);
    }
}

} // namespace gpu_ecm
