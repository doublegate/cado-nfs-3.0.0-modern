#include "cado.h" // IWYU pragma: keep

#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include <algorithm>
#include <atomic>
#include <vector>

#include "arith/modredc_15ul.h"
#include "arith/modredc_2ul2.h"
#include "arith/modredc_ul.h"
#include "ecm/facul.hpp"
#include "ecm/facul_strategies.hpp"
#include "ecm/gpu_cofac.hpp"
#include "ecm/gpu_ecm.hpp"
#include "las-cofactor.hpp"
#include "macros.h"
#include "params.hpp"

void cofactorization_statistics::declare_usage(cxx_param_list & pl)
{
    param_list_decl_usage(pl, "stats-cofact", "write statistics about the cofactorization step in file xxx");
}

//  las_info::{init,clear,print}_cof_stats
cofactorization_statistics::cofactorization_statistics(cxx_param_list & pl)
{
    const char * statsfilename = param_list_lookup_string (pl, "stats-cofact");
    if (!statsfilename) {
        file = nullptr;
        return;
    }
    file = fopen (statsfilename, "w");
    if (file == nullptr) {
        fprintf (stderr, "Error, cannot create file %s\n", statsfilename);
        exit (EXIT_FAILURE);
    }
}

void cofactorization_statistics::call(int bits0, int bits1)
{
    if (!file) return;
    std::lock_guard<std::mutex> const dummy(lock);
    size_t s0 = cof_call.size();
    if ((size_t) bits0 >= s0) {
        size_t const news0 = std::max((size_t) bits0+1, s0 + s0/2);
        cof_call.insert(cof_call.end(), news0-s0, std::vector<uint32_t>());
        cof_success.insert(cof_success.end(), news0-s0, std::vector<uint32_t>());
        s0 = news0;
    }
    size_t s1 = cof_call[bits0].size();
    if ((size_t) bits1 >= s1) {
        size_t const news1 = std::max((size_t) bits1+1, s1 + s1/2);
        cof_call[bits0].insert(cof_call[bits0].end(), news1-s1, 0);
        cof_success[bits0].insert(cof_success[bits0].end(), news1-s1, 0);
        s1 = news1;
    }
    ASSERT_ALWAYS((size_t) bits0 < s0);
    ASSERT_ALWAYS((size_t) bits1 < s1);
    /* no need to use a mutex here: either we use one thread only
       to compute the cofactorization data and if several threads
       the order is irrelevant. The only problem that can happen
       is when two threads increase the value at the same time,
       and it is increased by 1 instead of 2, but this should
       happen rarely. */
    cof_call[bits0][bits1]++;
}

void cofactorization_statistics::print()
{
    if (!file) return;
    for(size_t bits0 = 0 ; bits0 < cof_call.size() ; ++bits0) {
        for(size_t bits1 = 0 ; bits1 < cof_call[bits0].size() ; ++bits1) {
            fprintf (file, "%zu %zu %" PRIu32 " %" PRIu32 "\n",
                    bits0, bits1,
                    cof_call[bits0][bits1],
                    cof_success[bits0][bits1]);
        }
    }
}

cofactorization_statistics::~cofactorization_statistics()
{
    if (!file) return;
    fclose (file);
}
//

/* {{{ factor_leftover_norm */

#define NMILLER_RABIN 1 /* in the worst case, what can happen is that a
                           composite number is declared as prime, thus
                           a relation might be missed, but this will not
                           affect correctness */
#define IS_PROBAB_PRIME(X) (0 != mpz_probab_prime_p((X), NMILLER_RABIN))
#define BITSIZE(X)      (mpz_sizeinbase((X), 2))
#define NFACTORS        8 /* maximal number of large primes */

/************************ cofactorization ********************************/

/* {{{ cofactoring area */

/* Return 0 if the leftover norm n cannot yield a relation.

   Possible cases, where qj represents a prime in [B,L], and rj a prime > L
   (assuming L < B^2, which might be false for the DLP descent):
   (0) n >= 2^mfb
   (a) n < L:           1 or q1
   (b) L < n < B^2:     r1 -> cannot yield a relation
   (c) B^2 < n < B*L:   r1 or q1*q2
   (d) B*L < n < L^2:   r1 or q1*q2 or q1*r2
   (e) L^2 < n < B^3:   r1 or q1*r2 or r1*r2 -> cannot yield a relation
   (f) B^3 < n < B^2*L: r1 or q1*r2 or r1*r2 or q1*q2*q3
   (g) B^2*L < n < L^3: r1 or q1*r2 or r1*r2
   (h) L^3 < n < B^4:   r1 or q1*r2, r1*r2 or q1*q2*r3 or q1*r2*r3 or r1*r2*r3
                        -> cannot yield a relation
*/
int
check_leftover_norm (cxx_mpz const & n, siever_side_config const & scs)
{
  size_t const s = mpz_sizeinbase (n, 2);
  unsigned int const lpb = scs.lpb;
  unsigned int const mfb = scs.mfb;
  unsigned int klpb;
  double nd, kB, B;

  ASSERT_ALWAYS(mpz_cmp_ui(n, 0) != 0);

  if (s > mfb)
    return 0; /* n has more than mfb bits, which is the given limit */

  if (scs.lim == 0) {
      /* special case when not sieving */
      return 1;
  }

  if (s <= lpb)
    return 1; /* case (a) */
  /* Note that in the case where L > B^2, if we're below L it's still fine of
     course, but we have no guarantee that our cofactor is prime... */

  nd = mpz_get_d (n);
  B = (double) scs.lim;
  kB = B * B;
  for (klpb = lpb; klpb < s; klpb += lpb, kB *= B)
    {
      /* invariant: klpb = k * lpb, kB = B^(k+1) */
      if (nd < kB) /* L^k < n < B^(k+1) */
	return 0;
    }

  /* Here we have L < n < 2^mfb. If n is composite and we wrongly consider
     it prime, we'll return 0, thus we'll potentially miss a relation, but
     we won't output a relation with a composite ideal, thus a base-2 strong
     prime test is enough. */

  // TODO: maybe we should pass the modulus to the facul machinery
  // instead of reconstructing it.
  int prime=0;
  if (s <= MODREDCUL_MAXBITS) {
      modulusredcul_t m;
      ASSERT(mpz_fits_ulong_p(n));
      modredcul_initmod_ul (m, mpz_get_ui(n));
      prime = modredcul_sprp2(m);
      modredcul_clearmod (m);
  } else if (s <= MODREDC15UL_MAXBITS) {
      modulusredc15ul_t m;
      unsigned long t[2];
      modintredc15ul_t nn;
      size_t written;
      mpz_export (t, &written, -1, sizeof(unsigned long), 0, 0, n);
      ASSERT_ALWAYS(written <= 2);
      modredc15ul_intset_uls (nn, t, written);
      modredc15ul_initmod_int (m, nn);
      prime = modredc15ul_sprp2(m);
      modredc15ul_clearmod (m);
  } else if (s <= MODREDC2UL2_MAXBITS) {
      modulusredc2ul2_t m;
      unsigned long t[2];
      modintredc2ul2_t nn;
      size_t written;
      mpz_export (t, &written, -1, sizeof(unsigned long), 0, 0, n);
      ASSERT_ALWAYS(written <= 2);
      modredc2ul2_intset_uls (nn, t, written);
      modredc2ul2_initmod_int (m, nn);
      prime = modredc2ul2_sprp2(m);
      modredc2ul2_clearmod (m);
  } else {
      prime = mpz_probab_prime_p (n, 1);
  }
  if (prime)
    return 0; /* n is a pseudo-prime larger than L */
  return 1;
}

/* This is the header-comment for the old factor_leftover_norm()
 * function, that is now deleted */
/* This function was contributed by Jerome Milan (and bugs were introduced
   by Paul Zimmermann :-).
   Input: n - the number to be factored (leftover norm). Must be composite!
              Assumed to have no factor < B (factor base bound).
          L - large prime bound is L=2^l
   Assumes n > 0.
   Return value:
          -1 if n has a prime factor larger than L
          1 if all prime factors of n are < L
          0 if n could not be completely factored
   Output:
          the prime factors of n are factors->data[0..factors->length-1],
          with corresponding multiplicities multis[0..factors->length-1].
*/

/* This is the same function as factor_leftover_norm() but it works
   with all norms! It is used when we want to factor these norms
   simultaneously and not one after the other.
   Return values:
   -1  one of the cofactors is not smooth
   0   unable to fully factor one of the cofactors
   1   all cofactors are smooth

  Note: for more than two sides, we may still get a relation even if not all
  cofactors were smooth. Currently, it is not taken into account by this method.
*/

/* {{{ Optional GPU ECM cofactorization hook (Phase 3, this fork).
 *
 * Default OFF. Enabled by the CADO_GPU_ECM environment variable when a CUDA
 * GPU is present (gpu_ecm::available()); it runs the validated GPU ECM batch
 * (sieve/ecm/gpu_ecm.cu, via the cxx_mpz<->uint64 bridge gpu_cofac.cpp) over
 * the leftover cofactors that facul has just processed. Two modes:
 *
 *   shadow   (CADO_GPU_ECM=1 | shadow, the default when set) -- IDENTITY
 *            PRESERVING. The GPU result is only used to verify, on real CADO
 *            cofactors, that GPU ECM finds a dividing factor; facul's verdict
 *            is never changed, so the emitted relation set is byte-for-byte
 *            identical to the CPU-only path. This is the safe validation hook.
 *
 *   salvage  (CADO_GPU_ECM=salvage) -- retries facul give-ups (FACUL_MAYBE)
 *            with GPU ECM. It NEVER overrides a definitive SMOOTH/NOT_SMOOTH
 *            verdict; it only completes a give-up when the GPU fully splits the
 *            cofactor into two primes within the large-prime bound. This can
 *            emit extra *valid* relations (a superset of the CPU-only output),
 *            never wrong ones (the product==norm invariant is preserved).
 *
 *   batch    (CADO_GPU_ECM=batch) -- the THROUGHPUT path. The GPU does not run
 *            here per call; instead cofactoring_sync collects a whole bucket
 *            region's survivors, issues ONE GPU ECM launch over all their
 *            leftover cofactors, and stores a factor hint in each
 *            cofac_standalone::gpu_hint. factor_leftover_norms then DIVIDES the
 *            hint out (when it is a prime <= lpb) so facul factors the smaller
 *            remainder -- moving the ECM work to the GPU and freeing CPU. The
 *            hinted prime is re-attached afterwards, so product==norm holds.
 *            Like salvage this yields a valid *superset* (facul may now resolve
 *            a cofactor it would have given up on), never a wrong relation.
 *
 * Because relation::compress() sorts each side's primes, and integer
 * factorization is unique, the discovery method cannot change a relation's
 * text -- only whether a give-up becomes a (valid) relation. That is why
 * shadow mode is byte-identical and salvage/batch modes are a clean superset.
 *
 * Note: the per-call shadow/salvage launch has GPU-launch latency (one tiny
 * launch per cofactoring call), so those modes are for correctness validation;
 * batch mode is the one intended to be a net speedup.
 */
namespace {
    enum class gpu_ecm_mode { off, shadow, salvage, batch };

    /* GPU ECM curve budget + targeting, all env-tunable so the throughput regime
     * can be swept without recompiling (see docs/gpu-cofactorization.md):
     *   CADO_GPU_NCURVES  curves per cofactor on the GPU      (default 16)
     *   CADO_GPU_B1 / _B2 ECM stage-1 / stage-2 bounds        (default 2000/50000)
     *   CADO_GPU_MINBITS  hard-cofactor targeting: only cofactors with at least
     *                     this many bits are sent to the GPU; smaller ones are
     *                     left to facul, which cracks them in microseconds and
     *                     would only be slowed by GPU-launch overhead (default 0
     *                     = every eligible cofactor, the blanket behaviour). */
    long env_long(const char * name, long dflt)
    {
        const char * e = getenv(name);
        if (e == nullptr || *e == '\0') return dflt;
        char * end = nullptr;
        long const v = strtol(e, &end, 10);
        return (end == e) ? dflt : v;
    }
    const int           GPU_HOOK_NCURVES = (int) env_long("CADO_GPU_NCURVES", 16);
    const unsigned long GPU_HOOK_B1      = (unsigned long) env_long("CADO_GPU_B1", 2000);
    const unsigned long GPU_HOOK_B2      = (unsigned long) env_long("CADO_GPU_B2", 50000);
    const unsigned int  GPU_HOOK_MINBITS = (unsigned int) env_long("CADO_GPU_MINBITS", 0);

    gpu_ecm_mode gpu_ecm_mode_from_env()
    {
        const char * e = getenv("CADO_GPU_ECM");
        if (e == nullptr || *e == '\0')        return gpu_ecm_mode::off;
        if (strcmp(e, "salvage") == 0)         return gpu_ecm_mode::salvage;
        if (strcmp(e, "batch") == 0)           return gpu_ecm_mode::batch;
        return gpu_ecm_mode::shadow;           /* "1", "shadow", anything else */
    }

    /* parsed once, shared by the per-call hook and the batched drain */
    const gpu_ecm_mode g_gpu_mode = gpu_ecm_mode_from_env();

    struct gpu_hook_stats {
        std::atomic<unsigned long> calls{0};   /* factor_leftover_norms calls hooked */
        std::atomic<unsigned long> split{0};   /* cofactors GPU returned a factor for */
        std::atomic<unsigned long> salvaged{0};/* FACUL_MAYBE upgraded to SMOOTH */
        std::atomic<unsigned long> full{0};    /* GPU-eligible sides fully factored (leftover==1) */
        std::atomic<unsigned long> partial{0}; /* GPU-eligible sides left composite for facul */
        ~gpu_hook_stats() {
            unsigned long const c = calls.load();
            if (c)
                fprintf(stderr,
                    "# GPU ECM cofac hook: %lu calls, %lu primes from GPU,"
                    " %lu MAYBE salvaged, %lu sides full / %lu partial\n",
                    c, split.load(), salvaged.load(), full.load(), partial.load());
        }
    };
    gpu_hook_stats gpu_stats;

    void gpu_ecm_cofactor_hook(std::vector<cxx_mpz> const & n,
                               std::vector<facul_result> & fac,
                               facul_strategies const & strat,
                               gpu_ecm_mode mode)
    {
        gpu_stats.calls.fetch_add(1, std::memory_order_relaxed);

        /* one batched GPU launch over this call's cofactors (eligible ones,
         * i.e. a single odd word < 2^62, are processed; the rest return 1). */
        std::vector<cxx_mpz> const found =
            gpu_ecm::cofac_batch(n, GPU_HOOK_NCURVES, GPU_HOOK_B1, GPU_HOOK_B2);

        for (size_t s = 0; s < n.size(); s++) {
            if (mpz_cmp_ui(found[s], 1) <= 0) continue;     /* GPU found nothing */
            if (!mpz_divisible_p(n[s], found[s])) continue; /* paranoia */
            gpu_stats.split.fetch_add(1, std::memory_order_relaxed);

            if (mode == gpu_ecm_mode::shadow)
                continue;                       /* identity-preserving: only count */

            /* salvage: act only on a facul give-up, never override a verdict */
            if (fac[s].status != FACUL_MAYBE) continue;

            cxx_mpz g;
            mpz_divexact(g, n[s], found[s]);
            unsigned int const lpb = strat.lpb[s];
            /* complete a give-up only when GPU fully splits it into two primes
             * within the large-prime bound -> a genuine smooth relation. */
            if (mpz_sizeinbase(found[s], 2) <= lpb
                && mpz_sizeinbase(g, 2) <= lpb
                && mpz_probab_prime_p(found[s], 25)
                && mpz_probab_prime_p(g, 25))
            {
                fac[s].status = FACUL_SMOOTH;
                fac[s].primes.clear();
                fac[s].primes.push_back(found[s]);
                fac[s].primes.push_back(g);
                gpu_stats.salvaged.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
} // anonymous namespace
/* }}} */

bool gpu_ecm_batch_enabled()
{
    return g_gpu_mode == gpu_ecm_mode::batch && gpu_ecm::available();
}

void gpu_ecm_hook_params(int & ncurves, unsigned long & B1, unsigned long & B2)
{
    ncurves = GPU_HOOK_NCURVES;
    B1 = GPU_HOOK_B1;
    B2 = GPU_HOOK_B2;
}

unsigned int gpu_ecm_min_cofactor_bits()
{
    return GPU_HOOK_MINBITS;
}

facul_status factor_leftover_norms(
        std::vector<cxx_mpz> const & n,
        std::vector<std::vector<cxx_mpz>> & factors,
        std::vector<unsigned long> const & Bs,
        facul_strategies const & strat,
        std::vector<std::vector<cxx_mpz>> const & gpu_factors,
        std::vector<cxx_mpz> const & gpu_leftover)
{
    ASSERT_ALWAYS(Bs.size() == strat.B.size());
    ASSERT_ALWAYS(std::ranges::equal(Bs, strat.B));

    /* Batched GPU drain (CADO_GPU_ECM=batch): the per-bucket-region GPU ECM
     * pre-pass (gpu_cofac.cpp) fully factors each cofactor as far as ECM can,
     * giving per side a list of prime factors (gpu_factors[side]) and the
     * remaining part for facul (gpu_leftover[side], == 1 when fully factored).
     * facul therefore runs only on the (smaller, often trivial) leftover -- and
     * when the leftover is 1 it is skipped entirely. We re-attach the GPU primes
     * afterwards, so product(primes)==norm holds. A GPU-found prime that exceeds
     * the large-prime bound makes the side NOT_SMOOTH without any facul work --
     * the big saving on the non-smooth majority. Valid superset, never wrong
     * (the GPU primes are verified prime and dividing in gpu_cofac.cpp). */
    bool const any_hint = !gpu_leftover.empty();
    if (any_hint) gpu_stats.calls.fetch_add(1, std::memory_order_relaxed);

    /* call the facul library on the leftovers (== n when no hint) */
    auto fac = facul_all(any_hint ? gpu_leftover : n, strat);

    /* when the GPU fully factored a side (leftover == 1) the side is smooth by
     * construction -- the prime factors are all in gpu_factors[side]. facul on a
     * trivial input of 1 does not return a clean SMOOTH, so override its verdict
     * here; this is what lets a complete GPU factorization skip facul entirely. */
    if (any_hint)
        for (size_t s = 0; s < fac.size() && s < gpu_leftover.size(); s++) {
            bool const gpu_touched = s < gpu_factors.size() && !gpu_factors[s].empty();
            if (mpz_cmp_ui(gpu_leftover[s], 1) == 0) {
                fac[s].status = FACUL_SMOOTH;
                fac[s].primes.clear();
                if (gpu_touched) gpu_stats.full.fetch_add(1, std::memory_order_relaxed);
            } else if (gpu_touched) {
                gpu_stats.partial.fetch_add(1, std::memory_order_relaxed);
            }
        }

    /* per-call GPU hook (shadow/salvage) only when NOT using a batch hint */
    if (!any_hint && (g_gpu_mode == gpu_ecm_mode::shadow
                      || g_gpu_mode == gpu_ecm_mode::salvage)
            && gpu_ecm::available())
        gpu_ecm_cofactor_hook(n, fac, strat, g_gpu_mode);

    for(auto const & f : fac) {
        if (f.status == FACUL_NOT_SMOOTH)
            return FACUL_NOT_SMOOTH;
    }
    /* a GPU-extracted prime above the large-prime bound -> not smooth */
    if (any_hint)
        for (size_t side = 0; side < gpu_factors.size(); side++)
            for (auto const & p : gpu_factors[side])
                if (mpz_sizeinbase(p, 2) > strat.lpb[side])
                    return FACUL_NOT_SMOOTH;

    for(size_t side = 0 ; side < fac.size() ; side++) {
        auto & f = fac[side];
        if (f.status == FACUL_MAYBE) {
            /* We couldn't factor this number. So we don't know. It
             * happens also for tiny examples, which is a bit of a pity
             * (see full_p30_JL test for example).
             * Maybe it's due to our incomplete backtracking.
             * I don't have a firm opinion as to whether this needs
             * further investigation or not. At any rate, a "MAYBE" on
             * one side means a global "MAYBE", and for consistency with
             * what the code has been doing for quite some time, let's
             * return that.
             */
            return FACUL_MAYBE;
        }
        /* re-attach the GPU-extracted primes so the side's factorization (and
         * the product==norm invariant below) is complete */
        if (any_hint && side < gpu_factors.size())
            for (auto const & p : gpu_factors[side]) {
                f.primes.push_back(p);
                gpu_stats.split.fetch_add(1, std::memory_order_relaxed);
            }
#ifndef NDEBUG
        cxx_mpz z = 1;
        for(auto const & p : f.primes)
            z *= p;
        ASSERT_ALWAYS(z == n[side]);
#endif
        std::swap(factors[side], f.primes);
    }
    return FACUL_SMOOTH;
}


/*}}}*/
/*}}}*/

