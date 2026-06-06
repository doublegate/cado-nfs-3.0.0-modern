/*
 * ifma-modmul.c — AVX-512 IFMA batched Montgomery modular multiplication,
 * bit-exactly validated against GMP under Intel SDE (v3.1.0-modern, Track 1.4).
 *
 * This is the foundation kernel for an mpfq GF(p) IFMA backend: it does 8
 * INDEPENDENT modular multiplications at once (one per 512-bit lane) in
 * radix 2^52, using _mm512_madd52lo/hi_epu64 (the AVX-512-IFMA fused
 * 52-bit multiply-add). 52-bit limbs are the natural IFMA radix, so the
 * partial products land exactly in the lo/hi halves madd52 produces.
 *
 * The reference box (Comet Lake) has no AVX-512-IFMA, so — exactly as the
 * gf2x VPCLMULQDQ work (bench/vpclmul-mul1n.c) — we validate under Intel SDE
 * (-future emulates a CPU with IFMA). Full mpfq integration (hooking these
 * into the generated GF(p) arithmetic + the BWC GF(p) SpMV) is follow-up;
 * this proves the arithmetic primitive bit-exact.
 *
 *   nvcc is NOT needed; this is plain C + AVX-512 intrinsics:
 *   gcc -O2 -mavx512f -mavx512ifma ifma-modmul.c -lgmp -o ifma-modmul
 *   SDE=/opt/intel-sde/sde64 bench/ifma-validate.sh    # or run under sde -future
 */
#include <stdio.h>
#include <stdint.h>
#include <gmp.h>
#include <immintrin.h>

#define NLIMBS 5                 /* 5 * 52 = 260-bit capacity (>= 256-bit p) */
#define RADIX  52
#define MASK52 ((1ULL << 52) - 1)
#define LANES  8

/* 8-lane batched CIOS Montgomery multiply, radix 2^52.
 * a,b,m: arrays [NLIMBS] of __m512i, lane L holds the i-th 52-bit limb of the
 * L-th operand. mp = -m^{-1} mod 2^52 (per lane). Result r = a*b*2^(-52*NLIMBS)
 * mod m, in [0, m). Standard CIOS (Koc et al.): interleave multiply and reduce,
 * one extra guard limb; IFMA supplies the 52x52->104 partial products. */
static void mont_mul_ifma(__m512i *r, const __m512i *a, const __m512i *b,
                          const __m512i *m, __m512i mp)
{
    const __m512i vmask = _mm512_set1_epi64(MASK52);
    __m512i t[NLIMBS + 2];
    for (int k = 0; k < NLIMBS + 2; k++) t[k] = _mm512_setzero_si512();

    for (int i = 0; i < NLIMBS; i++) {
        __m512i ai = a[i];
        /* t += a[i] * b  (lo sweep into t[j], hi sweep into t[j+1]) */
        __m512i C = _mm512_setzero_si512();
        for (int j = 0; j < NLIMBS; j++) {
            /* lo = t[j] + lo(ai*b[j]) + C */
            __m512i lo = _mm512_madd52lo_epu64(t[j], ai, b[j]);
            lo = _mm512_add_epi64(lo, C);
            t[j] = _mm512_and_si512(lo, vmask);
            /* carry = (lo >> 52) + hi(ai*b[j]) */
            C = _mm512_madd52hi_epu64(_mm512_srli_epi64(lo, RADIX), ai, b[j]);
        }
        t[NLIMBS] = _mm512_add_epi64(t[NLIMBS], C);

        /* Montgomery reduction step: q = (t[0]*mp) mod 2^52 ; t += q*m ; t >>= 52 */
        __m512i q = _mm512_and_si512(_mm512_madd52lo_epu64(_mm512_setzero_si512(),
                                                           t[0], mp), vmask);
        /* t[0] += lo(q*m[0]); the low limb must become 0, its carry feeds on */
        __m512i lo0 = _mm512_madd52lo_epu64(t[0], q, m[0]);
        C = _mm512_madd52hi_epu64(_mm512_srli_epi64(lo0, RADIX), q, m[0]);
        for (int j = 1; j < NLIMBS; j++) {
            __m512i lo = _mm512_madd52lo_epu64(t[j], q, m[j]);
            lo = _mm512_add_epi64(lo, C);
            t[j - 1] = _mm512_and_si512(lo, vmask);
            C = _mm512_madd52hi_epu64(_mm512_srli_epi64(lo, RADIX), q, m[j]);
        }
        __m512i s = _mm512_add_epi64(t[NLIMBS], C);
        t[NLIMBS - 1] = _mm512_and_si512(s, vmask);
        t[NLIMBS] = _mm512_add_epi64(t[NLIMBS + 1], _mm512_srli_epi64(s, RADIX));
        t[NLIMBS + 1] = _mm512_setzero_si512();
    }

    /* conditional subtract m if t >= m (per lane), borrow-propagated */
    __m512i borrow = _mm512_setzero_si512();
    __m512i d[NLIMBS];
    for (int j = 0; j < NLIMBS; j++) {
        __m512i diff = _mm512_sub_epi64(_mm512_sub_epi64(t[j], m[j]), borrow);
        d[j] = _mm512_and_si512(diff, vmask);
        borrow = _mm512_and_si512(_mm512_srli_epi64(diff, 63), _mm512_set1_epi64(1));
    }
    /* if no final borrow (t >= m) use d, else keep t. borrow==0 -> t>=m */
    __mmask8 ge = _mm512_cmpeq_epi64_mask(borrow, _mm512_setzero_si512());
    for (int j = 0; j < NLIMBS; j++)
        r[j] = _mm512_mask_blend_epi64(ge, t[j], d[j]);
}

/* ---- host helpers: GMP <-> 52-bit limb lanes, and the reference ---- */
static void to_limbs52(uint64_t out[NLIMBS], const mpz_t v) {
    mpz_t t; mpz_init_set(t, v);
    for (int i = 0; i < NLIMBS; i++) {
        out[i] = (uint64_t)(mpz_get_ui(t) & MASK52);
        mpz_fdiv_q_2exp(t, t, RADIX);
    }
    mpz_clear(t);
}
static void from_limbs52(mpz_t v, const uint64_t in[NLIMBS]) {
    mpz_set_ui(v, 0);
    for (int i = NLIMBS - 1; i >= 0; i--) {
        mpz_mul_2exp(v, v, RADIX);
        mpz_add_ui(v, v, in[i]);
    }
}

static uint64_t xrnd(uint64_t *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

int main(void) {
    uint64_t seed = 0xC0FFEEULL;
    mpz_t m, a, b, R, Rinv, mp_z, prod, expect, got, gcd;
    mpz_inits(m, a, b, R, Rinv, mp_z, prod, expect, got, gcd, NULL);
    /* R = 2^(52*NLIMBS) */
    mpz_set_ui(R, 1); mpz_mul_2exp(R, R, RADIX * NLIMBS);

    long trials = 0, wrong = 0;
    for (int rep = 0; rep < 4000; rep++) {
        /* 8 independent odd moduli ~255-bit, and operands < m */
        uint64_t aL[NLIMBS][LANES], bL[NLIMBS][LANES], mL[NLIMBS][LANES], mpL[LANES];
        mpz_t ms[LANES], as[LANES], bs[LANES];
        for (int L = 0; L < LANES; L++) {
            mpz_inits(ms[L], as[L], bs[L], NULL);
            do {
                mpz_set_ui(ms[L], 0);
                for (int i = 0; i < NLIMBS; i++) {
                    mpz_mul_2exp(ms[L], ms[L], RADIX);
                    mpz_add_ui(ms[L], ms[L], xrnd(&seed) & MASK52);
                }
                mpz_setbit(ms[L], 0);                    /* odd */
                mpz_fdiv_q_2exp(ms[L], ms[L], 5);        /* keep < R with headroom */
                mpz_setbit(ms[L], 254);                  /* ~255-bit */
                mpz_setbit(ms[L], 0);
            } while (mpz_cmp_ui(ms[L], 1) <= 0);
            mpz_mod(as[L], ms[L], ms[L]);  /* placeholder */
            mpz_set_ui(as[L], 0); mpz_set_ui(bs[L], 0);
            for (int i = 0; i < NLIMBS; i++) {
                mpz_mul_2exp(as[L], as[L], RADIX); mpz_add_ui(as[L], as[L], xrnd(&seed)&MASK52);
                mpz_mul_2exp(bs[L], bs[L], RADIX); mpz_add_ui(bs[L], bs[L], xrnd(&seed)&MASK52);
            }
            mpz_mod(as[L], as[L], ms[L]);
            mpz_mod(bs[L], bs[L], ms[L]);
            /* mp = -m^{-1} mod 2^52 */
            mpz_t base; mpz_init_set_ui(base, 1); mpz_mul_2exp(base, base, RADIX);
            mpz_invert(mp_z, ms[L], base);
            mpz_sub(mp_z, base, mp_z); mpz_mod(mp_z, mp_z, base);
            mpL[L] = mpz_get_ui(mp_z) & MASK52;
            mpz_clear(base);
            uint64_t tmp[NLIMBS];
            to_limbs52(tmp, as[L]); for (int i=0;i<NLIMBS;i++) aL[i][L]=tmp[i];
            to_limbs52(tmp, bs[L]); for (int i=0;i<NLIMBS;i++) bL[i][L]=tmp[i];
            to_limbs52(tmp, ms[L]); for (int i=0;i<NLIMBS;i++) mL[i][L]=tmp[i];
        }
        __m512i av[NLIMBS], bv[NLIMBS], mv[NLIMBS], rv[NLIMBS];
        for (int i = 0; i < NLIMBS; i++) {
            av[i] = _mm512_loadu_si512((const void*)aL[i]);
            bv[i] = _mm512_loadu_si512((const void*)bL[i]);
            mv[i] = _mm512_loadu_si512((const void*)mL[i]);
        }
        __m512i mpv = _mm512_loadu_si512((const void*)mpL);
        mont_mul_ifma(rv, av, bv, mv, mpv);
        uint64_t rL[NLIMBS][LANES];
        for (int i = 0; i < NLIMBS; i++) _mm512_storeu_si512((void*)rL[i], rv[i]);

        for (int L = 0; L < LANES; L++) {
            /* expected = a*b*R^{-1} mod m */
            mpz_invert(Rinv, R, ms[L]);
            mpz_mul(expect, as[L], bs[L]); mpz_mul(expect, expect, Rinv);
            mpz_mod(expect, expect, ms[L]);
            uint64_t tmp[NLIMBS]; for (int i=0;i<NLIMBS;i++) tmp[i]=rL[i][L];
            from_limbs52(got, tmp);
            trials++;
            if (mpz_cmp(got, expect) != 0) {
                if (wrong < 3) gmp_printf("  MISMATCH lane %d: got %Zd expect %Zd\n",
                                          L, got, expect);
                wrong++;
            }
            mpz_clears(ms[L], as[L], bs[L], NULL);
        }
    }
    printf("%s: IFMA Montgomery modmul bit-exact vs GMP (%ld/%ld wrong, %d-bit, %d-way)\n",
           wrong == 0 ? "PASS" : "FAIL", wrong, trials, RADIX*NLIMBS, LANES);
    mpz_clears(m, a, b, R, Rinv, mp_z, prod, expect, got, gcd, NULL);
    return wrong != 0;
}
