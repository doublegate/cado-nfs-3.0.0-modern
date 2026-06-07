# Galois automorphism auto-detection (Roadmap A5)

> **Status: in progress (v3.3.0-modern).** The one genuinely *measurable*
> algorithmic win on the reference hardware this cycle. See
> [`ROADMAP-v3.3.0-modern.md`](ROADMAP-v3.3.0-modern.md).

## What a Galois automorphism buys you (for newcomers)

NFS works in a number field defined by the algebraic-side polynomial `f`. If `f`'s
field has a non-trivial **automorphism** σ (a symmetry mapping roots to roots),
then relations come in σ-orbits: if `(a,b)` is a relation, so is its image under σ.
That symmetry lets the siever cover only a **fundamental domain** (a 1/k slice for
an order-k automorphism) and lets filtering **quotient the matrix** by the group,
shrinking both the sieve region and the matrix — up to ~2× for a quadratic
automorphism, more for higher order. The classic case is `x^2 + 1` and its
relatives (CM/quadratic-twist-friendly inputs, and many SNFS targets).

## What CADO already has, and the gap

CADO already *implements* the action — `sieve/las-galois.{cpp,hpp}`,
`filter/filter_galois.cpp`, and the `--galois autom2.1` / `autom3.1` / … flags. But
it is **opt-in**: the user must know their field has an automorphism and pass the
right flag. The planner (`--plan`, `--autotune`) does not detect or recommend it.

## What A5 delivers

**`scripts/cadofactor/galois.py` — an exact automorphism detector.** Each CADO
automorphism is a Möbius map `σ: x → (a x + b)/(c x + d)` of finite order; `f`
admits it iff the homogenised substitution `(c x + d)^deg · f(σ(x))` is a scalar
multiple of `f`. The detector computes that substitution in **exact integer
arithmetic** and tests proportionality, for all six CADO maps. The Möbius
coefficients were read off the relation transforms in `sieve/las-galois.cpp`:

| flag | Möbius `(a,b,c,d)` | x ↦ | order |
|------|--------------------|-----|-------|
| autom2.1 | (0,1,1,0) | 1/x | 2 |
| autom2.2 | (−1,0,0,1) | −x | 2 |
| autom3.1 | (1,−1,1,0) | 1 − 1/x | 3 |
| autom3.2 | (−1,−1,1,0) | −1 − 1/x | 3 |
| autom4.1 | (−1,−1,1,−1) | −(x+1)/(x−1) | 4 |
| autom6.1 | (−2,−1,1,−1) | (−2x−1)/(x−1) | 6 |

A crucial guard: polynomial invariance under an order-k Möbius is **not** enough —
the roots must split into k-sized orbits, so `deg(f) % order == 0` is required.
Without it, `x² − x + 1` (a degree-2 field) would falsely match the order-3 map (it
*is* invariant under `1 − 1/x`, but the induced root permutation collapses to order
2). That guard is in the detector and regression-doctested.

**`cado-nfs.py --galois-detect POLYFILE`** reads a `.poly` (the `c0..cd` form or
the `poly0:/poly1:` form, picking the higher-degree algebraic side), reports the
detected automorphism(s), and prints the exact `--galois <name>` flag to add — or
states plainly that a generic polynomial has none (a correct no-op).

## Validation

* **Doctests** (registered as `test_python_galois`): `x²+1 → {2.1, 2.2}`,
  `x³−3x+1 → 3.1` (a cyclic cubic), generic degree-5 → none, plus the
  proportionality and degree-guard edge cases.
* **Cross-validated against CADO's own fixture** `tests/sieve/galois.poly`
  (`x⁴ − 9x² + 101`, the polynomial CADO's `galois_sieve_*` tests sieve with): the
  detector independently returns `autom2.2`, the automorphism that even polynomial
  carries.
* **No-op confirmed** on the generic GNFS `tests/misc/c60.poly` (degree 4, no
  automorphism) — matching the honest scope.

## Scope and the deliberate non-auto-insert

The downstream matrix/sieve reduction is **CADO's existing, upstream-validated
`--galois` feature** (exercised by `tests/sieve/galois_sieve_*` and
`tests/filter/test_filter_galois.sh`); A5 adds the *detection + recommendation*
that tells the user when it applies. Auto-inserting `--galois` into a *live* run is
left **advisory on purpose**: silently changing a factorization's sieve/filter
parameters mid-run is exactly the kind of outward-facing change this fork does not
make without the operator opting in. The recommended flow is `--galois-detect` →
add the flag yourself. (A polyselect-time auto-insert hook in `cadotask.py` is a
documented future option.) The reduction itself is the well-known property of the
Galois quotient: up to k-fold on an order-k automorphism.
