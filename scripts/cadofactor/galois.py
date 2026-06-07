"""Galois-automorphism detection for the algebraic-side polynomial (Roadmap A5).

When the number field defined by the algebraic polynomial ``f`` has a non-trivial
automorphism, NFS relations come in orbits under it: the siever can cover only a
fundamental domain and filtering can quotient the matrix by the group, shrinking
both the sieve region and the matrix by (up to) the automorphism order. CADO
already *implements* this -- the ``--galois autom2.1`` / ``autom2.2`` / ``autom3.1``
/ ``autom3.2`` / ``autom4.1`` / ``autom6.1`` flags (see ``sieve/las-galois.cpp``,
``filter/filter_galois.cpp``) -- but the user must know their field admits one.

This module *detects* which (if any) of those automorphisms ``f`` admits, so the
planner/doctor can recommend the flag (and a future hook could auto-insert it).
The check is exact integer arithmetic: each CADO automorphism is a Mobius map
``sigma: x -> (a x + b)/(c x + d)`` of finite order; ``f`` admits it iff the
homogenised substitution ``(c x + d)^deg * f(sigma(x))`` is a scalar multiple of
``f``. The Mobius coefficients below are taken from the relation transforms in
``sieve/las-galois.cpp`` (testing one generator suffices -- invariance under a
generator implies invariance under the whole cyclic group).

Honest scope: a *generic* GNFS polynomial (what `polyselect` produces for a random
N) has no automorphism, so detection is correctly a no-op there. The win is for
special / SNFS numbers (and DLP targets) whose polynomial is symmetric.
"""

# name -> (Mobius (a, b, c, d) for x -> (a x + b)/(c x + d), group order).
# Derived from the (a,b)-relation transforms in sieve/las-galois.cpp.
AUTOMORPHISMS = [
    ("autom6.1", (-2, -1, 1, -1), 6),
    ("autom4.1", (-1, -1, 1, -1), 4),
    ("autom3.1", (1, -1, 1, 0), 3),   # x -> 1 - 1/x
    ("autom3.2", (-1, -1, 1, 0), 3),  # x -> -1 - 1/x
    ("autom2.1", (0, 1, 1, 0), 2),    # x -> 1/x
    ("autom2.2", (-1, 0, 0, 1), 2),   # x -> -x
]


def _polymul(p, q):
    """Multiply two integer polynomials (ascending-degree coefficient lists)."""
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                r[i + j] += a * b
    return r


def _polypow(p, n):
    """p**n for an integer polynomial p (ascending), n >= 0."""
    r = [1]
    for _ in range(n):
        r = _polymul(r, p)
    return r


def _substitute_homog(coeffs, mobius):
    """Return ``(c x + d)^deg * f((a x + b)/(c x + d))`` as an integer poly list.

    ``coeffs`` is ascending (``[c0, c1, ..., cd]``); ``mobius`` is ``(a, b, c, d)``.
    Equals ``sum_i coeffs[i] * (a x + b)^i * (c x + d)^(deg - i)`` -- an integer
    polynomial of degree ``deg``.
    """
    a, b, c, d = mobius
    deg = len(coeffs) - 1
    num = [b, a]   # a x + b
    den = [d, c]   # c x + d
    out = [0] * (len(coeffs))
    for i, ci in enumerate(coeffs):
        if not ci:
            continue
        term = _polymul(_polypow(num, i), _polypow(den, deg - i))
        for k, t in enumerate(term):
            out[k] += ci * t
    return out


def _proportional(p, q):
    """True iff integer-coeff lists p and q are scalar multiples of each other
    (same length, both non-zero).

    >>> _proportional([1, 0, 1], [2, 0, 2])
    True
    >>> _proportional([1, 0, 1], [-1, 0, -1])
    True
    >>> _proportional([1, 2, 3], [1, 2, 4])
    False
    """
    if len(p) != len(q):
        return False
    # find a pivot where at least one is non-zero
    piv = next((i for i in range(len(p)) if p[i] or q[i]), None)
    if piv is None:
        return False  # both identically zero -- not a meaningful match
    # cross-multiply: p[i]*q[piv] == q[i]*p[piv] for all i
    return all(p[i] * q[piv] == q[i] * p[piv] for i in range(len(p)))


def detect_automorphisms(coeffs):
    """List the CADO automorphisms the algebraic poly admits, highest order first.

    ``coeffs`` is the ascending coefficient list ``[c0, c1, ..., cd]``.

    >>> detect_automorphisms([1, 0, 1])            # x^2 + 1
    [('autom2.1', 2), ('autom2.2', 2)]
    >>> detect_automorphisms([1, -3, 0, 1])        # x^3 - 3x + 1 (cyclic cubic)
    [('autom3.1', 3)]
    >>> detect_automorphisms([3, 1, 5, 1])         # generic -- none
    []
    >>> detect_automorphisms([1, 0, 0, 0, 1])      # x^4 + 1
    [('autom2.1', 2), ('autom2.2', 2)]
    """
    # strip a trailing zero leading coeff defensively
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
    if len(coeffs) < 2:
        return []
    deg = len(coeffs) - 1
    found = []
    for name, mobius, order in AUTOMORPHISMS:
        # Necessary condition for an order-k field automorphism: the roots must
        # partition into k-sized orbits, so deg(f) must be divisible by k.
        # (Polynomial invariance under an order-k Mobius alone is not enough -- on
        # too-low a degree the induced root permutation collapses to lower order;
        # e.g. x^2-x+1 is invariant under x->1-1/x but has only an order-2 field
        # automorphism.)
        if deg % order != 0:
            continue
        g = _substitute_homog(coeffs, mobius)
        if _proportional(g, coeffs):
            found.append((name, order))
    # highest order first (most reduction); AUTOMORPHISMS is already so ordered.
    return found


def best_galois(coeffs):
    """The single most beneficial ``--galois`` flag for ``coeffs``, or None.

    Picks the highest-order automorphism (largest matrix/sieve reduction).

    >>> best_galois([1, -3, 0, 1])
    'autom3.1'
    >>> best_galois([1, 0, 1])
    'autom2.1'
    >>> best_galois([3, 1, 5, 1]) is None
    True
    """
    found = detect_automorphisms(coeffs)
    return found[0][0] if found else None


def read_poly_file(path):
    """Read a CADO ``.poly`` file and return the algebraic-side coefficients
    (ascending ``[c0, c1, ...]``), or None if not found.

    Accepts the ``c0:``..``cd:`` form (and, as a fallback, a ``poly1: a,b,c,..``
    line). Lines like ``n:``, ``Y0:``/``Y1:``, ``skew:`` are ignored.
    """
    cs = {}
    polys = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if len(key) >= 2 and key[0] == "c" and key[1:].isdigit():
                    cs[int(key[1:])] = int(val)
                elif key in ("poly0", "poly1"):
                    polys[key] = [int(x) for x in val.replace(",", " ").split()]
    except (OSError, ValueError):
        return None
    if cs:
        # the c0..cd convention always names the algebraic-side polynomial
        return [cs.get(i, 0) for i in range(max(cs) + 1)]
    # poly0/poly1 form: the Galois automorphism lives on the algebraic (higher-
    # degree) side, so return whichever of the two has the larger degree.
    candidates = [p for p in polys.values() if p and len(p) >= 2]
    return max(candidates, key=len) if candidates else None


def recommend_for_poly_file(path):
    """(flag, order, all_found) for a .poly file; flag/order None if none/unreadable.

    A small convenience wrapper around :func:`read_poly_file` +
    :func:`detect_automorphisms` for the ``--galois-detect`` CLI.
    """
    coeffs = read_poly_file(path)
    if not coeffs:
        return None, None, []
    found = detect_automorphisms(coeffs)
    if not found:
        return None, None, []
    return found[0][0], found[0][1], found
