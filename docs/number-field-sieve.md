# The Mathematics of the Number Field Sieve

This document explains the algorithm and the mathematics that CADO-NFS
implements: the **(General) Number Field Sieve**, or **GNFS**, the
asymptotically fastest known classical algorithm for factoring large integers
(and, in a variant, for computing discrete logarithms in finite fields).

It is written to be read alongside the code: each phase below corresponds to a
top-level directory of this repository, and the final section maps the two
together. Mathematical notation is rendered with GitHub's LaTeX support.

> **Audience.** Some familiarity with elementary number theory and abstract
> algebra (rings, ideals, finite fields) is assumed. No prior knowledge of NFS
> is required; the ideas are built up from the classical congruence-of-squares
> method.

---

## 1. The goal: a congruence of squares

We are given a large odd composite $N$ (with no small factors and not a prime
power) and want a non-trivial factor. Every modern general-purpose factoring
method since the 1920s rests on one idea, due in spirit to Fermat and made
systematic by Kraitchik: find two integers $x, y$ with

$$x^2 \equiv y^2 \pmod{N}, \qquad x \not\equiv \pm y \pmod{N}.$$

Then $N \mid (x-y)(x+y)$ but $N \nmid (x-y)$ and $N \nmid (x+y)$, so

$$\gcd(x - y,\, N)$$

is a **non-trivial factor** of $N$. For a random such congruence the factor is
non-trivial with probability at least $\tfrac{1}{2}$, so a handful of
independent congruences suffices.

**A tiny example (direct difference of squares).** To factor $N = 5959$, note
$\lceil\sqrt{N}\rceil = 78$ and search $x^2 - N$ for a perfect square:

$$80^2 - 5959 = 6400 - 5959 = 441 = 21^2 \;\Rightarrow\; 80^2 - 21^2 = 5959,$$

so $5959 = (80-21)(80+21) = 59 \cdot 101$. Fermat's method finds such a
*direct* difference of squares, but it is hopeless when the factors are far
apart. The sieve methods instead manufacture congruences $x^2 \equiv y^2$
*modulo $N$* by combining many small relations. The art is in producing those
relations cheaply — which is where number fields enter.

---

## 2. Lineage: why number fields?

| Method | Idea | Heuristic cost |
|---|---|---|
| Trial division | test all primes $\le \sqrt N$ | $\exp\!\big(\tfrac12\ln N\big)$ |
| Pollard $\rho$ | cycle finding | $\exp\!\big(\tfrac14\ln N\big)$ |
| Quadratic Sieve (QS) | smooth values of $x^2 - N$ | $L_N[\tfrac12,\,1]$ |
| **Number Field Sieve** | smooth *algebraic* norms | $L_N[\tfrac13,\,(64/9)^{1/3}]$ |

The Quadratic Sieve collects integers $a$ for which $a^2 - N$ is **smooth**
(factors entirely into small primes); a linear-algebra step then multiplies a
subset of them into a square. NFS replaces the single quadratic polynomial
$a^2 - N$ with two cleverly chosen polynomials whose values are much smaller,
hence far more likely to be smooth. Smaller numbers to factor $\Rightarrow$
the exponent $\tfrac12$ drops to $\tfrac13$. That is the whole story; the rest
is making it precise.

---

## 3. Mathematical toolkit

### 3.1 Smoothness and the $L$-notation

An integer is **$B$-smooth** if all of its prime factors are $\le B$. NFS lives
or dies on how often the numbers it produces are smooth. The density of smooth
numbers is captured by the **Dickman–de Bruijn** function $\rho$: the
probability that a random integer in $[1,x]$ is $x^{1/u}$-smooth is
$\rho(u) = u^{-u(1+o(1))}$.

Sub-exponential running times are written with the $L$-notation

$$L_N[\alpha,\,c] \;=\; \exp\!\Big( \big(c + o(1)\big)\,(\ln N)^{\alpha}\,(\ln\ln N)^{1-\alpha} \Big), \qquad 0 \le \alpha \le 1 .$$

$\alpha = 1$ is fully exponential (in $\ln N$); $\alpha = 0$ is polynomial.
GNFS achieves $\alpha = \tfrac13$.

### 3.2 Number fields, orders, and norms

Let $f \in \mathbb{Z}[x]$ be irreducible of degree $d$ with a complex root
$\alpha$. Then

$$K = \mathbb{Q}(\alpha) \cong \mathbb{Q}[x]/(f)$$

is a **number field** of degree $d$, and $\mathbb{Z}[\alpha] \subseteq
\mathcal{O}_K$ is an order in its ring of integers. Writing $f(x) = c_d
\prod_{i=1}^{d}(x - \alpha_i)$ with conjugate roots $\alpha_i$, the **norm** of
the element $a - b\alpha$ (for integers $a,b$) is the product over conjugates:

$$N_{K/\mathbb{Q}}(a - b\alpha) \;=\; \prod_{i=1}^{d} (a - b\alpha_i) \;=\; \frac{1}{c_d}\,F(a,b), \qquad F(a,b) := b^{d} f\!\Big(\frac{a}{b}\Big) = \sum_{i=0}^{d} c_i\, a^{i} b^{\,d-i}.$$

The key point: **the norm is a homogeneous integer polynomial $F(a,b)$ in the
sieve variables.** (The leading coefficient $c_d$ is a fixed constant folded
into bookkeeping; smoothness is tested on the integer $F(a,b)$.)

### 3.3 Prime ideals of degree one

Smoothness of the *element* $a - b\alpha$ means smoothness of the *ideal*
$(a-b\alpha)$, which factors into prime ideals of $\mathcal{O}_K$. For NFS only
**degree-one** prime ideals matter, and these have a beautifully concrete
description: a degree-one prime above the rational prime $p$ corresponds to a
root of $f$ modulo $p$,

$$\mathfrak{p} \;\leftrightarrow\; (p, r) \quad\text{with}\quad f(r) \equiv 0 \pmod{p},$$

and $\mathfrak{p}$ divides $(a - b\alpha)$ exactly when

$$a \equiv b\,r \pmod{p}.$$

So the **algebraic factor base** is just the finite list of pairs
$\{(p,r) : f(r)\equiv 0 \ (\mathrm{mod}\ p),\ p \le B\}$ — and divisibility is a
single congruence, which is what makes sieving possible.

---

## 4. The two-polynomial setup

Choose **two** irreducible polynomials $f, g \in \mathbb{Z}[x]$, of degrees $d$
and $e$, that share a common root $m$ modulo $N$:

$$f(m) \equiv 0 \pmod{N}, \qquad g(m) \equiv 0 \pmod{N}.$$

(Classically $g$ is linear, $g(x) = x - m$, the "rational side," and $f$ has
degree $d \in \{4,5,6\}$, the "algebraic side." CADO-NFS also supports two
non-linear polynomials.) They define two number fields,

$$\mathbb{Q}(\alpha),\quad f(\alpha)=0 \qquad\text{and}\qquad \mathbb{Q}(\beta),\quad g(\beta)=0,$$

and — crucially — two ring homomorphisms into $\mathbb{Z}/N\mathbb{Z}$ that both
send the abstract root to the shared value $m$:

$$\varphi_f : \mathbb{Z}[\alpha] \to \mathbb{Z}/N\mathbb{Z},\ \ \alpha \mapsto m, \qquad \varphi_g : \mathbb{Z}[\beta] \to \mathbb{Z}/N\mathbb{Z},\ \ \beta \mapsto m.$$

These are well-defined precisely because $f(m) \equiv g(m) \equiv 0 \pmod N$.

---

## 5. The central identity

Suppose we find a finite set $S$ of coprime integer pairs $(a,b)$ such that the
two products are **simultaneously squares**, one in each ring:

$$\prod_{(a,b)\in S} (a - b\alpha) = \gamma^2 \in \mathbb{Z}[\alpha], \qquad \prod_{(a,b)\in S} (a - b\beta) = \delta^2 \in \mathbb{Z}[\beta].$$

Apply the homomorphisms and use that they are multiplicative:

$$\varphi_f(\gamma)^2 = \varphi_f\!\Big(\prod (a-b\alpha)\Big) = \prod (a - bm) = \varphi_g\!\Big(\prod (a-b\beta)\Big) = \varphi_g(\delta)^2 \pmod{N}.$$

Setting $x = \varphi_f(\gamma)$ and $y = \varphi_g(\delta)$ gives exactly the
**congruence of squares** $x^2 \equiv y^2 \pmod N$ from §1, and
$\gcd(x-y, N)$ splits $N$. (With $g(x)=x-m$ linear, the right-hand product is
the ordinary integer $\prod (a-bm)$, and $y$ is an ordinary modular square
root.)

Everything else in NFS is machinery to **find such a set $S$ efficiently**.
The strategy:

1. Collect many pairs $(a,b)$ for which *both* norms are smooth (**relations**).
2. Each relation records the parity of every prime exponent on both sides.
   A square is exactly an element all of whose exponents are even.
3. Finding a subset $S$ whose combined exponent vector is **zero modulo 2** is a
   **linear-algebra problem over $\mathbb{F}_2$**.
4. Convert the resulting "square of ideals" into an actual congruence of
   integers (the square-root step), then take a gcd.

The five phases below carry this out.

---

## 6. Phase 1 — Polynomial selection  ·  `polyselect/`

Goal: pick $f$ and $g$ sharing a root $m \bmod N$ that make the norms
$F(a,b)$ and $G(a,b)$ **as small and as smooth-friendly as possible** over the
sieve region. Polynomial quality is the single biggest lever on total runtime —
a better pair can cut the sieving time by a large constant factor.

A simple construction (**base-$m$**): pick $m \approx N^{1/(d+1)}$, write $N$ in
base $m$,

$$N = \sum_{i=0}^{d} c_i\, m^{i}, \qquad 0 \le c_i < m, \quad\text{then}\quad f(x) = \sum_{i=0}^d c_i x^i,\ \ g(x) = x - m.$$

Then $f(m) = N \equiv 0$ and $g(m) = 0 \pmod N$. CADO-NFS uses much stronger
methods (**Kleinjung's algorithm** and its refinements) that search enormous
families for polynomials with small coefficients and many roots modulo small
primes.

Quality is scored before sieving by:

- **Size** — the magnitude of the coefficients / norms over the region.
- **Root property** $\alpha(f)$ — measures how much more often than average
  $F(a,b)$ is divisible by small primes (good roots make smoothness more
  likely); contributes $e^{\alpha}$ to the effective smoothness.
- **Murphy's $E$** — an integral estimating the expected number of relations,

$$E(f,g) \approx \int \rho\!\Big(\frac{\ln |F| + \alpha(f)}{\ln B_f}\Big)\, \rho\!\Big(\frac{\ln |G| + \alpha(g)}{\ln B_g}\Big)\, \mathrm{d}(\text{region}).$$

- **Skewness** $s$ — the optimal aspect ratio of the (skewed) sieve rectangle,
  reflecting that good polynomials are unbalanced in $a$ vs. $b$.

The optimal degree grows slowly with $N$, roughly
$d \sim \big(3\ln N / \ln\ln N\big)^{1/3}$; in practice $d=4$ up to ~110
digits, $d=5$ to ~200, $d=6$ beyond.

---

## 7. Phase 2 — Relation collection (sieving)  ·  `sieve/`

This is the most time-consuming phase. We search a large region of coprime
pairs $(a,b)$, $b > 0$, $\gcd(a,b)=1$, for **relations**: pairs where *both*

$$F(a,b) = b^{d} f(a/b) \quad\text{and}\quad G(a,b) = b^{e} g(a/b)$$

are smooth (up to a few permitted **large primes** above the factor-base bound,
which add flexibility and are matched up later). Testing each value by trial
division would be far too slow, so NFS **sieves**: just as the sieve of
Eratosthenes marks multiples of $p$, here for each factor-base prime $(p,r)$ we
step through the arithmetic progression $a \equiv b r \pmod p$ and add
$\log p$ to a running accumulator at each hit. Positions whose accumulated
log approaches $\log|F(a,b)|$ are the smooth (or nearly smooth) candidates,
confirmed by a quick cofactorization (CADO uses ECM/$p\!-\!1$/$p\!+\!1$ on the
remaining cofactor).

**Lattice (special-$q$) sieving** — CADO's `las` program. To concentrate work
where relations are dense, fix a **special prime** $q$ (a "special-$q$") that is
required to divide one side. The pairs $(a,b)$ with $q \mid F(a,b)$ form a
rank-2 **lattice** $\Lambda_q \subset \mathbb{Z}^2$; reducing its basis (e.g.
Gaussian/Lagrange reduction) gives coordinates $(i,j)$ in which one sieves a
small rectangle, then changes variables back to $(a,b)$. Iterating over many
special-$q$ covers the search space efficiently and is embarrassingly parallel
across $q$ — which is exactly why this phase scales so well with cores.

Each surviving relation is recorded as the pair $(a,b)$ together with the full
prime factorization of both norms.

---

## 8. Phase 3 — Filtering  ·  `filter/`

Sieving yields far more relations than strictly necessary, in raw form. Before
the linear algebra we compress them:

1. **Duplicate removal** — the same $(a,b)$ can be found under different
   special-$q$; dedup them (`dup1`, `dup2`).
2. **Singleton removal (purge)** — a prime (or prime ideal) that occurs in only
   one relation can never be cancelled to an even exponent, so that relation is
   useless; delete it. Removing it may create new singletons, so iterate.
   We need at least as many relations as factor-base elements involved (the
   matrix must have a non-trivial kernel).
3. **Merging (clique/Gaussian preprocessing)** — combine relations to eliminate
   primes that appear in just a few relations, shrinking the matrix while
   keeping it **sparse**. This trades a slightly denser matrix for far fewer
   rows/columns, which dramatically speeds up the next phase.

The output is a sparse matrix $M$ over $\mathbb{F}_2$. Conceptually each
relation is a row; the columns are:

$$\underbrace{\text{sign of } a-bm}_{1} \;\big|\; \underbrace{\text{rational primes } p \le B_g}_{} \;\big|\; \underbrace{\text{algebraic prime ideals }(p,r)\le B_f}_{} \;\big|\; \underbrace{\text{quadratic characters}}_{}$$

and the entry is the **parity** ($\bmod 2$) of that prime's exponent in the
relation. A subset $S$ of relations whose rows sum to the zero vector is one in
which every exponent is even — a candidate square.

**Quadratic characters (the square obstruction).** Even-ness of every
prime-ideal valuation does *not* guarantee the product is a true square in
$\mathbb{Z}[\alpha]$: units and the 2-part of the class group obstruct it.
Adleman's fix adds extra columns — **quadratic character** maps
$\chi_q(a-b\alpha) = \big(\tfrac{a - b\,s_q}{q}\big)$ (Legendre symbols at
auxiliary primes $q$, with $f(s_q)\equiv 0$) — and requires them to agree. With
enough characters, a kernel vector yields a genuine square with overwhelming
probability.

---

## 9. Phase 4 — Linear algebra  ·  `linalg/bwc/`

We must find non-zero vectors in the (left) **kernel** of the sparse matrix
$M$ over $\mathbb{F}_2$:

$$\mathbf{v}^{\top} M = \mathbf{0} \pmod 2,$$

each solution $\mathbf{v}$ selecting a relation subset $S$ with all-even
exponents. The matrix is enormous (millions of rows for hundred-digit numbers)
but very sparse, so dense Gaussian elimination is out. NFS uses iterative
sparse solvers whose cost is dominated by repeated sparse matrix–vector
products:

- **Block Lanczos**, or
- **Block Wiedemann / Coppersmith** — CADO's choice (the directory name
  `bwc` = *Block Wiedemann/Coppersmith*).

Block Wiedemann computes a Krylov sequence $\{\, \mathbf{x}^{\top} M^{k}
\mathbf{y} \,\}_{k\ge 0}$ for blocks of random vectors, finds a matrix
**linear generator** of that sequence with a block Berlekamp–Massey step
(the `lingen` sub-phase), and reconstructs kernel vectors. It parallelizes and
even distributes over MPI. Its sub-steps in CADO are `prep` → `krylov` →
`lingen` → `mksol` → `gather`.

The phase emits several independent kernel vectors (**dependencies**), each a
candidate $S$; only about half yield a non-trivial factor, so having a few in
hand guarantees success.

---

## 10. Phase 5 — Square root  ·  `sqrt/`

Given a dependency $S$, we now realize the abstract "product of squares" as
concrete integers $x, y \bmod N$.

**Rational side** (with $g(x) = x - m$): the product

$$\prod_{(a,b)\in S} (a - b m) \;=\; y^2$$

is a perfect square integer (guaranteed by the even exponents and the sign
column); compute its integer square root and reduce: $y \bmod N$.

**Algebraic side:** we need $\gamma \in \mathbb{Z}[\alpha]$ with

$$\gamma^2 \;=\; \big(c_d\, f'(\alpha)\big)^2 \!\!\prod_{(a,b)\in S} (a - b\alpha),$$

where the correction factor $\big(c_d f'(\alpha)\big)^2$ (leading coefficient
and derivative) compensates for $\mathbb{Z}[\alpha]$ not being the full ring of
integers, keeping $\gamma$ an algebraic integer. Extracting this square root of
a gigantic algebraic number is itself non-trivial; CADO uses a
**Montgomery–Nguyen-style** approach (a CRT reconstruction from square roots
computed modulo many primes, lifting $f$ into its factors mod $p$). Finally
apply the homomorphism,

$$x \;=\; \varphi_f(\gamma) \;=\; \gamma\big|_{\alpha = m} \bmod N .$$

**The gcd.** Now $x^2 \equiv y^2 \pmod N$ (after dividing out the same
correction factor on the rational side), so

$$\gcd(x - y,\, N)$$

is, with probability $\ge \tfrac12$, a non-trivial factor of $N$. If a given
dependency gives only the trivial $1$ or $N$, try the next one.

---

## 11. Complexity

Heuristically, one balances two competing costs against the smoothness bound
$B$. Larger $B$ makes each norm more likely to be smooth (cheaper sieving per
relation) but needs more relations and a bigger matrix (costlier linear
algebra). Writing $B = L_N[\tfrac13, \beta]$ and optimizing, the smoothness
probabilities (via $\rho$) and the matrix size both come out as
$L_N[\tfrac13, \cdot]$, and the optimum yields the celebrated

$$\boxed{\,L_N\!\left[\tfrac13,\ \sqrt[3]{64/9}\,\right], \qquad \sqrt[3]{64/9} \approx 1.9230\,}$$

for the **General** NFS. For integers of special form $r^e \pm s$ (where a
polynomial with tiny coefficients exists for free), the **Special** NFS does
even better, $L_N[\tfrac13,\,(32/9)^{1/3}]$ with $(32/9)^{1/3}\approx 1.5263$.
The drop from QS's exponent $\tfrac12$ to NFS's $\tfrac13$ is the entire reason
NFS dominates beyond ~100 digits.

See [`../BENCHMARKS.md`](../BENCHMARKS.md) for measured timings on this build;
the empirical "wall-time roughly doubles per +10 digits" there is exactly this
sub-exponential growth in action.

---

## 12. The discrete-logarithm variant

The same framework solves the **discrete logarithm problem** (DLP) in
$\mathbb{F}_p^{\*}$ (and extension fields), which underlies Diffie–Hellman and
DSA. The structure is identical — polynomial selection, sieving for smooth
relations, linear algebra — but with two changes:

- Relations become **linear equations among logarithms** of factor-base
  elements, solved as a sparse linear system **modulo $\ell$** (a large prime
  dividing the group order) instead of modulo $2$.
- A final **individual-logarithm** (descent) step expresses the target as a
  combination of known logs.

CADO-NFS implements this path too; see [`../README.dlp`](../README.dlp). The
shared sieving and linear-algebra engines are why the same codebase serves both
problems.

---

## 13. How the mathematics maps to the code

| Phase (this document) | Repository directory | CADO program(s) | Produces |
|---|---|---|---|
| §6 Polynomial selection | [`polyselect/`](../polyselect) | `polyselect`, rootsieve | $f, g$ sharing root $m \bmod N$ |
| §7 Relation collection | [`sieve/`](../sieve) | `las` (lattice siever), `makefb`, `freerel` | smooth relations $(a,b)$ |
| §8 Filtering | [`filter/`](../filter) | `dup1`, `dup2`, `purge`, `merge`, `replay` | sparse $\mathbb{F}_2$ matrix |
| §9 Linear algebra | [`linalg/bwc/`](../linalg/bwc) | Block Wiedemann (`krylov`, `lingen`, `mksol`) | kernel vectors (dependencies) |
| §10 Square root | [`sqrt/`](../sqrt) | `sqrt` (rational + algebraic) | $x, y$ with $x^2\equiv y^2$ |
| supporting theory | [`numbertheory/`](../numbertheory), [`utils/`](../utils) | ideal arithmetic, helpers | — |

The Python layer in [`scripts/cadofactor/`](../scripts/cadofactor) orchestrates
these as a pipeline of work units, distributing sieving and linear algebra
across cores and machines. Parameter files in
[`parameters/factor/`](../parameters/factor) encode tuned choices (degree,
factor-base bounds $B_f, B_g$, large-prime bounds, sieve region, special-$q$
range) per input size; the driver interpolates between them based on the
number of digits in $N$.

---

## References

- A. K. Lenstra and H. W. Lenstra, Jr. (eds.), *The Development of the Number
  Field Sieve*, Lecture Notes in Mathematics 1554, Springer, 1993.
- J. P. Buhler, H. W. Lenstra, Jr., C. Pomerance, *Factoring integers with the
  number field sieve*, in the above volume.
- C. Pomerance, *A Tale of Two Sieves*, Notices of the AMS, 1996.
- P. L. Montgomery, *Square roots of products of algebraic numbers*, 1994.
- T. Kleinjung, *On polynomial selection for the general number field sieve*,
  Math. Comp., 2006.
- D. Coppersmith, *Solving homogeneous linear equations over GF(2) via block
  Wiedemann*, Math. Comp., 1994.
- The CADO-NFS documentation and source: upstream
  <https://gitlab.inria.fr/cado-nfs/cado-nfs>.

---

*Part of [cado-nfs-2.3.1-modern](https://github.com/doublegate/cado-nfs-2.3.1-modern).
This is expository documentation added by the fork; the implementation and
algorithms are the work of the upstream CADO-NFS team (see [`../AUTHORS`](../AUTHORS)).*
