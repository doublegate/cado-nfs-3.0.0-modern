# GPU mixed-representation ECM (Roadmap A2)

This documents the v3.2.0-modern work on item **A2** of the roadmap, *"mixed-
representation ECM cofactorization"* (Bouvier–Imbert, *Faster Cofactorization
with ECM Using Mixed Representations*, Springer 2020). It has two findings — one
an honest "already done upstream", one a new, validated GPU win.

## What ECM cofactorization is (for newcomers)

After the **siever** finds a candidate relation, the part of each norm that the
factor base didn't divide out — the **cofactor** — must be tested for smoothness
(does it split into primes below the large-prime bound?). CADO does this with a
chain of methods in `sieve/ecm/` (P−1, P+1, and the **Elliptic Curve Method**,
ECM). ECM picks a random elliptic curve mod the cofactor and computes a big
scalar multiple `[s]·P` of a point; the cost is dominated by **modular
multiplications** ("modmuls") inside the elliptic-curve point operations. Fewer
modmuls per curve ⇒ more curves per second ⇒ more cofactors cleared.

**Mixed representations** (the paper) means doing the curve arithmetic in
*twisted-Edwards* form (the cheapest doublings/additions known) and switching to
*Montgomery* form where that is cheaper — choosing, per operation, the
representation that minimises modmuls.

## Finding 1 — the CPU path already implements A2 (honest)

CADO's in-sieve `facul` ECM **already uses mixed representations.** The evidence
is in the tree:

- `sieve/ecm/ec_arith_cost.h` defines `EDWARDS_ADDmontgomery 4.` — *"addition
  extended,extended → Montgomery"* — the paper's key trick (a final Edwards
  addition that outputs a Montgomery point, saving 4 M).
- `sieve/ecm/bytecode_mishmash_B1_data.h` is the **"mishmash"** bytecode: a
  precomputed, per-`B1` optimal Edwards addition/doubling/tripling chain, with
  comments like *"Switch to Montgomery / last ADDa is in fact a ADDd / −4 M"*.
- `ec_arith_Edwards.h` + `ec_arith_Montgomery.h` + `bytecode.c` are the Edwards
  and Montgomery arithmetic and the bytecode interpreter that mixes them.

This is unsurprising: the paper's authors (Bouvier, Imbert) are CADO authors, and
upstream 3.0.0 ships their work. **So A2 for the CPU is already done; there is
nothing to add there** (recorded like the other honest "already optimal"
findings).

## Finding 2 — the fork's GPU ECM did *not* — and now there's a validated win

The fork's GPU ECM (`bench/gpu-ecm-mp.cu`, `sieve/ecm/gpu_ecm.cu`, the v3.1.0
`--gpu-prefactor` engine) used a pure **Montgomery XZ ladder**: one doubling
(`cdbl`) + one differential addition (`cadd`) per scalar bit ≈ **11 modmuls/bit**.
That is the natural GPU choice (tiny per-point state: just `X,Z`), but it leaves
the Edwards modmul savings on the table.

Whether porting mixed-representation Edwards to the GPU actually *wins* was
**genuinely uncertain**, so it was measured rather than assumed:

- Twisted-Edwards `a=−1` extended coordinates need **4 field elements/point**
  (`X,Y,Z,T`) versus the ladder's 2 — and a windowed scalar mult needs a
  per-thread precompute **table**. GPU ECM occupancy is already register/limb
  bound (see `BENCHMARKS.md` §3), so the extra state could have erased the win.

### The benchmark — `bench/gpu-ecm-edwards.cu`

Implements, over the existing K-limb CIOS-Montgomery field arithmetic (K∈{2,4,8}
= 128/256/512-bit), twisted-Edwards `a=−1` extended-coordinate arithmetic using
the **exact EFD / Hisil–Wong–Carter–Dawson 2008** formulas (verified against the
Explicit-Formulas Database):

- `edbl` — `dbl-2008-hwcd`, 8 modmuls (`a=−1` makes the `a·X²` term a negation;
  uses no `T` input).
- `eadd` — `add-2008-hwcd`, 9 modmuls (`a=−1` makes `H=B−aA = B+A` free).

Stage-1 `[s]·P` (with `s = ∏ prime-powers ≤ B1`) is computed two ways: plain
**double-and-add**, and **wNAF(w=4)** (signed-window, table = `[1]P,[3]P,[5]P,[7]P`).
Because `s` is identical for every curve in the batch, the wNAF digit stream is
recoded **once on the host** and the device just executes a fixed dbl/add
sequence — **no warp divergence**, the same property that makes the CPU mishmash
bytecode efficient.

### Correctness gate — bit-exact vs the ladder, no square roots

The rigorous check exploits the **Montgomery ↔ twisted-Edwards birational map**
(Bernstein–Birkner–Joye–Lange–Peters 2008): `(u,v)↦(x,y)=(u/v,(u−1)/(u+1))` with
`a=(A+2)/B, d=(A−2)/B`. A random Edwards point `(u0,v0)` is chosen on an `a=−1`
curve and `d` is *derived* (no square root), then mapped to the equivalent
Montgomery curve `(A, x0)`. The Montgomery XZ ladder and the Edwards chain both
compute `[s]·P`; the Edwards result's Montgomery x is `(W+V)/(W−V)`, and we assert

```
    Xladder · (W − V)  ==  Zladder · (W + V)   (mod n)
```

i.e. **identical x([s]P)**, bit-for-bit, host *and* device. Result: **PASS,
0 / 8192 lanes wrong at every width** (and `host == device`, 0/512).

### Measured throughput (RTX 3090, vs the single-scalar Montgomery ladder)

curves/s, `B1=2000` (`s` ≈ 2878 bits), 8192 curves; representative of 3 runs
(small-batch run-to-run variance is moderate — ranges given, not false precision):

| width | ladder | Edwards wNAF4 | speedup | Edwards double-and-add |
|-------|-------:|--------------:|:-------:|:----------------------:|
| 128-bit | ~360 K | ~570–660 K | **~1.5–1.8×** | ~1.1–1.3× (≈ break-even) |
| 256-bit | ~100–123 K | ~136–266 K | **~1.4–2.2×** | ~0.8–0.9× (≈ break-even) |
| 512-bit | ~21–26 K | ~60 K | **~2.3–2.9×** | ~1.7–2.1× |

**Honest reading.** Plain double-and-add is ≈ break-even (theory: ~12.5 modmuls/bit
vs the ladder's 11 — the unwindowed Edwards point adds cost as much as the cheaper
doubling saves). The **win is wNAF**: mostly 8-modmul doublings with sparse
(≈1-in-5) 9-modmul additions, ≈ 9.8 modmuls/bit, and it **grows with modulus
width** — at 512-bit the modmul count dominates the (small, 4-point) table's
occupancy cost, giving ~2.5–2.9×. The feared extended-coordinate occupancy
penalty did **not** materialise at `w=4`. This is a real single-machine GPU win
for the pre-factor / cofactor ECM regime.

### Reproducing

```bash
nvcc -arch=sm_86 -O3 bench/gpu-ecm-edwards.cu -o gpu-ecm-edwards && ./gpu-ecm-edwards
```

## Status & next step

This commit lands the **validated foundation kernel + measured win** — the same
staging used for the C2 collision-search work (foundation first, then live
wiring). The remaining step is to wire the wNAF Edwards stage-1 into the live
engines (`sieve/ecm/gpu_ecm.cu` and the `misc/gpu_prefactor` front-end) behind
the existing GPU paths, gated on the bit-exact relation-set / `product == N`
check, plus a dedicated tripling (`EDWARDS_TPL`, 12 M) for the powers of 3 in
`B1` and the final Edwards→Montgomery switch so stage-2 (the Montgomery BSGS)
reuses the result — the full CPU "mishmash" recipe, on the GPU. That also feeds
**C3** (the GPU batch-smoothness product tree), which reuses this device
arithmetic.

## Sources

- Bouvier, Imbert. *Faster Cofactorization with ECM Using Mixed Representations.*
  PKC 2020 / Springer (10.1007/978-3-030-45388-6_17).
- Hisil, Wong, Carter, Dawson. *Twisted Edwards Curves Revisited.* ASIACRYPT 2008
  — the extended-coordinate `dbl/add-2008-hwcd` formulas (also in the EFD,
  hyperelliptic.org/EFD).
- Bernstein, Birkner, Joye, Lange, Peters. *Twisted Edwards Curves.* AFRICACRYPT
  2008 (eprint 2008/013) — the Montgomery ↔ twisted-Edwards birational map.
- CADO-NFS `sieve/ecm/` — the upstream "mishmash" mixed-representation bytecode.
