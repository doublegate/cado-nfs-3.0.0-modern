# GPU polyselect stage-2 root-sieve (Roadmap C5)

> **Status: in progress (v3.3.0-modern), experimental.** A real win only at large
> `N`; likely a wash at the c60–c100 sizes testable on the reference box. See
> [`ROADMAP-v3.3.0-modern.md`](ROADMAP-v3.3.0-modern.md).

## Where this sits in polynomial selection (for newcomers)

Polynomial selection has two stages. **Stage 1** searches for a raw polynomial pair
with a small leading coefficient and good "size" (the norms it produces are small);
v3.2.0 C2 offloaded its collision search to the GPU. **Stage 2** — *root
optimisation* (`ropt`) — then rotates/translates the candidate to improve its
**root property** (how often small primes divide its values), scored by **Murphy-E**
(an integral estimating the relation yield). A better polynomial can cut total NFS
time by a meaningful margin, so stage 2 is worth real effort at scale.

## What's CPU-only today

Stage 2 lives entirely on the CPU: `polyselect/ropt_stage2.cpp` (34.5K),
`ropt_main.cpp`, `ropt_linear.cpp`, with the Murphy-E score in `murphyE.cpp`. The
root sieve fills a valuation table over many small primes — an **embarrassingly
parallel** structure that maps naturally onto the GPU, reusing the validated
per-prime modular arithmetic from A2/C2.

## What C5 delivers

`bench/gpu-ropt-stage2.cu` models and validates the **GPU-fittable arithmetic
core** of stage 2 — the root sieve, `polyselect/ropt_stage2.cpp::rootsieve_run_line()`:

```c
for (j = root; j <= j_bound; j += pe)  sa[j] -= sub;   // int16 array
```

across many `(pe, root, sub)` triples. Because the update is pure **subtraction**
(associative mod 2^16), a cell's final value is `(init - Σ sub) mod 2^16`
independent of order, so the GPU scatters with `int32 atomicAdd` into an
accumulator and finalises `sa[j] = (int16_t)(init[j] - acc[j])` — **bit-identical**
to the CPU's stepwise int16 loop (two's-complement wrap matches the cast). This is
the same "validate + measure the GPU-fittable slice, don't reimplement the whole
stage" approach used for C3/C4.

## Measured (RTX 3090, bit-exact)

A 4 M-cell line, 763 `(pe,root,sub)` entries (small primes dominate the strided
work):

| | time |
|---|---|
| CPU int16 root sieve | ~216 ms |
| GPU scatter (kernels only) | ~124 ms |
| ratio | **~1.7× on the raw apply step** |

Correctness: **PASS, 0 mismatches** over all 4 M cells vs the faithful int16 CPU
reference.

## Honest scope — why ~1.7× on the apply step is *not* a stage-2 win

Real `ropt` does **not** sieve one big line: it root-sieves a *small* array
**per rotation candidate**, over a search tree of many candidates. So the relevant
cost is dominated by per-call launch overhead + PCIe round-trips on small arrays,
not the bulk apply throughput this bench isolates. At the c60–c100 sizes testable
on one desktop the tuned, cache-resident CPU loop wins once that overhead is
included — a **wash/negative**, exactly as C2's stage-1 *root-finding* offload was
(the C2 *collision* offload, by contrast, was a real large-N win). The kernel here
shows the apply *can* be faster on raw throughput; translating that into a stage-2
win needs the whole sieve resident on-device across the candidate tree (large-N
only). Recorded as such; the stage-1↔stage-2 threshold in `ropt_param.cpp` is
unchanged.

## C5+ (v3.4.0-modern) — the conditional-launch threshold

C5 left the GPU root-sieve kernel correct but a wash at desktop-testable sizes:
shipping it unconditionally would *regress* small/medium runs. C5+ closes that
with a cheap, calibrated launch heuristic
(`bench/gpu-ropt-threshold-c5plus.cu`): predict the crossover from the problem
dimensions and run the GPU path **only above it**, the CPU loop below — unlocking
the kernel at large N with no small-N regression.

The heuristic is a work-volume threshold with a line-length floor — the same
predicate `ropt_stage2.cpp` would call before deciding to offload:

```c
should_use_gpu(work, L) := work >= 100M scatter-updates  AND  L >= 8M cells
```

Measured RTX-3090 sweep (bit-exact at every size; "match" = the heuristic does
not route to the slower path):

| line L (cells) | scatter work | CPU (ms) | GPU (ms) | faster | heuristic |
|---:|---:|---:|---:|:--:|:--:|
| 65 536 | 0.8 M | 0.40 | 12.1 | CPU | CPU ✓ |
| 262 144 | 3.2 M | 2.5 | 52.4 | CPU | CPU ✓ |
| 1 048 576 | 13 M | 10.3 | 197 | CPU | CPU ✓ |
| 4 194 304 | 52 M | 48.9 | 108 | CPU | CPU ✓ |
| 16 777 216 | 208 M | 1490 | 385 | **GPU (3.9×)** | GPU ✓ |

The crossover sits at ~16 M-cell lines / ~2×10⁸ scatter updates — quantitatively
confirming C5's "large-N only" conclusion. **Honest scope:** the heuristic does
not make the GPU win at small sizes; it routes each size to the measured-faster
path. Validate with `bench/gpu-research-v340-validate.sh`.
