# GPU GF(p) lingen NTT (Roadmap C6)

> **Status: in progress (v3.3.0-modern), research-grade.** Tiny single-machine net;
> the real value is multi-GPU / cluster DLP. Framed and gated as such. See
> [`ROADMAP-v3.3.0-modern.md`](ROADMAP-v3.3.0-modern.md).

## What lingen is (for newcomers)

Block Wiedemann (BWC) solves the giant sparse linear system at the heart of NFS. It
runs in three parts: **krylov** (build a matrix sequence by repeated sparse
matrix×vector products), **lingen** (the *linear generator*: compute a small matrix
polynomial that "explains" that sequence, via a matrix Berlekamp–Massey /
half-gcd), and **mksol** (reconstruct the solution). Lingen's core operation is a
**polynomial-matrix middle product**; Thomé's subquadratic algorithm reduces it to
fast polynomial multiplication, which over GF(p) is an **NTT** (number-theoretic
transform — an FFT in a finite field).

## What's CPU-only today

The Fourier path is `linalg/bwc/lingen_matpoly_ft.cpp`, selected via
`lingen_fft_select.hpp`. The GF(2) factorization path
(`lingen_matpoly_binary.cpp`) is already fast via gf2x; the **GF(p) / DLP** path
(`lingen_qcode_prime.cpp`, FLINT-backed NTT) is the GPU candidate. GPU NTT
implementations exist (custom radix kernels; cuFFT-over-a-prime) and report large
kernel speedups in other domains.

## What C6 delivers

`bench/gpu-lingen-ntt.cu` implements the **GPU NTT core** of the GF(p) polynomial
multiply — iterative Cooley–Tukey (bit-reverse + `log2 N` butterfly stages with a
precomputed twiddle table), pointwise multiply, inverse NTT, scale by `N^-1`. A
linear polynomial multiply is a zero-pad-to-`2N` cyclic convolution. The prime is
`p = 15·2^27 + 1` (a standard 31-bit NTT prime, primitive root 31), so products
`a·b < 2^62` stay in `uint64`.

**Validated bit-exact** (mod p) vs an `O(n²)` schoolbook reference: **PASS, 0/1199
wrong** at degree 600.

**Measured (RTX 3090):** a degree-65536 × degree-65536 multiply (NTT size 2^17)
runs in **~0.5 ms** (kernels only).

## Honest scope — the single-prime inner transform, and why the net is tiny

Two honest qualifiers:

1. **This is one prime of a multi-modular basis.** A real GF(p) lingen has
   *large* (hundreds-of-bit) coefficients; FLINT-style NTT handles them by
   multi-modular CRT over **several** NTT primes. This bench is the GPU-fittable
   *inner* transform (one prime); the CPU CRT wrapper that would surround it (split
   coefficients into residues, recombine) is not reimplemented.
2. **lingen is ~3–8 % of BWC.** Even a large NTT speedup is **<1 % of a
   single-machine run** — this is explicitly *not* a single-machine win. The value
   is at multi-GPU / cluster DLP scale, where the distributed lingen's polynomial
   products dominate, gated behind `CADO_GPU_LINGEN_NTT`. Recorded as a validated,
   measured kernel with a scale/HW-gated integration, not a desktop win.

## C6+ (v3.4.0-modern) — the multi-modular CRT wrapper

C6 validated a single-NTT-prime GPU multiply and noted that real GF(p) lingen
coefficients are hundreds of bits wide and therefore need a CPU multi-modular CRT
wrapper over several such primes (exactly what FLINT-style NTT does). C6+ supplies
and validates that wrapper (`bench/gpu-lingen-ntt-crt-c6plus.cu`):

```
for each NTT prime p_i:   A_i = A mod p_i,  B_i = B mod p_i
                          C_i = GPU-NTT-multiply(A_i, B_i) mod p_i
per output coefficient:   c   = CRT_i(C_i)            (the exact integer)
                          c   = c mod P_target        (the GF(p) result)
```

The GPU NTT is now parameterised by modulus (so the same kernel serves every
prime), and per-coefficient reconstruction uses Garner's algorithm. Four
NTT-friendly primes (2013265921, 2281701377, 3221225473, 3489660929; product
~2²³) are validated against an `__int128` integer convolution:

| check | result |
|---|---|
| CRT reconstruction == integer convolution (degree 1500×1500, NTT size 2¹²) | **PASS, 0/2999 wrong** |
| reduced mod a ~107-bit target prime | **PASS, 0/2999 wrong** |

**Honest scope (unchanged):** lingen is ~3–8 % of BWC, so even a fast NTT is
<1 % of a single-machine run — this validates the CRT *mechanism* (the piece C6
said was missing for real GF(p) coefficients), not a single-machine win. A
production ~256-bit DLP prime needs ~17 NTT primes and a bignum CRT; the shape is
identical, only wider. Multi-GPU / cluster-DLP play, HW/scale-gated. Validate with
`bench/gpu-research-v340-validate.sh`.
