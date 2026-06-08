/*
 * gpu-lingen-ntt-crt-c6plus.cu — multi-modular CRT wrapper around the GPU GF(p)
 * lingen NTT (Roadmap C6+, the follow-up to C6 / bench/gpu-lingen-ntt.cu).
 *
 * The v3.3.0 C6 bench validated a SINGLE NTT-friendly-prime polynomial multiply on
 * the GPU and noted, honestly, that a real GF(p) lingen has large (hundreds-of-bit)
 * coefficients and therefore needs a CPU multi-modular CRT wrapper over SEVERAL such
 * primes (exactly what FLINT-style NTT does). C6+ supplies that wrapper and validates
 * the whole pipeline bit-exact:
 *
 *   for each NTT prime p_i:  A_i = A mod p_i,  B_i = B mod p_i
 *                            C_i = GPU-NTT-multiply(A_i, B_i)  mod p_i
 *   per output coefficient:  c    = CRT_i(C_i)                  (the exact integer)
 *                            c    = c mod P_target              (the GF(p) result)
 *
 * Validation. The true product polynomial over the integers is computed with an
 * __int128 schoolbook convolution; the CRT-reconstructed coefficients must match it
 * exactly, and then match it again after reduction mod the target DLP prime. The K
 * NTT primes are chosen so their product exceeds the largest possible output
 * coefficient (degree * (coeff_bound)^2), so CRT is exact and fits in 128 bits.
 *
 * Honest scope (unchanged from C6). lingen is ~3-8% of BWC, so even a fast NTT
 * multiply is <1% of a single-machine run; the value is multi-GPU / cluster DLP,
 * gated behind CADO_GPU_LINGEN_NTT. A production ~256-bit DLP prime needs ~17 NTT
 * primes and a bignum CRT — the mechanism is identical to this bench, only wider.
 * This validates the wrapper; it is not a single-machine speed claim.
 *
 *   nvcc -arch=sm_86 -O3 bench/gpu-lingen-ntt-crt-c6plus.cu -o gpu-lingen-ntt-crt \
 *     && ./gpu-lingen-ntt-crt
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    exit(1);} } while(0)

typedef unsigned __int128 u128;

/* Four NTT-friendly primes (each = k*2^a + 1 with a >= 24, so they support the
 * transform sizes here) with a known primitive root. Product ~2^123. */
struct NttPrime { uint64_t mod, root; };
static const NttPrime NP[] = {
    {2013265921ULL, 31}, /* 15*2^27 + 1 */
    {2281701377ULL, 3},  /* 17*2^27 + 1 */
    {3221225473ULL, 5},  /*  3*2^30 + 1 */
    {3489660929ULL, 3},  /* 13*2^28 + 1 */
};
static const int K = 4;

static uint64_t powmod(uint64_t a, uint64_t e, uint64_t p){
    uint64_t r=1; a%=p; while(e){ if(e&1) r=(__uint128_t)r*a%p; a=(__uint128_t)a*a%p; e>>=1; } return r;
}
static uint64_t invmod(uint64_t a, uint64_t p){ return powmod(a,p-2,p); }

__device__ __forceinline__ uint64_t dmul(uint64_t a, uint64_t b, uint64_t p){
    return (uint64_t)((__uint128_t)a*b % p);
}
__device__ __forceinline__ uint64_t dadd(uint64_t a, uint64_t b, uint64_t p){ uint64_t s=a+b; return s>=p?s-p:s; }
__device__ __forceinline__ uint64_t dsub(uint64_t a, uint64_t b, uint64_t p){ return a>=b?a-b:a+p-b; }

__global__ void bitrev(uint64_t* a, int N, int lgN){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=N) return;
    int j=0; for(int k=0;k<lgN;k++) if(i&(1<<k)) j|=1<<(lgN-1-k);
    if (j>i){ uint64_t t=a[i]; a[i]=a[j]; a[j]=t; }
}
__global__ void ntt_stage(uint64_t* a, int N, int len, const uint64_t* wtab, uint64_t p){
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t>=N/2) return;
    int half=len>>1, blk=t/half, j=t%half, i=blk*len;
    uint64_t w = wtab[(long)(N/len)*j];
    uint64_t u=a[i+j], v=dmul(a[i+j+half], w, p);
    a[i+j]=dadd(u,v,p); a[i+j+half]=dsub(u,v,p);
}
__global__ void pointwise(uint64_t* a, const uint64_t* b, int N, uint64_t p){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i<N) a[i]=dmul(a[i],b[i],p);
}
__global__ void scale(uint64_t* a, int N, uint64_t ninv, uint64_t p){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i<N) a[i]=dmul(a[i],ninv,p);
}

static void ntt(uint64_t* da, int N, int lgN, const uint64_t* dwtab, uint64_t p){
    bitrev<<<(N+255)/256,256>>>(da,N,lgN);
    for (int len=2; len<=N; len<<=1)
        ntt_stage<<<(N/2+255)/256,256>>>(da,N,len,dwtab,p);
}
static void make_wtab(std::vector<uint64_t>& w, int N, uint64_t root, uint64_t p){
    w.resize(N/2); w[0]=1; for(int k=1;k<N/2;k++) w[k]=(uint64_t)((__uint128_t)w[k-1]*root%p);
}

/* GPU linear polynomial multiply mod a single NTT prime; out has size N=2^lgN. */
static void gpu_polymul_modp(std::vector<uint64_t>& out, const std::vector<uint64_t>& A,
                             const std::vector<uint64_t>& B, int lgN, uint64_t p, uint64_t g){
    int N=1<<lgN;
    uint64_t root=powmod(g,(p-1)/N,p), iroot=invmod(root,p), ninv=invmod((uint64_t)N,p);
    std::vector<uint64_t> wf,wi; make_wtab(wf,N,root,p); make_wtab(wi,N,iroot,p);
    std::vector<uint64_t> a(N,0),b(N,0);
    for(size_t i=0;i<A.size();i++)a[i]=A[i]%p; for(size_t i=0;i<B.size();i++)b[i]=B[i]%p;
    uint64_t *da,*db,*dwf,*dwi;
    CK(cudaMalloc(&da,N*8)); CK(cudaMalloc(&db,N*8));
    CK(cudaMalloc(&dwf,(N/2)*8)); CK(cudaMalloc(&dwi,(N/2)*8));
    CK(cudaMemcpy(da,a.data(),N*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,b.data(),N*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwf,wf.data(),(N/2)*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwi,wi.data(),(N/2)*8,cudaMemcpyHostToDevice));
    ntt(da,N,lgN,dwf,p); ntt(db,N,lgN,dwf,p);
    pointwise<<<(N+255)/256,256>>>(da,db,N,p);
    ntt(da,N,lgN,dwi,p);
    scale<<<(N+255)/256,256>>>(da,N,ninv,p);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    out.assign(N,0); CK(cudaMemcpy(out.data(),da,N*8,cudaMemcpyDeviceToHost));
    cudaFree(da);cudaFree(db);cudaFree(dwf);cudaFree(dwi);
}

/* Garner CRT: reconstruct the exact integer from residues r_i mod NP[i].mod,
 * assuming it is < prod(mod_i). Returns a u128 (product here ~2^123). */
static u128 garner(const std::vector<uint64_t>& r){
    /* x = r0 + m0*(t1 + m1*(t2 + ...)) with mixed-radix digits computed via the
     * inverse of the partial product modulo each prime. */
    u128 x = r[0];
    u128 prod = NP[0].mod;
    for (int i=1;i<K;i++){
        uint64_t p = NP[i].mod;
        uint64_t cur = (uint64_t)(x % p);
        uint64_t diff = (r[i] >= cur) ? (r[i]-cur) : (r[i]+p-cur);
        uint64_t pinv = invmod((uint64_t)(prod % p), p);
        uint64_t t = (uint64_t)((__uint128_t)diff * pinv % p);
        x += prod * (u128)t;
        prod *= p;
    }
    return x;
}

static uint64_t xrnd(uint64_t*s){*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;return *s;}

/* print a u128 in decimal (small helper for diagnostics). */
static void print_u128(u128 v){ if(v==0){printf("0");return;} char b[40]; int n=0; while(v){ b[n++]='0'+(int)(v%10); v/=10;} while(n) putchar(b[--n]); }

int main(void){
    uint64_t seed=0xC6C6C6C6ULL;
    /* coefficient bound: < 2^45, so a degree-d convolution coefficient is
     * < d * (2^45)^2 = d * 2^90; with d <= 2^11 that is < 2^101 < prod(NP) ~2^123,
     * so the 4-prime CRT is exact AND fits a u128 (and the __int128 reference). */
    const uint64_t COEFF_BOUND = (1ULL<<45);
    int deg = 1500;                       /* degree of each input polynomial */
    /* a target ~120-bit DLP prime, as a u128 (2^107 - 1 is prime-ish enough for a
     * reduction demo; correctness does not depend on it being prime). */
    u128 Ptarget = ((u128)1<<107) - 1;

    std::vector<uint64_t> A(deg),B(deg);
    for(int i=0;i<deg;i++){ A[i]=xrnd(&seed)%COEFF_BOUND; B[i]=xrnd(&seed)%COEFF_BOUND; }

    int need = 2*deg-1, lgN=0; while((1<<lgN)<need) lgN++;

    /* integer reference convolution in __int128 (exact). */
    std::vector<u128> cref(need, 0);
    for(int i=0;i<deg;i++) if(A[i]) for(int j=0;j<deg;j++)
        cref[i+j] += (u128)A[i]*B[j];

    /* GPU NTT multiply modulo each NTT prime. */
    std::vector<std::vector<uint64_t>> residues(K);
    for (int s=0;s<K;s++)
        gpu_polymul_modp(residues[s], A, B, lgN, NP[s].mod, NP[s].root);

    /* CRT-reconstruct each output coefficient and compare to the reference. */
    long wrong_int=0, wrong_mod=0;
    for (int i=0;i<need;i++){
        std::vector<uint64_t> r(K);
        for (int s=0;s<K;s++) r[s]=residues[s][i];
        u128 c = garner(r);
        if (c != cref[i]) { if(wrong_int<4){ printf("  CRT MISMATCH i=%d got=",i); print_u128(c); printf(" want="); print_u128(cref[i]); printf("\n"); } wrong_int++; }
        if ((c % Ptarget) != (cref[i] % Ptarget)) wrong_mod++;
    }

    printf("C6+ multi-modular CRT wrapper for the GPU GF(p) lingen NTT (RTX 3090)\n");
    printf("  NTT primes (K=%d):", K);
    for (int s=0;s<K;s++) printf(" %lu", NP[s].mod);
    printf("\n  product ~2^123, degree %d x %d (NTT size 2^%d)\n", deg, deg, lgN);
    printf("  CRT == integer convolution : %s (%ld/%d wrong)\n",
           wrong_int?"FAIL":"PASS", wrong_int, need);
    printf("  reduced mod a ~107-bit P    : %s (%ld/%d wrong)\n",
           wrong_mod?"FAIL":"PASS", wrong_mod, need);
    printf("Honest: lingen is ~3-8%% of BWC; this validates the CRT *mechanism* (the\n"
           "piece C6 said was needed for real GF(p) coeffs), not a single-machine win.\n"
           "A production ~256-bit prime needs ~17 NTT primes + bignum CRT; same shape.\n");
    long wrong = wrong_int + wrong_mod;
    printf("%s\n", wrong?"FAILURES":"ALL PASS");
    return wrong!=0;
}
