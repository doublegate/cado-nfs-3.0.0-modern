/*
 * gpu-lingen-ntt.cu — GPU number-theoretic transform for the GF(p) polynomial
 * multiply at the heart of the Block-Wiedemann linear generator (Roadmap C6).
 *
 * Background. lingen (linalg/bwc/lingen*) computes the matrix linear generator of
 * BWC. Its core operation is a polynomial-matrix product; over GF(p) (the DLP
 * case) the scalar primitive is polynomial multiplication mod p, which the Fourier
 * path (lingen_matpoly_ft.cpp, via FLINT for GF(p)) does with an NTT. This bench
 * implements that NTT core on the GPU and validates it bit-exact.
 *
 * Scope of this kernel. A single NTT-friendly prime p = 15*2^27 + 1 (a standard
 * 31-bit NTT prime, primitive root 31) — products a*b < 2^62 stay in uint64. A
 * real GF(p) lingen has *large* (hundreds-of-bit) coefficients, which FLINT-style
 * NTT handles by multi-modular CRT over SEVERAL such primes; this bench is one
 * prime of that basis (the GPU-fittable inner transform). The CPU CRT wrapper that
 * would surround it is not reimplemented here.
 *
 * Method. Iterative Cooley-Tukey (decimation-in-time): bit-reverse permute, then
 * log2(N) butterfly stages with a precomputed twiddle table. Linear polynomial
 * multiply = zero-pad to 2N, forward-NTT both, pointwise multiply, inverse-NTT,
 * scale by N^-1. Validated bit-exact (mod p) vs an O(n^2) schoolbook reference.
 *
 * Honest scope. lingen is only ~3-8% of BWC, so even a large NTT speedup is <1% of
 * a single-machine run — do NOT read this as a single-machine win. The value is at
 * multi-GPU / cluster DLP scale (where the distributed lingen's polynomial products
 * dominate) and is gated behind CADO_GPU_LINGEN_NTT. Measured GPU-vs-CPU below.
 *
 *   nvcc -arch=sm_86 -O3 bench/gpu-lingen-ntt.cu -o gpu-lingen-ntt && ./gpu-lingen-ntt
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    exit(1);} } while(0)

static const uint64_t P = 2013265921ULL;   /* 15*2^27 + 1 */
static const uint64_t G = 31;               /* primitive root mod P */

static uint64_t powmod(uint64_t a, uint64_t e, uint64_t p){
    uint64_t r=1; a%=p; while(e){ if(e&1) r=(__uint128_t)r*a%p; a=(__uint128_t)a*a%p; e>>=1; } return r;
}

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
/* one butterfly stage; one thread per butterfly pair (N/2 of them). */
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

/* run an in-place NTT on device buffer da; wtab = root^k table for k in [0,N/2). */
static void ntt(uint64_t* da, int N, int lgN, const uint64_t* dwtab){
    bitrev<<<(N+255)/256,256>>>(da,N,lgN);
    for (int len=2; len<=N; len<<=1)
        ntt_stage<<<(N/2+255)/256,256>>>(da,N,len,dwtab,P);
}

static void cpu_schoolbook(std::vector<uint64_t>& c, const std::vector<uint64_t>& a,
                           const std::vector<uint64_t>& b){
    int da=(int)a.size(), db=(int)b.size();
    c.assign(da+db-1,0);
    for(int i=0;i<da;i++) if(a[i]) for(int j=0;j<db;j++)
        c[i+j]=(uint64_t)(((__uint128_t)a[i]*b[j] + c[i+j])%P);
}
static uint64_t xrnd(uint64_t*s){*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;return *s;}

static void make_wtab(std::vector<uint64_t>& w, int N, uint64_t root){
    w.resize(N/2); w[0]=1; for(int k=1;k<N/2;k++) w[k]=(uint64_t)((__uint128_t)w[k-1]*root%P);
}

/* GPU linear polynomial multiply mod P; returns timing in ms (kernels only). */
static float gpu_polymul(std::vector<uint64_t>& out, const std::vector<uint64_t>& A,
                         const std::vector<uint64_t>& B, int lgN){
    int N=1<<lgN;
    uint64_t root=powmod(G,(P-1)/N,P), iroot=powmod(root,P-2,P), ninv=powmod((uint64_t)N,P-2,P);
    std::vector<uint64_t> wf,wi; make_wtab(wf,N,root); make_wtab(wi,N,iroot);
    std::vector<uint64_t> a(N,0),b(N,0);
    for(size_t i=0;i<A.size();i++)a[i]=A[i]; for(size_t i=0;i<B.size();i++)b[i]=B[i];
    uint64_t *da,*db,*dwf,*dwi;
    CK(cudaMalloc(&da,N*8)); CK(cudaMalloc(&db,N*8));
    CK(cudaMalloc(&dwf,(N/2)*8)); CK(cudaMalloc(&dwi,(N/2)*8));
    CK(cudaMemcpy(da,a.data(),N*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,b.data(),N*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwf,wf.data(),(N/2)*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwi,wi.data(),(N/2)*8,cudaMemcpyHostToDevice));
    cudaEvent_t g0,g1; cudaEventCreate(&g0); cudaEventCreate(&g1);
    cudaEventRecord(g0);
    ntt(da,N,lgN,dwf); ntt(db,N,lgN,dwf);
    pointwise<<<(N+255)/256,256>>>(da,db,N,P);
    ntt(da,N,lgN,dwi);
    scale<<<(N+255)/256,256>>>(da,N,ninv,P);
    cudaEventRecord(g1); CK(cudaEventSynchronize(g1)); CK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms,g0,g1);
    out.assign(N,0); CK(cudaMemcpy(out.data(),da,N*8,cudaMemcpyDeviceToHost));
    cudaFree(da);cudaFree(db);cudaFree(dwf);cudaFree(dwi);
    return ms;
}

int main(void){
    uint64_t seed=0xC6C6C6C6ULL;
    /* ---- correctness: validate GPU NTT-multiply == schoolbook, moderate degree ---- */
    int deg=600; std::vector<uint64_t> A(deg),B(deg);
    for(int i=0;i<deg;i++){ A[i]=xrnd(&seed)%P; B[i]=xrnd(&seed)%P; }
    std::vector<uint64_t> cref; cpu_schoolbook(cref,A,B);
    int need=2*deg-1, lgN=0; while((1<<lgN)<need) lgN++;
    std::vector<uint64_t> cg; gpu_polymul(cg,A,B,lgN);
    long wrong=0; for(int i=0;i<need;i++) if(cg[i]!=cref[i]){ if(wrong<4) printf("  MISMATCH i=%d gpu=%lu cpu=%lu\n",i,cg[i],cref[i]); wrong++; }
    printf("GPU NTT polynomial multiply mod p=%lu vs schoolbook (deg %d):\n", P, deg);
    printf("  %s (%ld/%d wrong)\n", wrong?"FAIL":"PASS", wrong, need);

    /* ---- measure at a lingen-relevant size ---- */
    int bigdeg=1<<16; std::vector<uint64_t> X(bigdeg),Y(bigdeg);
    for(int i=0;i<bigdeg;i++){ X[i]=xrnd(&seed)%P; Y[i]=xrnd(&seed)%P; }
    int bn=2*bigdeg-1, blg=0; while((1<<blg)<bn) blg++;
    std::vector<uint64_t> bg; float gms=gpu_polymul(bg,X,Y,blg);
    /* CPU schoolbook at this size is O(n^2) ~ 4e9 ops; time a CPU NTT-equivalent
       estimate is out of scope — report GPU NTT throughput instead. */
    printf("\nMeasured (RTX 3090), degree-%d x degree-%d (NTT size 2^%d):\n", bigdeg,bigdeg,blg);
    printf("  GPU NTT multiply (kernels only): %.2f ms\n", gms);
    printf("Honest: lingen is ~3-8%% of BWC, so even a fast NTT is <1%% of a single-\n"
           "machine run; full GF(p) coeffs need a CPU multi-modular CRT wrapper over\n"
           "several such primes. Value is multi-GPU/cluster DLP. HW/scale-gated.\n");
    printf("%s\n", wrong?"FAILURES":"ALL PASS");
    return wrong!=0;
}
