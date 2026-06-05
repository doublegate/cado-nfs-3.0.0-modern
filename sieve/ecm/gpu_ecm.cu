/* CUDA implementation of the batched GPU ECM cofactorization backend declared
 * in gpu_ecm.hpp. The device ECM (Montgomery-curve XZ ladder, stage 1 + stage-2
 * BSGS, on-device binary gcd) is the bit-exact-validated code from
 * bench/gpu-ecm-stage2.cu, packaged as a library callable from facul_all().
 *
 * Scope: factor_batch handles odd moduli < 2^62 (one word; the common cofactor
 * size). factor_batch_128 below handles odd moduli < 2^126 (two words) over the
 * bit-exact-validated 2-limb CIOS montmul (bench/gpu-mont128.cu), for the
 * larger cofactors that arise when mfb > 62 (e.g. c175's mfb1=90). The bridge
 * (gpu_cofac.cpp) routes each cofactor to the right width.
 */
#include "gpu_ecm.hpp"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

typedef uint64_t u64;
typedef unsigned __int128 u128;
#define WBS 60                                 /* stage-2 baby-step table size */

__host__ __device__ static inline u64 mm(u64 a,u64 b,u64 n,u64 np){ unsigned __int128 T=(unsigned __int128)a*b; u64 m=(u64)T*np; unsigned __int128 s=T+(unsigned __int128)m*n; u64 t=(u64)(s>>64); return t>=n?t-n:t; }
__host__ __device__ static inline u64 ad(u64 a,u64 b,u64 n){ u64 s=a+b; return s>=n?s-n:s; }
__host__ __device__ static inline u64 sb(u64 a,u64 b,u64 n){ return a>=b?a-b:a+n-b; }
__device__ static inline int ctz64(u64 x){ return __ffsll((long long)x)-1; }
__device__ static u64 bgcd(u64 a,u64 b){ if(!a)return b; if(!b)return a; int sh=ctz64(a|b); a>>=ctz64(a); do{ b>>=ctz64(b); if(a>b){u64 t=a;a=b;b=t;} b-=a; }while(b); return a<<sh; }

struct PT{ u64 X,Z; };
__host__ __device__ static inline PT dbl(PT p,u64 a,u64 n,u64 np){ u64 A=mm(ad(p.X,p.Z,n),ad(p.X,p.Z,n),n,np),B=mm(sb(p.X,p.Z,n),sb(p.X,p.Z,n),n,np),C=sb(A,B,n); PT r; r.X=mm(A,B,n,np); r.Z=mm(C,ad(B,mm(a,C,n,np),n),n,np); return r; }
__host__ __device__ static inline PT dadd(PT p1,PT p2,PT pd,u64 n,u64 np){ u64 DA=mm(sb(p1.X,p1.Z,n),ad(p2.X,p2.Z,n),n,np),CB=mm(ad(p1.X,p1.Z,n),sb(p2.X,p2.Z,n),n,np),s=ad(DA,CB,n),d=sb(DA,CB,n); PT r; r.X=mm(pd.Z,mm(s,s,n,np),n,np); r.Z=mm(pd.X,mm(d,d,n,np),n,np); return r; }
__device__ static PT ladder(PT P,u64 k,u64 a,u64 n,u64 np){ if(k==1)return P; PT R0=P,R1=dbl(P,a,n,np); int b=63; while(!((k>>b)&1))b--; for(b--;b>=0;b--){ if((k>>b)&1){R0=dadd(R0,R1,P,n,np);R1=dbl(R1,a,n,np);} else {R1=dadd(R0,R1,P,n,np);R0=dbl(R0,a,n,np);} } return R0; }

__global__ void ecm_kernel(const u64* n_,const u64* np_,const u64* R1_,const u64* R2_,const u64* seed_,
                           const u64* s,int ns,const u64* pr,int npr,u64* fac,int lanes,int ncurves)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=lanes) return;
    u64 n=n_[i]; if(n==0){ fac[i]=0; return; }
    u64 np=np_[i],R1=R1_[i],R2=R2_[i];
    u64 a24=seed_[i]%n; if(a24<2)a24=2;
    PT P; P.X=ad(R1,R1,n); P.Z=R1;             /* x0=2 */
    u64 a=mm(a24,R2,n,np);
    for(int t=0;t<ns;t++) P=ladder(P,s[t],a,n,np);   /* stage 1 */
    PT Q=P;
    u64 g=bgcd(mm(Q.Z,1,n,np),n);
    if(g==1 && npr>0){                          /* stage 2 BSGS */
        PT T[WBS]; T[1]=Q; T[2]=dbl(Q,a,n,np);
        for(int r=3;r<WBS;r++) T[r]=dadd(T[r-1],Q,T[r-2],n,np);
        PT Wg=ladder(Q,WBS,a,n,np);
        int m=(int)((pr[0]+WBS-1)/WBS);
        PT V=ladder(Q,(u64)m*WBS,a,n,np), Vp=ladder(Q,(u64)(m-1)*WBS,a,n,np);
        u64 acc=R1;
        for(int k=0;k<npr;k++){ u64 p=pr[k]; int mp=(int)((p+WBS-1)/WBS);
            while(m<mp){ PT Vn=dadd(V,Wg,Vp,n,np); Vp=V; V=Vn; m++; }
            int r=m*WBS-(int)p; if(r<=0||r>=WBS) continue;
            acc=mm(acc, sb(mm(V.X,T[r].Z,n,np), mm(T[r].X,V.Z,n,np), n), n, np);
        }
        g=bgcd(mm(acc,1,n,np),n);
    }
    fac[i]=(g>1 && g<n)?g:0;
    (void)ncurves;
}

/* ===================== 128-bit (2-limb) ECM ============================ *
 * Same Montgomery-curve XZ ladder + stage-2 BSGS as above, but over the
 * bit-exact-validated 2-limb CIOS montmul128 (bench/gpu-mont128.cu). Handles
 * odd moduli up to < 2^126 -- the cofactors that overflow the 64-bit path when
 * mfb > 62 (e.g. c175 has mfb1=90). */
__host__ __device__ static inline u128 mm128(u128 a,u128 b,u128 n,u64 np){
    u64 A[2]={(u64)a,(u64)(a>>64)}, B[2]={(u64)b,(u64)(b>>64)}, N[2]={(u64)n,(u64)(n>>64)};
    u64 t[4]={0,0,0,0};
    for(int i=0;i<2;i++){
        u128 c=0;
        for(int j=0;j<2;j++){ u128 x=(u128)t[j]+(u128)A[j]*B[i]+c; t[j]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[2]+c; t[2]=(u64)x; t[3]=(u64)(x>>64); }
        u64 m=t[0]*np;
        u128 x0=(u128)t[0]+(u128)m*N[0]; c=x0>>64;
        { u128 x=(u128)t[1]+(u128)m*N[1]+c; t[0]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[2]+c; t[1]=(u64)x; c=x>>64; }
        t[2]=t[3]+(u64)c; t[3]=0;
    }
    u128 r=((u128)t[1]<<64)|t[0];
    return r>=n?r-n:r;
}
__host__ __device__ static inline u128 ad128(u128 a,u128 b,u128 n){ u128 s=a+b; return (s>=n||s<a)?s-n:s; }
__host__ __device__ static inline u128 sb128(u128 a,u128 b,u128 n){ return a>=b?a-b:a+n-b; }
__device__ static inline int ctz128(u128 x){ u64 lo=(u64)x; return lo?ctz64(lo):64+ctz64((u64)(x>>64)); }
__device__ static u128 bgcd128(u128 a,u128 b){ if(!a)return b; if(!b)return a; int sh=ctz128(a|b); a>>=ctz128(a); do{ b>>=ctz128(b); if(a>b){u128 t=a;a=b;b=t;} b-=a; }while(b); return a<<sh; }

struct PT128{ u128 X,Z; };
__host__ __device__ static inline PT128 dbl128(PT128 p,u128 a,u128 n,u64 np){ u128 A=mm128(ad128(p.X,p.Z,n),ad128(p.X,p.Z,n),n,np),B=mm128(sb128(p.X,p.Z,n),sb128(p.X,p.Z,n),n,np),C=sb128(A,B,n); PT128 r; r.X=mm128(A,B,n,np); r.Z=mm128(C,ad128(B,mm128(a,C,n,np),n),n,np); return r; }
__host__ __device__ static inline PT128 dadd128(PT128 p1,PT128 p2,PT128 pd,u128 n,u64 np){ u128 DA=mm128(sb128(p1.X,p1.Z,n),ad128(p2.X,p2.Z,n),n,np),CB=mm128(ad128(p1.X,p1.Z,n),sb128(p2.X,p2.Z,n),n,np),s=ad128(DA,CB,n),d=sb128(DA,CB,n); PT128 r; r.X=mm128(pd.Z,mm128(s,s,n,np),n,np); r.Z=mm128(pd.X,mm128(d,d,n,np),n,np); return r; }
__device__ static PT128 ladder128(PT128 P,u64 k,u128 a,u128 n,u64 np){ if(k==1)return P; PT128 R0=P,R1=dbl128(P,a,n,np); int b=63; while(!((k>>b)&1))b--; for(b--;b>=0;b--){ if((k>>b)&1){R0=dadd128(R0,R1,P,n,np);R1=dbl128(R1,a,n,np);} else {R1=dadd128(R0,R1,P,n,np);R0=dbl128(R0,a,n,np);} } return R0; }

__global__ void ecm_kernel128(const u128* n_,const u64* np_,const u128* R1_,const u128* R2_,const u128* seed_,
                              const u64* s,int ns,const u64* pr,int npr,u128* fac,int lanes)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=lanes) return;
    u128 n=n_[i]; if(n==0){ fac[i]=0; return; }
    u64 np=np_[i]; u128 R1=R1_[i],R2=R2_[i];
    u128 a24=seed_[i]; if(a24>=n)a24-=n; if(a24<2)a24=2;  /* seed pre-reduced on host */
    PT128 P; P.X=ad128(R1,R1,n); P.Z=R1;
    u128 a=mm128(a24,R2,n,np);
    for(int t=0;t<ns;t++) P=ladder128(P,s[t],a,n,np);
    PT128 Q=P;
    u128 g=bgcd128(mm128(Q.Z,1,n,np),n);
    if(g==1 && npr>0){
        PT128 T[WBS]; T[1]=Q; T[2]=dbl128(Q,a,n,np);
        for(int r=3;r<WBS;r++) T[r]=dadd128(T[r-1],Q,T[r-2],n,np);
        PT128 Wg=ladder128(Q,WBS,a,n,np);
        int m=(int)((pr[0]+WBS-1)/WBS);
        PT128 V=ladder128(Q,(u64)m*WBS,a,n,np), Vp=ladder128(Q,(u64)(m-1)*WBS,a,n,np);
        u128 acc=R1;
        for(int k=0;k<npr;k++){ u64 p=pr[k]; int mp=(int)((p+WBS-1)/WBS);
            while(m<mp){ PT128 Vn=dadd128(V,Wg,Vp,n,np); Vp=V; V=Vn; m++; }
            int r=m*WBS-(int)p; if(r<=0||r>=WBS) continue;
            acc=mm128(acc, sb128(mm128(V.X,T[r].Z,n,np), mm128(T[r].X,V.Z,n,np), n), n, np);
        }
        g=bgcd128(mm128(acc,1,n,np),n);
    }
    fac[i]=(g>1 && g<n)?g:0;
}

/* host helpers */
static u64 montinv(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }
static u64 rmod(u64 n){ return (u64)(((unsigned __int128)1<<64)%n); }
static u64 r2mod(u64 n){ u64 r=rmod(n); return (u64)(((unsigned __int128)r*r)%n); }

/* 128-bit Montgomery constants (host setup, via binary add-mod like the
 * validated bench/gpu-mont128.cu). R = 2^128. */
static u128 hadd128(u128 a,u128 b,u128 n){ u128 s=a+b; return (s>=n||s<a)?s-n:s; }
static u128 r1mod128(u128 n){ u128 r=1%n; for(int i=0;i<128;i++) r=hadd128(r,r,n); return r; } /* 2^128 mod n */
static u128 r2mod128(u128 n){ u128 r=1%n; for(int i=0;i<256;i++) r=hadd128(r,r,n); return r; } /* 2^256 mod n */

namespace gpu_ecm {

bool available()
{
    int dev=0; cudaError_t e=cudaGetDeviceCount(&dev);
    return e==cudaSuccess && dev>0;
}

void factor_batch(std::vector<uint64_t> const & moduli, int ncurves,
                  unsigned long B1, unsigned long B2, std::vector<uint64_t> & factor)
{
    int M=(int)moduli.size();
    factor.assign(M,0);
    if(M==0 || !available()) return;

    /* stage-1 prime powers <= B1, and primes in (B1,B2] for stage 2 */
    std::vector<char> comp(B2+1,0); std::vector<u64> s,pr;
    for(unsigned long p=2;p<=B2;p++) if(!comp[p]){ for(unsigned long q=(unsigned long)p*p;q<=B2;q+=p) comp[q]=1;
        if(p<=B1){ unsigned long pe=p; while(pe*p<=B1) pe*=p; s.push_back(pe); } else pr.push_back(p); }
    int ns=(int)s.size(), npr=(int)pr.size();

    /* one lane per (modulus, curve); skip non-eligible moduli (lane n=0) */
    int L=M*ncurves;
    std::vector<u64> n(L),np(L),R1(L),R2(L),seed(L);
    u64 rs=0x9E3779B97F4A7C15ULL;
    for(int i=0;i<M;i++){ u64 N=moduli[i]; bool ok=(N>2)&&(N&1)&&(N<(1ULL<<62));
        for(int j=0;j<ncurves;j++){ int l=i*ncurves+j;
            if(ok){ n[l]=N; np[l]=montinv(N); R1[l]=rmod(N); R2[l]=r2mod(N);
                    rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; seed[l]=(rs%N)|2; }
            else  { n[l]=0; np[l]=0; R1[l]=0; R2[l]=0; seed[l]=0; } } }

    u64 *dn,*dnp,*dR1,*dR2,*dse,*ds,*dpr,*dfac;
    cudaMalloc(&dn,L*8);cudaMalloc(&dnp,L*8);cudaMalloc(&dR1,L*8);cudaMalloc(&dR2,L*8);cudaMalloc(&dse,L*8);cudaMalloc(&dfac,L*8);
    cudaMalloc(&ds,(ns?ns:1)*8);cudaMalloc(&dpr,(npr?npr:1)*8);
    cudaMemcpy(dn,n.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dnp,np.data(),L*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR1,R1.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dR2,R2.data(),L*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dse,seed.data(),L*8,cudaMemcpyHostToDevice);
    if(ns) cudaMemcpy(ds,s.data(),ns*8,cudaMemcpyHostToDevice);
    if(npr) cudaMemcpy(dpr,pr.data(),npr*8,cudaMemcpyHostToDevice);

    ecm_kernel<<<(L+63)/64,64>>>(dn,dnp,dR1,dR2,dse,ds,ns,dpr,npr,dfac,L,ncurves);
    cudaDeviceSynchronize();

    std::vector<u64> fl(L); cudaMemcpy(fl.data(),dfac,L*8,cudaMemcpyDeviceToHost);
    for(int i=0;i<M;i++) for(int j=0;j<ncurves;j++){ u64 f=fl[i*ncurves+j]; if(f){ factor[i]=f; break; } }

    cudaFree(dn);cudaFree(dnp);cudaFree(dR1);cudaFree(dR2);cudaFree(dse);cudaFree(dfac);cudaFree(ds);cudaFree(dpr);
}

void factor_batch_128(std::vector<uint64_t> const & mod_lo,
                      std::vector<uint64_t> const & mod_hi,
                      int ncurves, unsigned long B1, unsigned long B2,
                      std::vector<uint64_t> & fac_lo,
                      std::vector<uint64_t> & fac_hi)
{
    int M=(int)mod_lo.size();
    fac_lo.assign(M,0); fac_hi.assign(M,0);
    if(M==0 || !available()) return;

    /* same stage-1 prime powers <= B1 and stage-2 primes in (B1,B2] */
    std::vector<char> comp(B2+1,0); std::vector<u64> s,pr;
    for(unsigned long p=2;p<=B2;p++) if(!comp[p]){ for(unsigned long q=(unsigned long)p*p;q<=B2;q+=p) comp[q]=1;
        if(p<=B1){ unsigned long pe=p; while(pe*p<=B1) pe*=p; s.push_back(pe); } else pr.push_back(p); }
    int ns=(int)s.size(), npr=(int)pr.size();

    int L=M*ncurves;
    std::vector<u128> n(L),R1(L),R2(L),seed(L); std::vector<u64> np(L);
    const u128 LIM=(u128)1<<126;
    u64 rs=0x9E3779B97F4A7C15ULL;
    for(int i=0;i<M;i++){ u128 N=((u128)mod_hi[i]<<64)|mod_lo[i]; bool ok=(N>2)&&((u64)N&1)&&(N<LIM);
        for(int j=0;j<ncurves;j++){ int l=i*ncurves+j;
            if(ok){ n[l]=N; np[l]=montinv((u64)N); R1[l]=r1mod128(N); R2[l]=r2mod128(N);
                    rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; u128 sd=(u128)rs<<64;
                    rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; sd|=rs; seed[l]=(sd%N)|2; }
            else  { n[l]=0; np[l]=0; R1[l]=0; R2[l]=0; seed[l]=0; } } }

    u128 *dn,*dR1,*dR2,*dse,*dfac; u64 *dnp,*ds,*dpr;
    cudaMalloc(&dn,L*16);cudaMalloc(&dR1,L*16);cudaMalloc(&dR2,L*16);cudaMalloc(&dse,L*16);cudaMalloc(&dfac,L*16);
    cudaMalloc(&dnp,L*8);cudaMalloc(&ds,(ns?ns:1)*8);cudaMalloc(&dpr,(npr?npr:1)*8);
    cudaMemcpy(dn,n.data(),L*16,cudaMemcpyHostToDevice);cudaMemcpy(dR1,R1.data(),L*16,cudaMemcpyHostToDevice);
    cudaMemcpy(dR2,R2.data(),L*16,cudaMemcpyHostToDevice);cudaMemcpy(dse,seed.data(),L*16,cudaMemcpyHostToDevice);
    cudaMemcpy(dnp,np.data(),L*8,cudaMemcpyHostToDevice);
    if(ns) cudaMemcpy(ds,s.data(),ns*8,cudaMemcpyHostToDevice);
    if(npr) cudaMemcpy(dpr,pr.data(),npr*8,cudaMemcpyHostToDevice);

    ecm_kernel128<<<(L+63)/64,64>>>(dn,dnp,dR1,dR2,dse,ds,ns,dpr,npr,dfac,L);
    cudaDeviceSynchronize();

    std::vector<u128> fl(L); cudaMemcpy(fl.data(),dfac,L*16,cudaMemcpyDeviceToHost);
    for(int i=0;i<M;i++) for(int j=0;j<ncurves;j++){ u128 f=fl[i*ncurves+j]; if(f){ fac_lo[i]=(u64)f; fac_hi[i]=(u64)(f>>64); break; } }

    cudaFree(dn);cudaFree(dR1);cudaFree(dR2);cudaFree(dse);cudaFree(dfac);cudaFree(dnp);cudaFree(ds);cudaFree(dpr);
}

} // namespace gpu_ecm
