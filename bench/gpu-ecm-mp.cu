/*
 * gpu-ecm-mp.cu — multi-precision (K-limb) ECM stage-1 on the GPU, the math
 * foundation of the v3.1.0 GPU pre-NFS factoring front-end (Track 2.1).
 *
 * The existing GPU ECM (sieve/ecm/gpu_ecm.cu, bench/gpu-ecm*.cu) handles only
 * moduli < 2^126 — fine for sieve cofactors, useless for stripping a 25-30 digit
 * factor from a real NFS-sized N (>85 digits), because ECM runs *modulo N*. This
 * generalizes the validated 2-limb CIOS Montgomery (bench/gpu-mont128.cu) and the
 * Montgomery-curve XZ ladder (bench/gpu-ecm.cu) to K 64-bit limbs, so ECM can run
 * modulo a multi-hundred-bit N. Same __host__ __device__ code runs on CPU and
 * GPU, so the device math is validated bit-exact.
 *
 * Requires (as the 2-limb code did) ~2 bits of headroom: n < 2^(64K-2). Pick K =
 * ceil((bits+2)/64). Here K in {2,4,8} -> up to 510-bit (~153 digit) moduli.
 *
 *   nvcc -arch=sm_86 -O3 bench/gpu-ecm-mp.cu -o /tmp/gpu-ecm-mp && /tmp/gpu-ecm-mp
 *
 * Two checks per width: (1) montmulK bit-exact vs an independent binary mulmod
 * reference over random moduli; (2) ECM stage-1 cracks crafted composites
 * n = p*q (p findable at B1), with the GPU lanes validated bit-exact vs CPU.
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>

typedef uint64_t u64;
typedef unsigned __int128 u128;
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

/* ---------- K-limb little-endian unsigned arithmetic (all HD) ---------- */

template<int K> HD void mp_set0(u64 *r){ for(int i=0;i<K;i++) r[i]=0; }
template<int K> HD void mp_copy(u64 *r,const u64 *a){ for(int i=0;i<K;i++) r[i]=a[i]; }
template<int K> HD bool mp_geq(const u64 *a,const u64 *b){      /* a >= b */
    for(int i=K-1;i>=0;i--){ if(a[i]!=b[i]) return a[i]>b[i]; }
    return true;
}
template<int K> HD bool mp_is0(const u64 *a){ for(int i=0;i<K;i++) if(a[i]) return false; return true; }

/* r = a - b (assumes a >= b), returns nothing */
template<int K> HD void mp_sub(u64 *r,const u64 *a,const u64 *b){
    u128 br=0;
    for(int i=0;i<K;i++){ u128 x=(u128)a[i]-(u128)b[i]-br; r[i]=(u64)x; br=(x>>64)&1; }
}
/* r = a + b mod n  (a,b < n) */
template<int K> HD void addmod(u64 *r,const u64 *a,const u64 *b,const u64 *n){
    u64 t[K]; u128 c=0;
    for(int i=0;i<K;i++){ u128 x=(u128)a[i]+(u128)b[i]+c; t[i]=(u64)x; c=x>>64; }
    if(c || mp_geq<K>(t,n)) mp_sub<K>(r,t,n); else mp_copy<K>(r,t);
}
/* r = a - b mod n  (a,b < n) */
template<int K> HD void submod(u64 *r,const u64 *a,const u64 *b,const u64 *n){
    u64 t[K]; u128 br=0;
    for(int i=0;i<K;i++){ u128 x=(u128)a[i]-(u128)b[i]-br; t[i]=(u64)x; br=(x>>64)&1; }
    if(br){ u128 c=0; for(int i=0;i<K;i++){ u128 x=(u128)t[i]+(u128)n[i]+c; r[i]=(u64)x; c=x>>64; } }
    else mp_copy<K>(r,t);
}
/* r = a*b*R^{-1} mod n, R=2^{64K}. CIOS; requires n odd, n < 2^{64K-2}, a,b<n.
 * np = -n^{-1} mod 2^64 (low limb). Generalizes the validated 2-limb montmul. */
template<int K> HD void montmul(u64 *r,const u64 *A,const u64 *B,const u64 *N,u64 np){
    u64 t[K+2]; for(int i=0;i<K+2;i++) t[i]=0;
    for(int i=0;i<K;i++){
        /* t += A * B[i] */
        u128 c=0;
        for(int j=0;j<K;j++){ u128 x=(u128)t[j]+(u128)A[j]*B[i]+c; t[j]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[K]+c; t[K]=(u64)x; t[K+1]=(u64)(x>>64); }
        /* m = t[0]*np; t += m*N; t >>= one limb */
        u64 m=t[0]*np;
        { u128 x=(u128)t[0]+(u128)m*N[0]; c=x>>64; }          /* low limb cancels */
        for(int j=1;j<K;j++){ u128 x=(u128)t[j]+(u128)m*N[j]+c; t[j-1]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[K]+c; t[K-1]=(u64)x; c=x>>64; }
        t[K]=t[K+1]+(u64)c; t[K+1]=0;
    }
    /* result in t[0..K-1] (t[K]==0 given the 2-bit headroom); one conditional sub */
    if(mp_geq<K>(t,N)) mp_sub<K>(r,t,N); else mp_copy<K>(r,t);
}

/* ---------- Montgomery-curve XZ ops over K-limb field (Montgomery form) ---------- */
template<int K> struct PT { u64 X[K], Z[K]; };

template<int K> HD void cdbl(PT<K>&r,const PT<K>&p,const u64*a24,const u64*n,u64 np){
    u64 apz[K],amz[K],A[K],B[K],C[K],t[K];
    addmod<K>(apz,p.X,p.Z,n); montmul<K>(A,apz,apz,n,np);     /* (X+Z)^2 */
    submod<K>(amz,p.X,p.Z,n); montmul<K>(B,amz,amz,n,np);     /* (X-Z)^2 */
    submod<K>(C,A,B,n);                                       /* 4XZ */
    montmul<K>(r.X,A,B,n,np);
    montmul<K>(t,a24,C,n,np); addmod<K>(t,B,t,n); montmul<K>(r.Z,C,t,n,np);
}
template<int K> HD void cadd(PT<K>&r,const PT<K>&p1,const PT<K>&p2,const PT<K>&pd,
                             const u64*n,u64 np){
    u64 p1pz[K],p1mz[K],p2pz[K],p2mz[K],DA[K],CB[K],s[K],d[K],ss[K],dd[K];
    submod<K>(p1mz,p1.X,p1.Z,n); addmod<K>(p2pz,p2.X,p2.Z,n);
    addmod<K>(p1pz,p1.X,p1.Z,n); submod<K>(p2mz,p2.X,p2.Z,n);
    montmul<K>(DA,p1mz,p2pz,n,np); montmul<K>(CB,p1pz,p2mz,n,np);
    addmod<K>(s,DA,CB,n); submod<K>(d,DA,CB,n);
    montmul<K>(ss,s,s,n,np); montmul<K>(dd,d,d,n,np);
    montmul<K>(r.X,pd.Z,ss,n,np); montmul<K>(r.Z,pd.X,dd,n,np);
}
/* [k]P, k>=1, k a 64-bit scalar (prime power) */
template<int K> HD void ladder(PT<K>&out,const PT<K>&P,u64 k,const u64*a24,
                               const u64*n,u64 np){
    if(k==1){ out=P; return; }
    PT<K> R0=P, R1; cdbl<K>(R1,P,a24,n,np);
    int b=63; while(!((k>>b)&1)) b--;
    for(b--;b>=0;b--){
        if((k>>b)&1){ PT<K> t; cadd<K>(t,R0,R1,P,n,np); R0=t; cdbl<K>(R1,R1,a24,n,np); }
        else        { PT<K> t; cadd<K>(t,R1,R0,P,n,np); R1=t; cdbl<K>(R0,R0,a24,n,np); }
    }
    out=R0;
}
/* ECM stage 1: returns leave-Montgomery Z of Q=[prod s]P (host then gcds with n).
 * R1=R mod n ("1"), R2=R^2 mod n, a24 = small seed (plain, < n). */
template<int K> HD void ecm_stage1(u64 *zout,const u64 *n,u64 np,
                                   const u64 *R1,const u64 *R2,
                                   const u64 *a24,const u64 *s,int ns){
    PT<K> P;
    addmod<K>(P.X,R1,R1,n);          /* x0 = 2  -> 2 in Montgomery form */
    mp_copy<K>(P.Z,R1);              /* z0 = 1  -> R mod n */
    u64 a24m[K]; montmul<K>(a24m,a24,R2,n,np);   /* a24 -> Montgomery */
    for(int i=0;i<ns;i++){ PT<K> t; ladder<K>(t,P,s[i],a24m,n,np); P=t; }
    u64 one[K]; mp_set0<K>(one); one[0]=1;
    montmul<K>(zout,P.Z,one,n,np);   /* leave Montgomery -> Z mod n */
}

/* ---------- independent host reference: binary mulmod (no Montgomery) ---------- */
template<int K> static void ref_mulmod(u64 *r,const u64 *a0,const u64 *b,const u64 *n){
    u64 a[K]; mp_copy<K>(a,a0); u64 acc[K]; mp_set0<K>(acc);
    for(int bit=0; bit<64*K; bit++){
        if((b[bit>>6]>>(bit&63))&1) addmod<K>(acc,acc,a,n);
        addmod<K>(a,a,a,n);
    }
    mp_copy<K>(r,acc);
}
/* R2 = 2^{128K} mod n, by doubling 1 a total of 128K times */
template<int K> static void compute_R2(u64 *R2,const u64 *n){
    u64 r[K]; mp_set0<K>(r); r[0]=1; if(mp_geq<K>(r,n)) mp_sub<K>(r,r,n);
    for(int i=0;i<128*K;i++) addmod<K>(r,r,r,n);
    mp_copy<K>(R2,r);
}
/* R1 = R mod n = 2^{64K} mod n */
template<int K> static void compute_R1(u64 *R1,const u64 *n){
    u64 r[K]; mp_set0<K>(r); r[0]=1; if(mp_geq<K>(r,n)) mp_sub<K>(r,r,n);
    for(int i=0;i<64*K;i++) addmod<K>(r,r,r,n);
    mp_copy<K>(R1,r);
}
static u64 ninv64(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }

/* host K-limb binary gcd (for the factor-finding check) */
template<int K> static void mp_rshift1(u64 *a){
    for(int i=0;i<K-1;i++) a[i]=(a[i]>>1)|(a[i+1]<<63);
    a[K-1]>>=1;
}
template<int K> static void host_gcd(u64 *g,const u64 *a0,const u64 *b0){
    u64 a[K],b[K]; mp_copy<K>(a,a0); mp_copy<K>(b,b0);
    if(mp_is0<K>(a)){ mp_copy<K>(g,b); return; }
    if(mp_is0<K>(b)){ mp_copy<K>(g,a); return; }
    while(!(a[0]&1)) mp_rshift1<K>(a);
    while(!mp_is0<K>(b)){
        while(!(b[0]&1)) mp_rshift1<K>(b);
        if(mp_geq<K>(a,b)){ u64 t[K]; mp_copy<K>(t,a); mp_copy<K>(a,b); mp_copy<K>(b,t); }
        mp_sub<K>(b,b,a);
    }
    mp_copy<K>(g,a);
}

static u64 rnd(u64 *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
static bool isp64(u64 n){ if(n<2)return false; for(u64 p=2;p*p<=n;p++) if(n%p==0) return false; return true; }
static u64 randprime(u64 lo,u64 hi,u64 *st){ for(;;){ u64 c=lo+rnd(st)%(hi-lo); c|=1; if(isp64(c)) return c; } }

/* ---------- GPU kernels (instantiated per width) ---------- */
template<int K> __global__ void mmK_kernel(const u64*A,const u64*B,const u64*N,
                                           const u64*NP,const u64*R2,u64*out,int lanes){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=lanes) return;
    const u64*a=A+i*K,*b=B+i*K,*n=N+i*K,*r2=R2+i*K; u64 np=NP[i];
    u64 am[K],bm[K],pm[K],mo[K];
    montmul<K>(am,a,r2,n,np); montmul<K>(bm,b,r2,n,np);
    montmul<K>(pm,am,bm,n,np);
    u64 one[K]; mp_set0<K>(one); one[0]=1; montmul<K>(mo,pm,one,n,np);
    mp_copy<K>(out+i*K,mo);
}
template<int K> __global__ void ecmK_kernel(const u64*N,const u64*NP,const u64*R1,
                const u64*R2,const u64*SEED,const u64*s,int ns,u64*Z,int lanes){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=lanes) return;
    ecm_stage1<K>(Z+i*K,N+i*K,NP[i],R1+i*K,R2+i*K,SEED+i*K,s,ns);
}

/* ---------- per-width test driver ---------- */
template<int K>
static int test_width(const char* label, const std::vector<u64>& sP, int ns, u64 B1)
{
    int fails=0;
    /* ---- (1) montmul bit-exact vs binary mulmod, random moduli ---- */
    {
        const int N=20000; u64 st=0x1234567ULL+K;
        std::vector<u64> A(N*K),B(N*K),Nm(N*K),R2(N*K),NP(N);
        for(int i=0;i<N;i++){
            u64 *n=&Nm[i*K]; for(int j=0;j<K;j++) n[j]=rnd(&st);
            n[K-1]>>=2; n[0]|=1;                 /* odd, < 2^{64K-2} */
            if(K==1 && n[0]<3) n[0]=3;
            NP[i]=ninv64(n[0]); compute_R2<K>(&R2[i*K],n);
            u64 *a=&A[i*K],*b=&B[i*K];
            for(int j=0;j<K;j++){ a[j]=rnd(&st); b[j]=rnd(&st); }
            /* reduce a,b mod n by conditional subtract a few times (cheap) */
            for(int t=0;t<3;t++){ if(mp_geq<K>(a,n)) mp_sub<K>(a,a,n); if(mp_geq<K>(b,n)) mp_sub<K>(b,b,n); }
            if(mp_geq<K>(a,n)) mp_set0<K>(a); if(mp_geq<K>(b,n)) mp_set0<K>(b);
        }
        u64 *dA,*dB,*dN,*dNP,*dR2,*dO;
        cudaMalloc(&dA,N*K*8);cudaMalloc(&dB,N*K*8);cudaMalloc(&dN,N*K*8);
        cudaMalloc(&dNP,N*8);cudaMalloc(&dR2,N*K*8);cudaMalloc(&dO,N*K*8);
        cudaMemcpy(dA,A.data(),N*K*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dB,B.data(),N*K*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dN,Nm.data(),N*K*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dNP,NP.data(),N*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dR2,R2.data(),N*K*8,cudaMemcpyHostToDevice);
        mmK_kernel<K><<<(N+127)/128,128>>>(dA,dB,dN,dNP,dR2,dO,N);
        cudaDeviceSynchronize();
        cudaError_t e=cudaGetLastError();
        std::vector<u64> O(N*K); cudaMemcpy(O.data(),dO,N*K*8,cudaMemcpyDeviceToHost);
        long mis=0;
        for(int i=0;i<N;i++){ u64 ref[K]; ref_mulmod<K>(ref,&A[i*K],&B[i*K],&Nm[i*K]);
            for(int j=0;j<K;j++) if(ref[j]!=O[i*K+j]){ mis++; break; } }
        printf("  [%s] montmul%d : %s (%ld/%d wrong vs binary mulmod)%s\n", label,64*K,
               mis==0?"PASS":"FAIL", mis, N, e?" CUDAERR":"");
        if(mis||e) fails++;
        cudaFree(dA);cudaFree(dB);cudaFree(dN);cudaFree(dNP);cudaFree(dR2);cudaFree(dO);
    }
    /* ---- (2) ECM cracks composites n=p*q; GPU validated vs CPU ---- */
    {
        const int NCOMP=128, CURVES=64, L=NCOMP*CURVES; u64 st=0xC0FFEEULL+K*7;
        std::vector<u64> N(L*K),NP(L),R1(L*K),R2(L*K),SEED(L*K);
        std::vector<u64> pf(NCOMP*K);
        for(int c=0;c<NCOMP;c++){
            /* n = p * q with p ~20-bit (findable at B1=2000) and q filling the
             * rest of the K-limb width. Build q < 2^{64K-24} (odd), then the
             * single-word-p * K-limb-q product is < 2^{64K-3}, fits without
             * truncation (so p is a genuine factor) and keeps the 2-bit
             * Montgomery headroom. */
            u64 p=randprime(1u<<19,1u<<20,&st);
            u64 q[K]; for(int j=0;j<K;j++) q[j]=rnd(&st);
            q[K-1] >>= 24;                                      /* leave room for *p + headroom */
            q[0] |= 1;                                          /* odd */
            u64 nlimb[K]; u128 carry=0;
            for(int j=0;j<K;j++){ u128 x=(u128)q[j]*p+carry; nlimb[j]=(u64)x; carry=x>>64; }
            /* carry == 0 by construction; n is odd (p,q odd) */
            u64 *pfp=&pf[c*K]; mp_set0<K>(pfp); pfp[0]=p;
            u64 r1[K],r2[K]; compute_R1<K>(r1,nlimb); compute_R2<K>(r2,nlimb); u64 npv=ninv64(nlimb[0]);
            for(int j=0;j<CURVES;j++){ int i=c*CURVES+j;
                mp_copy<K>(&N[i*K],nlimb); NP[i]=npv; mp_copy<K>(&R1[i*K],r1); mp_copy<K>(&R2[i*K],r2);
                u64 *sd=&SEED[i*K]; mp_set0<K>(sd); sd[0]=(rnd(&st)%1000000)|2;  /* small a24 seed */
            }
        }
        u64 *dN,*dNP,*dR1,*dR2,*dSEED,*ds,*dZ;
        cudaMalloc(&dN,L*K*8);cudaMalloc(&dNP,L*8);cudaMalloc(&dR1,L*K*8);
        cudaMalloc(&dR2,L*K*8);cudaMalloc(&dSEED,L*K*8);cudaMalloc(&dZ,L*K*8);cudaMalloc(&ds,ns*8);
        cudaMemcpy(dN,N.data(),L*K*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dNP,NP.data(),L*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dR1,R1.data(),L*K*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dR2,R2.data(),L*K*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dSEED,SEED.data(),L*K*8,cudaMemcpyHostToDevice);
        cudaMemcpy(ds,sP.data(),ns*8,cudaMemcpyHostToDevice);
        auto t0=std::chrono::steady_clock::now();
        ecmK_kernel<K><<<(L+63)/64,64>>>(dN,dNP,dR1,dR2,dSEED,ds,ns,dZ,L);
        cudaDeviceSynchronize();
        auto t1=std::chrono::steady_clock::now();
        cudaError_t e=cudaGetLastError();
        std::vector<u64> Z(L*K); cudaMemcpy(Z.data(),dZ,L*K*8,cudaMemcpyDeviceToHost);
        double sec=std::chrono::duration<double>(t1-t0).count();
        /* CPU validation on a subset */
        int CPU=L/16; long mis=0;
        for(int i=0;i<CPU;i++){ u64 z[K]; ecm_stage1<K>(z,&N[i*K],NP[i],&R1[i*K],&R2[i*K],&SEED[i*K],sP.data(),ns);
            for(int j=0;j<K;j++) if(z[j]!=Z[i*K+j]){ mis++; break; } }
        /* factor-finding: host gcd(Z,n) per lane */
        int cracked=0;
        for(int c=0;c<NCOMP;c++){ bool ok=false;
            for(int j=0;j<CURVES;j++){ int i=c*CURVES+j; u64 g[K]; host_gcd<K>(g,&Z[i*K],&N[i*K]);
                bool one=(g[0]==1); for(int t=1;t<K;t++) if(g[t]) one=false; one=one&&(g[0]==1);
                if(!one && !mp_geq<K>(g,&N[i*K])){ /* 1 < g < n */
                    bool eqp=true; for(int t=0;t<K;t++) if(g[t]!=pf[c*K+t]) eqp=false;
                    if(eqp) ok=true; }
            }
            cracked+=ok;
        }
        printf("  [%s] ecm%d    : validation %s (%ld/%d lanes vs CPU); cracked %d/%d; %.0f curves/s%s\n",
               label,64*K, mis==0?"PASS":"FAIL", mis, CPU, cracked, NCOMP, L/sec, e?" CUDAERR":"");
        if(mis||e||cracked==0) fails++;
        cudaFree(dN);cudaFree(dNP);cudaFree(dR1);cudaFree(dR2);cudaFree(dSEED);cudaFree(dZ);cudaFree(ds);
    }
    return fails;
}

int main(){
    const u64 B1=2000;
    std::vector<u64> s; std::vector<char> comp(B1+1,0);
    for(u64 p=2;p<=B1;p++) if(!comp[p]){ for(u64 q=p*p;q<=B1;q+=p) comp[q]=1;
        u64 pe=p; while(pe*p<=B1) pe*=p; s.push_back(pe); }
    int ns=(int)s.size();
    printf("multi-precision GPU ECM (K-limb) — B1=%llu, %d prime-power multipliers\n",
           (unsigned long long)B1, ns);
    int fails=0;
    fails += test_width<2>("128-bit ", s, ns, B1);
    fails += test_width<4>("256-bit ", s, ns, B1);
    fails += test_width<8>("512-bit ", s, ns, B1);
    printf("%s\n", fails==0 ? "ALL PASS" : "FAILURES");
    return fails!=0;
}
