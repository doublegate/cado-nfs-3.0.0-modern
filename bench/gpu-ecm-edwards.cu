/*
 * gpu-ecm-edwards.cu — does mixed-representation (twisted-Edwards) ECM stage-1
 * actually beat the Montgomery XZ ladder *on the GPU*?  (Roadmap A2.)
 *
 * Background.  The CPU `facul` ECM in CADO already uses mixed representations —
 * the "mishmash" bytecode (sieve/ecm/bytecode_mishmash_B1_data.h, the
 * Bouvier-Imbert 2020 scheme: twisted-Edwards stage-1 with a final switch to
 * Montgomery, "-4 M").  The fork's GPU ECM (bench/gpu-ecm-mp.cu,
 * sieve/ecm/gpu_ecm.cu) does NOT — it is a pure Montgomery XZ ladder (1 dbl +
 * 1 dadd per scalar bit = 11 modmuls/bit).  A2 asks whether porting the Edwards
 * mixed-rep approach to the GPU is a win.  It is genuinely uncertain: Edwards
 * cuts modmuls via doubling+addition chains (windowing/NAF), but extended
 * coordinates need 4 field elements/point vs the ladder's 2, and a per-thread
 * wNAF precompute table costs local memory — and GPU ECM occupancy is already
 * register/limb bound (BENCHMARKS.md s3).  So we MEASURE rather than assume.
 *
 * What this does, per modulus width K in {2,4,8} (128/256/512-bit):
 *  (1) Correctness, bit-exact, NO square roots: jointly pick a random Edwards
 *      point (u0,v0) on an a=-1 twisted-Edwards curve (choose u0,v0 -> derive d),
 *      map it to the birationally-equivalent Montgomery curve (A, x0).  Run the
 *      (already-validated) Montgomery XZ ladder for [s]P giving x=(Xm:Zm); run
 *      the Edwards a=-1 chain for [s]P giving (Ue:Ve:We); the Montgomery x of the
 *      Edwards result is (We+Ve : We-Ve).  Assert  Xm*(We-Ve) == Zm*(We+Ve) (mod n),
 *      i.e. identical x([s]P).  Checked host AND device (same __host__ __device__
 *      code) for both the double-and-add and the wNAF Edwards scalar mults.
 *  (2) Throughput: curves/s for the ladder vs Edwards double-and-add vs Edwards
 *      wNAF(w=4), same B1, same scalar s, same batch — the honest verdict.
 *
 *   nvcc -arch=sm_86 -O3 bench/gpu-ecm-edwards.cu -o gpu-ecm-edwards && ./gpu-ecm-edwards
 *
 * Field arithmetic (K-limb CIOS Montgomery) is the same code as gpu-ecm-mp.cu,
 * copied here so the bench is standalone (matching the existing bench pattern).
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
template<int K> HD bool mp_geq(const u64 *a,const u64 *b){
    for(int i=K-1;i>=0;i--){ if(a[i]!=b[i]) return a[i]>b[i]; } return true; }
template<int K> HD bool mp_is0(const u64 *a){ for(int i=0;i<K;i++) if(a[i]) return false; return true; }
template<int K> HD bool mp_is1(const u64 *a){ if(a[0]!=1) return false; for(int i=1;i<K;i++) if(a[i]) return false; return true; }
template<int K> HD void mp_sub(u64 *r,const u64 *a,const u64 *b){
    u128 br=0; for(int i=0;i<K;i++){ u128 x=(u128)a[i]-(u128)b[i]-br; r[i]=(u64)x; br=(x>>64)&1; } }
template<int K> HD void addmod(u64 *r,const u64 *a,const u64 *b,const u64 *n){
    u64 t[K]; u128 c=0;
    for(int i=0;i<K;i++){ u128 x=(u128)a[i]+(u128)b[i]+c; t[i]=(u64)x; c=x>>64; }
    if(c || mp_geq<K>(t,n)) mp_sub<K>(r,t,n); else mp_copy<K>(r,t); }
template<int K> HD void submod(u64 *r,const u64 *a,const u64 *b,const u64 *n){
    u64 t[K]; u128 br=0;
    for(int i=0;i<K;i++){ u128 x=(u128)a[i]-(u128)b[i]-br; t[i]=(u64)x; br=(x>>64)&1; }
    if(br){ u128 c=0; for(int i=0;i<K;i++){ u128 x=(u128)t[i]+(u128)n[i]+c; r[i]=(u64)x; c=x>>64; } }
    else mp_copy<K>(r,t); }
template<int K> HD void negmod(u64 *r,const u64 *a,const u64 *n){    /* r = -a mod n */
    if(mp_is0<K>(a)) mp_set0<K>(r); else mp_sub<K>(r,n,a); }
template<int K> HD void montmul(u64 *r,const u64 *A,const u64 *B,const u64 *N,u64 np){
    u64 t[K+2]; for(int i=0;i<K+2;i++) t[i]=0;
    for(int i=0;i<K;i++){
        u128 c=0;
        for(int j=0;j<K;j++){ u128 x=(u128)t[j]+(u128)A[j]*B[i]+c; t[j]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[K]+c; t[K]=(u64)x; t[K+1]=(u64)(x>>64); }
        u64 m=t[0]*np;
        { u128 x=(u128)t[0]+(u128)m*N[0]; c=x>>64; }
        for(int j=1;j<K;j++){ u128 x=(u128)t[j]+(u128)m*N[j]+c; t[j-1]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[K]+c; t[K-1]=(u64)x; c=x>>64; }
        t[K]=t[K+1]+(u64)c; t[K+1]=0;
    }
    if(mp_geq<K>(t,N)) mp_sub<K>(r,t,N); else mp_copy<K>(r,t);
}

/* ---------- Montgomery-curve XZ ladder (baseline; from gpu-ecm-mp.cu) ---------- */
template<int K> struct PT { u64 X[K], Z[K]; };
template<int K> HD void cdbl(PT<K>&r,const PT<K>&p,const u64*a24,const u64*n,u64 np){
    u64 apz[K],amz[K],A[K],B[K],C[K],t[K];
    addmod<K>(apz,p.X,p.Z,n); montmul<K>(A,apz,apz,n,np);
    submod<K>(amz,p.X,p.Z,n); montmul<K>(B,amz,amz,n,np);
    submod<K>(C,A,B,n);
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
template<int K> HD void ladder1(PT<K>&out,const PT<K>&P,u64 k,const u64*a24,
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
/* ladder stage-1: P=(x0:1) plain -> Montgomery; [prod s]P; return x=(X:Z) plain */
template<int K> HD void mont_stage1(u64 *Xo,u64 *Zo,const u64 *n,u64 np,
        const u64 *R1,const u64 *R2,const u64 *x0,const u64 *a24,const u64 *s,int ns){
    PT<K> P; montmul<K>(P.X,x0,R2,n,np); mp_copy<K>(P.Z,R1);   /* (x0:1) -> Mont */
    u64 a24m[K]; montmul<K>(a24m,a24,R2,n,np);
    for(int i=0;i<ns;i++){ PT<K> t; ladder1<K>(t,P,s[i],a24m,n,np); P=t; }
    u64 one[K]; mp_set0<K>(one); one[0]=1;
    montmul<K>(Xo,P.X,one,n,np); montmul<K>(Zo,P.Z,one,n,np);  /* leave Mont */
}
/* single big-scalar XZ ladder: [s]P with s given MSB-first as a 0/1 bit array.
 * The fairest ladder baseline vs the Edwards single-scalar wNAF (same s). */
template<int K> HD void mont_stage1_big(u64 *Xo,u64 *Zo,const u64 *n,u64 np,
        const u64 *R1,const u64 *R2,const u64 *x0,const u64 *a24,
        const unsigned char*bits,int nb){
    PT<K> P; montmul<K>(P.X,x0,R2,n,np); mp_copy<K>(P.Z,R1);
    u64 a24m[K]; montmul<K>(a24m,a24,R2,n,np);
    int i=0; while(i<nb && !bits[i]) i++;        /* skip to leading 1 */
    PT<K> R0=P, R1pt; cdbl<K>(R1pt,P,a24m,n,np);  /* R0=[1]P, R1=[2]P (diff P) */
    for(i++; i<nb; i++){
        if(bits[i]){ PT<K> t; cadd<K>(t,R0,R1pt,P,n,np); R0=t; cdbl<K>(R1pt,R1pt,a24m,n,np); }
        else       { PT<K> t; cadd<K>(t,R0,R1pt,P,n,np); R1pt=t; cdbl<K>(R0,R0,a24m,n,np); }
    }
    u64 one[K]; mp_set0<K>(one); one[0]=1;
    montmul<K>(Xo,R0.X,one,n,np); montmul<K>(Zo,R0.Z,one,n,np);
}

/* ---------- twisted-Edwards a=-1, extended coords (X:Y:Z:T), T=XY/Z ---------- */
/* all field elements in Montgomery form; dm = curve d in Montgomery form. */
template<int K> struct EPT { u64 X[K], Y[K], Z[K], T[K]; };

/* doubling: 8 montmuls (a=-1 -> the a*X^2 term is just a negation). Needs no T1. */
template<int K> HD void edbl(EPT<K>&r,const EPT<K>&p,const u64*n,u64 np){
    u64 A[K],B[K],C[K],D[K],E[K],G[K],F[K],H[K],t[K];
    montmul<K>(A,p.X,p.X,n,np);                 /* A = X^2          */
    montmul<K>(B,p.Y,p.Y,n,np);                 /* B = Y^2          */
    montmul<K>(C,p.Z,p.Z,n,np); addmod<K>(C,C,C,n); /* C = 2 Z^2     */
    negmod<K>(D,A,n);                           /* D = a A = -A     */
    addmod<K>(t,p.X,p.Y,n); montmul<K>(E,t,t,n,np);
    submod<K>(E,E,A,n); submod<K>(E,E,B,n);     /* E = (X+Y)^2-A-B  */
    addmod<K>(G,D,B,n);                         /* G = D + B        */
    submod<K>(F,G,C,n);                         /* F = G - C        */
    submod<K>(H,D,B,n);                         /* H = D - B        */
    montmul<K>(r.X,E,F,n,np);
    montmul<K>(r.Y,G,H,n,np);
    montmul<K>(r.T,E,H,n,np);
    montmul<K>(r.Z,F,G,n,np);
}
/* unified addition: 9 montmuls (a=-1 -> H=B+A free). Needs T of both inputs. */
template<int K> HD void eadd(EPT<K>&r,const EPT<K>&p1,const EPT<K>&p2,
                             const u64*dm,const u64*n,u64 np){
    u64 A[K],B[K],C[K],D[K],E[K],F[K],G[K],H[K],t1[K],t2[K];
    montmul<K>(A,p1.X,p2.X,n,np);               /* A = X1 X2        */
    montmul<K>(B,p1.Y,p2.Y,n,np);               /* B = Y1 Y2        */
    montmul<K>(C,p1.T,p2.T,n,np); montmul<K>(C,C,dm,n,np); /* C = d T1 T2 */
    montmul<K>(D,p1.Z,p2.Z,n,np);               /* D = Z1 Z2        */
    addmod<K>(t1,p1.X,p1.Y,n); addmod<K>(t2,p2.X,p2.Y,n);
    montmul<K>(E,t1,t2,n,np); submod<K>(E,E,A,n); submod<K>(E,E,B,n); /* E=(X1+Y1)(X2+Y2)-A-B */
    submod<K>(F,D,C,n);                         /* F = D - C        */
    addmod<K>(G,D,C,n);                         /* G = D + C        */
    addmod<K>(H,B,A,n);                         /* H = B - a A = B+A*/
    montmul<K>(r.X,E,F,n,np);
    montmul<K>(r.Y,G,H,n,np);
    montmul<K>(r.T,E,H,n,np);
    montmul<K>(r.Z,F,G,n,np);
}
template<int K> HD void eneg(EPT<K>&r,const EPT<K>&p,const u64*n){  /* -(X:Y:Z:T)=(-X:Y:Z:-T) */
    negmod<K>(r.X,p.X,n); mp_copy<K>(r.Y,p.Y); mp_copy<K>(r.Z,p.Z); negmod<K>(r.T,p.T,n);
}
template<int K> HD void eident(EPT<K>&r,const u64*R1){              /* identity (0:1:1:0) Mont */
    mp_set0<K>(r.X); mp_copy<K>(r.Y,R1); mp_copy<K>(r.Z,R1); mp_set0<K>(r.T);
}

/* set up the Edwards start point in Montgomery form from plain (u0,v0): T0=u0 v0 */
template<int K> HD void esetup(EPT<K>&P,const u64*u0,const u64*v0,
                               const u64*R1,const u64*R2,const u64*n,u64 np){
    montmul<K>(P.X,u0,R2,n,np);  montmul<K>(P.Y,v0,R2,n,np);  mp_copy<K>(P.Z,R1);
    montmul<K>(P.T,P.X,P.Y,n,np);
}

/* Edwards [s]P, s = product of the prime powers, processed bitwise as ONE big
 * scalar 'bits' (len bits, MSB-first array of bytes 0/1). double-and-add. */
template<int K> HD void edw_da(EPT<K>&out,const EPT<K>&P,const u64*dm,
        const unsigned char*bits,int nb,const u64*R1,const u64*n,u64 np){
    EPT<K> R; eident<K>(R,R1);
    for(int i=0;i<nb;i++){
        EPT<K> t; edbl<K>(t,R,n,np); R=t;
        if(bits[i]){ EPT<K> u; eadd<K>(u,R,P,dm,n,np); R=u; }
    }
    out=R;
}
/* Edwards [s]P via wNAF(w): 'dig' is MSB-first, each entry in (-2^(w-1),2^(w-1))
 * odd or 0; tbl precomputed = [1]P,[3]P,...  (2^(w-2) entries). */
template<int K> HD void edw_wnaf(EPT<K>&out,const EPT<K>*tbl,int /*ntbl*/,
        const signed char*dig,int nd,const u64*dm,const u64*R1,const u64*n,u64 np){
    EPT<K> R; eident<K>(R,R1);
    for(int i=0;i<nd;i++){
        EPT<K> t; edbl<K>(t,R,n,np); R=t;
        int d=dig[i];
        if(d){
            int idx=(d>0? d:-d); idx=(idx-1)>>1;
            EPT<K> A=tbl[idx];
            if(d<0){ EPT<K> m; eneg<K>(m,A,n); A=m; }
            EPT<K> u; eadd<K>(u,R,A,dm,n,np); R=u;
        }
    }
    out=R;
}
template<int K> HD void edw_build_table(EPT<K>*tbl,int ntbl,const EPT<K>&P,
                                        const u64*dm,const u64*n,u64 np){
    tbl[0]=P;                                   /* [1]P */
    if(ntbl>1){ EPT<K> P2; edbl<K>(P2,P,n,np);  /* [2]P */
        for(int j=1;j<ntbl;j++){ EPT<K> t; eadd<K>(t,tbl[j-1],P2,dm,n,np); tbl[j]=t; } }
}

/* ---------- host helpers (setup math): K-limb modinv for odd n ---------- */
static u64 ninv64(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }
template<int K> static void mp_rshift1(u64 *a){
    for(int i=0;i<K-1;i++) a[i]=(a[i]>>1)|(a[i+1]<<63); a[K-1]>>=1; }
/* a = (a+ (odd?n:0))/2 , preserving value mod n when halving an odd representative */
template<int K> static void mp_half_mod(u64 *a,const u64 *n){
    if(a[0]&1){ u128 c=0; u64 hi=0;
        for(int i=0;i<K;i++){ u128 x=(u128)a[i]+(u128)n[i]+c; a[i]=(u64)x; c=x>>64; } hi=(u64)c;
        mp_rshift1<K>(a); a[K-1]|=(hi<<63);
    } else mp_rshift1<K>(a);
}
template<int K> static bool mp_eq(const u64*a,const u64*b){ for(int i=0;i<K;i++) if(a[i]!=b[i]) return false; return true; }
/* r = a^{-1} mod n, n odd, 0<a<n. Binary extended GCD; returns false (and leaves
 * r untouched) if gcd(a,n)!=1 — caller resamples the lane. Bounded so a random
 * composite modulus sharing a factor can never spin forever. */
template<int K> static bool mp_modinv(u64*r,const u64*a0,const u64*n){
    u64 u[K],v[K],x1[K],x2[K];
    mp_copy<K>(u,a0); mp_copy<K>(v,n);
    mp_set0<K>(x1); x1[0]=1; mp_set0<K>(x2);
    int guard=0, cap=600*K;
    while(!mp_is1<K>(u) && !mp_is1<K>(v)){
        if(mp_is0<K>(u) || mp_is0<K>(v) || ++guard>cap) return false;  /* gcd!=1 */
        while(!(u[0]&1)){ mp_rshift1<K>(u); mp_half_mod<K>(x1,n); if(mp_is0<K>(u)) return false; }
        while(!(v[0]&1)){ mp_rshift1<K>(v); mp_half_mod<K>(x2,n); if(mp_is0<K>(v)) return false; }
        if(mp_geq<K>(u,v)){ mp_sub<K>(u,u,v); submod<K>(x1,x1,x2,n); }
        else              { mp_sub<K>(v,v,u); submod<K>(x2,x2,x1,n); }
    }
    if(mp_is1<K>(u)) mp_copy<K>(r,x1); else mp_copy<K>(r,x2);
    return true;
}
/* r = a mod n for an arbitrary K-limb a (n need not be near 2^{64K}); binary. */
template<int K> static void mp_mod(u64*r,const u64*a,const u64*n){
    u64 acc[K],one[K]; mp_set0<K>(acc); mp_set0<K>(one); one[0]=1;
    for(int bit=64*K-1;bit>=0;bit--){
        addmod<K>(acc,acc,acc,n);                       /* acc = 2*acc mod n */
        if((a[bit>>6]>>(bit&63))&1) addmod<K>(acc,acc,one,n);
    }
    mp_copy<K>(r,acc);
}
template<int K> static void mul_plain(u64*r,const u64*a,const u64*b,const u64*n){
    /* plain modmul via binary (a,b<n); small use in setup only */
    u64 aa[K]; mp_copy<K>(aa,a); u64 acc[K]; mp_set0<K>(acc);
    for(int bit=0;bit<64*K;bit++){ if((b[bit>>6]>>(bit&63))&1) addmod<K>(acc,acc,aa,n); addmod<K>(aa,aa,aa,n); }
    mp_copy<K>(r,acc);
}
template<int K> static void compute_R1(u64 *R1,const u64 *n){
    u64 r[K]; mp_set0<K>(r); r[0]=1; if(mp_geq<K>(r,n)) mp_sub<K>(r,r,n);
    for(int i=0;i<64*K;i++) addmod<K>(r,r,r,n); mp_copy<K>(R1,r); }
template<int K> static void compute_R2(u64 *R2,const u64 *n){
    u64 r[K]; mp_set0<K>(r); r[0]=1; if(mp_geq<K>(r,n)) mp_sub<K>(r,r,n);
    for(int i=0;i<128*K;i++) addmod<K>(r,r,r,n); mp_copy<K>(R2,r); }
static u64 rnd(u64 *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

/* ---------- host big integer for s = prod(prime powers) and wNAF ---------- */
struct Big { std::vector<uint32_t> d; };        /* little-endian base 2^32 */
static void big_mul_small(Big&a,uint32_t m){
    uint64_t c=0; for(size_t i=0;i<a.d.size();i++){ uint64_t x=(uint64_t)a.d[i]*m+c; a.d[i]=(uint32_t)x; c=x>>32; }
    if(c) a.d.push_back((uint32_t)c);
}
static bool big_is0(const Big&a){ for(uint32_t w:a.d) if(w) return false; return true; }
static int big_bitlen(const Big&a){ int n=(int)a.d.size()*32; for(int i=(int)a.d.size()-1;i>=0;i--){ if(a.d[i]){ for(int b=31;b>=0;b--) if((a.d[i]>>b)&1) return i*32+b+1; } } return 0; }
static int big_getbit(const Big&a,int i){ if(i<0||(size_t)(i>>5)>=a.d.size()) return 0; return (a.d[i>>5]>>(i&31))&1; }
static void big_rshift1(Big&a){ uint32_t carry=0; for(int i=(int)a.d.size()-1;i>=0;i--){ uint32_t nc=a.d[i]&1; a.d[i]=(a.d[i]>>1)|(carry<<31); carry=nc; } }
static void big_sub_small(Big&a,int m){ /* a -= m, m may be negative; assumes result>=0 */
    if(m==0) return; if(m>0){ uint64_t borrow=m; for(size_t i=0;i<a.d.size()&&borrow;i++){ uint64_t cur=a.d[i]; if(cur>=borrow){ a.d[i]=(uint32_t)(cur-borrow); borrow=0; } else { a.d[i]=(uint32_t)(cur+ (1ull<<32) - borrow); borrow=1; } } }
    else { uint64_t add=-m,c=0; for(size_t i=0;i<a.d.size();i++){ uint64_t x=(uint64_t)a.d[i]+ (i==0?add:0)+c; a.d[i]=(uint32_t)x; c=x>>32; } if(c) a.d.push_back((uint32_t)c); }
}
/* MSB-first plain bit array of s (for double-and-add) */
static std::vector<unsigned char> big_bits_msb(const Big&s){
    int nb=big_bitlen(s); std::vector<unsigned char> b(nb);
    for(int i=0;i<nb;i++) b[i]=(unsigned char)big_getbit(s,nb-1-i); return b;
}
/* MSB-first wNAF(w) digit array of s */
static std::vector<signed char> big_wnaf_msb(Big s,int w){
    std::vector<signed char> lo;                /* LSB-first first */
    int mask=(1<<w)-1, half=1<<(w-1);
    while(!big_is0(s)){
        if(s.d[0]&1){ int d=s.d[0]&mask; if(d>=half) d-=(1<<w); lo.push_back((signed char)d); big_sub_small(s,d); }
        else lo.push_back(0);
        big_rshift1(s);
    }
    std::vector<signed char> hi(lo.rbegin(),lo.rend());   /* MSB-first */
    return hi;
}
/* ============================ per-width driver ============================ */
template<int K> struct LaneSetup {
    /* per-lane (per random modulus) all the plain-form constants */
    std::vector<u64> N,NP,R1,R2;                 /* n, np, R, R^2 */
    std::vector<u64> X0,A24;                      /* Montgomery ladder start x0, a24 (plain) */
    std::vector<u64> U0,V0,DM;                    /* Edwards start (plain), curve d (plain) */
    int L;
};

/* build L lanes: random n, random Edwards point (u0,v0,a=-1)->derive d-> map to
 * Montgomery (A->a24, x0).  NO sqrt anywhere. */
template<int K> static LaneSetup<K> build_lanes(int L,u64 seed,int D){
    /* compute D distinct (modulus,curve) setups, tiled across L GPU lanes — the
     * host setup (binary modinv) is the slow part, so D<<L keeps it cheap while
     * the GPU batch stays large enough to saturate. */
    LaneSetup<K> S; S.L=L; if(D>L) D=L;
    S.N.resize(L*K);S.NP.resize(L);S.R1.resize(L*K);S.R2.resize(L*K);
    S.X0.resize(L*K);S.A24.resize(L*K);S.U0.resize(L*K);S.V0.resize(L*K);S.DM.resize(L*K);
    u64 st=seed;
    u64 two[K]; mp_set0<K>(two); two[0]=2;
    u64 four[K]; mp_set0<K>(four); four[0]=4;
    for(int i=0;i<L;i++){
        if(i>=D){    /* tile: copy distinct lane (i%D) */
            int s=i%D;
            mp_copy<K>(&S.N[i*K],&S.N[s*K]); S.NP[i]=S.NP[s];
            mp_copy<K>(&S.R1[i*K],&S.R1[s*K]); mp_copy<K>(&S.R2[i*K],&S.R2[s*K]);
            mp_copy<K>(&S.X0[i*K],&S.X0[s*K]); mp_copy<K>(&S.A24[i*K],&S.A24[s*K]);
            mp_copy<K>(&S.U0[i*K],&S.U0[s*K]); mp_copy<K>(&S.V0[i*K],&S.V0[s*K]);
            mp_copy<K>(&S.DM[i*K],&S.DM[s*K]); continue;
        }
      u64 one[K]; mp_set0<K>(one); one[0]=1;
      int tries=0;
      for(;;){    /* (re)generate n and resample (u0,v0) until setup inverses exist */
        u64 *n=&S.N[i*K];
        for(int j=0;j<K;j++) n[j]=rnd(&st); n[K-1]>>=2; n[0]|=1;     /* odd < 2^{64K-2} */
        if(K==1 && n[0]<5) n[0]=5;
        S.NP[i]=ninv64(n[0]); compute_R1<K>(&S.R1[i*K],n); compute_R2<K>(&S.R2[i*K],n);
        u64 inv4[K]; if(!mp_modinv<K>(inv4,four,n)) continue;        /* n odd -> always ok */
        bool good=false;
        for(int attempt=0; attempt<64; attempt++){  /* try a few (u0,v0) per n */
            if(++tries>100000){ fprintf(stderr,"build_lanes: lane %d stuck after %d tries\n",i,tries); }
            u64 u0[K],v0[K],rr[K];
            for(int j=0;j<K;j++) rr[j]=rnd(&st); mp_mod<K>(u0,rr,n);
            for(int j=0;j<K;j++) rr[j]=rnd(&st); mp_mod<K>(v0,rr,n);
            if(mp_is0<K>(u0)) u0[0]=3; if(mp_is0<K>(v0)) v0[0]=3;
            /* d = (v0^2 - u0^2 - 1)/(u0^2 v0^2)   (a=-1) */
            u64 uu[K],vv[K],num[K],den[K],dinv[K],dd[K];
            mul_plain<K>(uu,u0,u0,n); mul_plain<K>(vv,v0,v0,n);
            submod<K>(num,vv,uu,n); submod<K>(num,num,one,n);
            mul_plain<K>(den,uu,vv,n);
            if(!mp_modinv<K>(dinv,den,n)) continue;
            mul_plain<K>(dd,num,dinv,n);                           /* d (plain) */
            /* A = 2(1-d)/(1+d) ; a24=(A+2)/4 ; x0=(1+v0)/(1-v0) */
            u64 onepd[K],onemd[K],inv[K],A[K],t[K],a24[K],x0[K],onemv[K],onepv[K];
            addmod<K>(onepd,one,dd,n); submod<K>(onemd,one,dd,n);
            if(!mp_modinv<K>(inv,onepd,n)) continue;
            mul_plain<K>(t,onemd,inv,n); addmod<K>(A,t,t,n);
            addmod<K>(a24,A,two,n); mul_plain<K>(a24,a24,inv4,n);  /* (A+2)/4 */
            submod<K>(onemv,one,v0,n); addmod<K>(onepv,one,v0,n);
            if(!mp_modinv<K>(inv,onemv,n)) continue;
            mul_plain<K>(x0,onepv,inv,n);                          /* x0=(1+v0)/(1-v0) */
            mp_copy<K>(&S.DM[i*K],dd); mp_copy<K>(&S.U0[i*K],u0); mp_copy<K>(&S.V0[i*K],v0);
            mp_copy<K>(&S.A24[i*K],a24); mp_copy<K>(&S.X0[i*K],x0);
            good=true; break;
        }
        if(good) break;     /* else regenerate n and try again */
      }
    }
    return S;
}

/* kernels */
template<int K> __global__ void k_mont(const u64*N,const u64*NP,const u64*R1,const u64*R2,
        const u64*X0,const u64*A24,const unsigned char*bits,int nb,u64*Xo,u64*Zo,int L){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=L) return;
    mont_stage1_big<K>(Xo+i*K,Zo+i*K,N+i*K,NP[i],R1+i*K,R2+i*K,X0+i*K,A24+i*K,bits,nb);
}
template<int K> __global__ void k_edw_da(const u64*N,const u64*NP,const u64*R1,const u64*R2,
        const u64*U0,const u64*V0,const u64*DM,const unsigned char*bits,int nb,
        u64*Vo,u64*Wo,int L){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=L) return;
    const u64*n=N+i*K; u64 np=NP[i]; const u64*R1i=R1+i*K,*R2i=R2+i*K;
    u64 dm[K]; montmul<K>(dm,DM+i*K,R2i,n,np);
    EPT<K> P; esetup<K>(P,U0+i*K,V0+i*K,R1i,R2i,n,np);
    EPT<K> Q; edw_da<K>(Q,P,dm,bits,nb,R1i,n,np);
    u64 one[K]; mp_set0<K>(one); one[0]=1;
    montmul<K>(Vo+i*K,Q.Y,one,n,np); montmul<K>(Wo+i*K,Q.Z,one,n,np);  /* leave Mont: v=Y/Z */
}
template<int K> __global__ void k_edw_wnaf(const u64*N,const u64*NP,const u64*R1,const u64*R2,
        const u64*U0,const u64*V0,const u64*DM,const signed char*dig,int nd,int ntbl,
        u64*Vo,u64*Wo,int L){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=L) return;
    const u64*n=N+i*K; u64 np=NP[i]; const u64*R1i=R1+i*K,*R2i=R2+i*K;
    u64 dm[K]; montmul<K>(dm,DM+i*K,R2i,n,np);
    EPT<K> P; esetup<K>(P,U0+i*K,V0+i*K,R1i,R2i,n,np);
    EPT<K> tbl[8];                                   /* up to w=5 (2^3=8) */
    edw_build_table<K>(tbl,ntbl,P,dm,n,np);
    EPT<K> Q; edw_wnaf<K>(Q,tbl,ntbl,dig,nd,dm,R1i,n,np);
    u64 one[K]; mp_set0<K>(one); one[0]=1;
    montmul<K>(Vo+i*K,Q.Y,one,n,np); montmul<K>(Wo+i*K,Q.Z,one,n,np);
}

/* x([s]P) match: ladder x=(Xm:Zm) vs Edwards x=(W+V:W-V).  Xm*(W-V)==Zm*(W+V) */
template<int K> static bool xmatch(const u64*Xm,const u64*Zm,const u64*V,const u64*W,const u64*n){
    u64 WpV[K],WmV[K],l[K],r[K];
    addmod<K>(WpV,W,V,n); submod<K>(WmV,W,V,n);
    mul_plain<K>(l,Xm,WmV,n); mul_plain<K>(r,Zm,WpV,n);
    return mp_eq<K>(l,r);
}

template<int K>
static int run_width(const char*label,const std::vector<u64>&sP,int ns,
                     const std::vector<unsigned char>&bitsDA,
                     const std::vector<signed char>&wnaf,int w){
    int fails=0; (void)sP; (void)ns;
    const int L=8192, D=512;        /* 512 distinct setups tiled over 8192 GPU lanes */
    LaneSetup<K> S=build_lanes<K>(L,0xE0DA11ULL+K*131,D);
    int nb=(int)bitsDA.size(), nd=(int)wnaf.size(), ntbl=1<<(w-2);

    /* device buffers */
    u64 *dN,*dNP,*dR1,*dR2,*dX0,*dA24,*dU0,*dV0,*dDM,*dXm,*dZm,*dVda,*dWda,*dVw,*dWw;
    cudaMalloc(&dN,L*K*8);cudaMalloc(&dNP,L*8);cudaMalloc(&dR1,L*K*8);cudaMalloc(&dR2,L*K*8);
    cudaMalloc(&dX0,L*K*8);cudaMalloc(&dA24,L*K*8);cudaMalloc(&dU0,L*K*8);cudaMalloc(&dV0,L*K*8);
    cudaMalloc(&dDM,L*K*8);
    cudaMalloc(&dXm,L*K*8);cudaMalloc(&dZm,L*K*8);
    cudaMalloc(&dVda,L*K*8);cudaMalloc(&dWda,L*K*8);cudaMalloc(&dVw,L*K*8);cudaMalloc(&dWw,L*K*8);
    unsigned char*dbits; signed char*ddig;
    cudaMalloc(&dbits,nb); cudaMalloc(&ddig,nd);
    cudaMemcpy(dN,S.N.data(),L*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dNP,S.NP.data(),L*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR1,S.R1.data(),L*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR2,S.R2.data(),L*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dX0,S.X0.data(),L*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dA24,S.A24.data(),L*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dU0,S.U0.data(),L*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dV0,S.V0.data(),L*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dDM,S.DM.data(),L*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dbits,bitsDA.data(),nb,cudaMemcpyHostToDevice);
    cudaMemcpy(ddig,wnaf.data(),nd,cudaMemcpyHostToDevice);

    int TPB=64;
    /* --- Montgomery ladder (baseline): single big scalar s --- */
    auto t0=std::chrono::steady_clock::now();
    k_mont<K><<<(L+TPB-1)/TPB,TPB>>>(dN,dNP,dR1,dR2,dX0,dA24,dbits,nb,dXm,dZm,L);
    cudaDeviceSynchronize(); auto t1=std::chrono::steady_clock::now();
    cudaError_t e1=cudaGetLastError();
    double sMont=std::chrono::duration<double>(t1-t0).count();
    /* --- Edwards double-and-add --- */
    auto t2=std::chrono::steady_clock::now();
    k_edw_da<K><<<(L+TPB-1)/TPB,TPB>>>(dN,dNP,dR1,dR2,dU0,dV0,dDM,dbits,nb,dVda,dWda,L);
    cudaDeviceSynchronize(); auto t3=std::chrono::steady_clock::now();
    cudaError_t e2=cudaGetLastError();
    double sDA=std::chrono::duration<double>(t3-t2).count();
    /* --- Edwards wNAF --- */
    auto t4=std::chrono::steady_clock::now();
    k_edw_wnaf<K><<<(L+TPB-1)/TPB,TPB>>>(dN,dNP,dR1,dR2,dU0,dV0,dDM,ddig,nd,ntbl,dVw,dWw,L);
    cudaDeviceSynchronize(); auto t5=std::chrono::steady_clock::now();
    cudaError_t e3=cudaGetLastError();
    double sW=std::chrono::duration<double>(t5-t4).count();

    std::vector<u64> Xm(L*K),Zm(L*K),Vda(L*K),Wda(L*K),Vw(L*K),Ww(L*K);
    cudaMemcpy(Xm.data(),dXm,L*K*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(Zm.data(),dZm,L*K*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(Vda.data(),dVda,L*K*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(Wda.data(),dWda,L*K*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(Vw.data(),dVw,L*K*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(Ww.data(),dWw,L*K*8,cudaMemcpyDeviceToHost);

    /* correctness: x([s]P) of Edwards (DA & wNAF) == ladder, bit-exact via map */
    long mda=0,mw=0;
    for(int i=0;i<L;i++){
        if(!xmatch<K>(&Xm[i*K],&Zm[i*K],&Vda[i*K],&Wda[i*K],&S.N[i*K])) mda++;
        if(!xmatch<K>(&Xm[i*K],&Zm[i*K],&Vw[i*K], &Ww[i*K], &S.N[i*K])) mw++;
    }
    /* also confirm host==device for a subset of the Edwards DA path */
    long hd=0; int sub=L/16;
    for(int i=0;i<sub;i++){
        const u64*n=&S.N[i*K]; u64 np=S.NP[i];
        u64 dm[K]; montmul<K>(dm,&S.DM[i*K],&S.R2[i*K],n,np);
        EPT<K> P; esetup<K>(P,&S.U0[i*K],&S.V0[i*K],&S.R1[i*K],&S.R2[i*K],n,np);
        EPT<K> Q; edw_da<K>(Q,P,dm,bitsDA.data(),nb,&S.R1[i*K],n,np);
        u64 one[K]; mp_set0<K>(one); one[0]=1; u64 v[K],wv[K];
        montmul<K>(v,Q.Y,one,n,np); montmul<K>(wv,Q.Z,one,n,np);
        for(int j=0;j<K;j++) if(v[j]!=Vda[i*K+j]||wv[j]!=Wda[i*K+j]){ hd++; break; }
    }
    bool err=e1||e2||e3;
    printf("  [%s] Edwards a=-1: DA match %s(%ld/%d) wNAF%d match %s(%ld/%d) host=dev %s(%ld/%d)%s\n",
        label, mda==0?"PASS":"FAIL",mda,L, w, mw==0?"PASS":"FAIL",mw,L,
        hd==0?"PASS":"FAIL",hd,sub, err?" CUDAERR":"");
    printf("       throughput: ladder %.0f c/s | Edwards-DA %.0f c/s (%.2fx) | wNAF%d %.0f c/s (%.2fx)\n",
        L/sMont, L/sDA, sMont/sDA, w, L/sW, sMont/sW);
    if(mda||mw||hd||err) fails++;

    cudaFree(dN);cudaFree(dNP);cudaFree(dR1);cudaFree(dR2);cudaFree(dX0);cudaFree(dA24);
    cudaFree(dU0);cudaFree(dV0);cudaFree(dDM);cudaFree(dXm);cudaFree(dZm);
    cudaFree(dVda);cudaFree(dWda);cudaFree(dVw);cudaFree(dWw);cudaFree(dbits);cudaFree(ddig);
    return fails;
}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    const u64 B1=2000; const int W=4;
    /* prime-power multipliers (for the ladder) + s = their product (for Edwards) */
    std::vector<u64> sP; std::vector<char> comp(B1+1,0); Big s; s.d.push_back(1);
    for(u64 p=2;p<=B1;p++) if(!comp[p]){ for(u64 q=p*p;q<=B1;q+=p) comp[q]=1;
        u64 pe=p; while(pe*p<=B1) pe*=p; sP.push_back(pe); big_mul_small(s,(uint32_t)pe); }
    int ns=(int)sP.size();
    std::vector<unsigned char> bitsDA=big_bits_msb(s);
    std::vector<signed char> wnaf=big_wnaf_msb(s,W);
    printf("GPU mixed-rep ECM stage-1: twisted-Edwards a=-1 vs Montgomery ladder\n");
    printf("B1=%llu, %d prime powers, s has %d bits; ladder 11 mm/bit, Edwards dbl=8 add=9\n",
           (unsigned long long)B1, ns, big_bitlen(s));
    printf("theory mm/bit: ladder 11.0 | Edwards-DA ~%.1f | wNAF%d ~%.1f (table %d pts/thread)\n",
           8.0+9.0*0.5, W, 8.0+9.0/(W+1), 1<<(W-2));
    int fails=0;
    fails+=run_width<2>("128-bit ",sP,ns,bitsDA,wnaf,W);
    fails+=run_width<4>("256-bit ",sP,ns,bitsDA,wnaf,W);
    fails+=run_width<8>("512-bit ",sP,ns,bitsDA,wnaf,W);
    printf("%s\n", fails==0?"ALL PASS":"FAILURES");
    return fails!=0;
}
