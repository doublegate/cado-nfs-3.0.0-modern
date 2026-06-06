/*
 * gpu-prefactor.cu — standalone GPU ECM pre-factoring front-end (v3.1.0 Track 2.1).
 *
 * Strips small/medium prime factors from a large N on the GPU with batched ECM,
 * BEFORE handing the (reduced) cofactor to NFS — which CADO already advises users
 * to do by hand (README: "strip small prime factors ... before using it"). This
 * sidesteps the Amdahl wall that makes in-sieve GPU cofactorization a no-win
 * (docs/gpu-cofactorization.md): pre-factoring is a *separate* stage, so the GPU's
 * ~39x ECM throughput is pure upside when N has a findable factor.
 *
 * The device math is the bit-exact-validated multi-precision Montgomery ECM
 * (gpu_ecm_mp.cuh; see bench/gpu-ecm-mp.cu). GMP handles N parsing, the
 * per-modulus Montgomery setup (n^{-1}, R mod n, R^2 mod n), and gcd(Z,n).
 *
 *   nvcc -arch=sm_86 -O3 misc/gpu_prefactor/gpu-prefactor.cu -lgmp -o gpu-prefactor
 *   ./gpu-prefactor <N> [B1=50000] [curves=4096]
 *
 * Stage-1 only for now (stage-2 BSGS, Suyama-sigma, and multi-GPU are the next
 * Track 2.1 increments). Exit code 0 if at least one factor was stripped.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <gmp.h>
#include "gpu_ecm_mp.cuh"

static u64 ninv64(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }

/* export mpz -> exactly K 64-bit little-endian limbs (zero-padded) */
static void to_limbs(u64 *out, int K, const mpz_t v){
    for(int i=0;i<K;i++) out[i]=0;
    size_t count=0;
    mpz_export(out, &count, -1, 8, 0, 0, v);   /* little-endian, 8-byte words */
}
static void from_limbs(mpz_t v, const u64 *in, int K){
    mpz_import(v, K, -1, 8, 0, 0, in);
}

/* Suyama-sigma curve setup mod N (GMP): sigma -> (X0,Z0,a24). Returns true and
 * fills X0/Z0/a24; if a denominator is not invertible mod N, returns false and
 * sets `lucky` = gcd(denominator,N), itself a nontrivial factor. */
static bool suyama_setup(mpz_t X0,mpz_t Z0,mpz_t a24,mpz_t lucky,
                         unsigned long sigma,const mpz_t N){
    mpz_t s,u,v,u3,num,den,t1,di; mpz_inits(s,u,v,u3,num,den,t1,di,NULL);
    mpz_set_ui(s,sigma); mpz_mod(s,s,N);
    mpz_mul(u,s,s); mpz_sub_ui(u,u,5); mpz_mod(u,u,N);          /* u = s^2 - 5 */
    mpz_mul_ui(v,s,4); mpz_mod(v,v,N);                          /* v = 4 s   */
    mpz_powm_ui(u3,u,3,N);                                      /* u^3       */
    mpz_set(X0,u3);                                             /* X0 = u^3  */
    mpz_powm_ui(Z0,v,3,N);                                      /* Z0 = v^3  */
    mpz_sub(t1,v,u); mpz_powm_ui(t1,t1,3,N);                    /* (v-u)^3   */
    mpz_mul_ui(num,u,3); mpz_add(num,num,v); mpz_mod(num,num,N);/* 3u+v      */
    mpz_mul(num,num,t1); mpz_mod(num,num,N);                    /* (v-u)^3(3u+v) */
    mpz_mul(den,u3,v); mpz_mul_ui(den,den,16); mpz_mod(den,den,N); /* 16 u^3 v */
    bool ok=true;
    if(mpz_invert(di,den,N)==0){ mpz_gcd(lucky,den,N); ok=false; }
    else { mpz_mul(a24,num,di); mpz_mod(a24,a24,N); }
    mpz_clears(s,u,v,u3,num,den,t1,di,NULL);
    return ok;
}

/* One GPU ECM pass (stage1 to B1 + stage2 BSGS to B2, Suyama curves) on N at the
 * chosen width K; append any nontrivial factors (gcd of z1 or g2 with N, plus
 * Suyama lucky factors). */
template<int K>
static bool ecm_pass(const mpz_t N, int ncurves,
                     const std::vector<u64> &spow, const std::vector<u64> &pr,
                     std::vector<std::string> &found, double &cps){
    int ns=(int)spow.size(), npr=(int)pr.size();
    u64 Nl[K]; to_limbs(Nl, K, N);
    u64 np = ninv64(Nl[0]);
    mpz_t t,R1m,R2m,g,z; mpz_inits(t,R1m,R2m,g,z,NULL);
    mpz_setbit(t, 64*K); mpz_mod(R1m, t, N);
    mpz_set_ui(t,0); mpz_setbit(t,128*K); mpz_mod(R2m, t, N);
    u64 R1[K],R2[K]; to_limbs(R1,K,R1m); to_limbs(R2,K,R2m);

    /* per-curve Suyama setup on the host */
    std::vector<u64> X0(ncurves*K), Z0(ncurves*K), A24(ncurves*K);
    mpz_t mx,mz,ma,luck; mpz_inits(mx,mz,ma,luck,NULL);
    bool any=false;
    auto add_factor=[&](const mpz_t f){
        if(mpz_cmp_ui(f,1)>0 && mpz_cmp(f,N)<0){
            char *s=mpz_get_str(NULL,10,f); std::string fac(s); free(s);
            for(auto&x:found) if(x==fac) return; found.push_back(fac); any=true; }
    };
    for(int i=0;i<ncurves;i++){
        unsigned long sigma = 6 + (unsigned long)i*2 + 1;       /* distinct sigmas */
        if(suyama_setup(mx,mz,ma,luck,sigma,N)){
            to_limbs(&X0[i*K],K,mx); to_limbs(&Z0[i*K],K,mz); to_limbs(&A24[i*K],K,ma);
        } else {
            add_factor(luck);                                  /* free factor */
            /* fall back to a POC curve for this lane so the kernel has valid data */
            mpz_set_ui(mx,2); mpz_set_ui(mz,1); mpz_set_ui(ma,sigma);
            to_limbs(&X0[i*K],K,mx); to_limbs(&Z0[i*K],K,mz); to_limbs(&A24[i*K],K,ma);
        }
    }
    mpz_clears(mx,mz,ma,luck,NULL);

    /* host per-lane N/np/R1/R2 (identical across lanes) */
    std::vector<u64> Nv(ncurves*K),R1v(ncurves*K),R2v(ncurves*K),NPv(ncurves);
    for(int i=0;i<ncurves;i++){ mp_copy<K>(&Nv[i*K],Nl); mp_copy<K>(&R1v[i*K],R1);
        mp_copy<K>(&R2v[i*K],R2); NPv[i]=np; }
    std::vector<u64> Z1(ncurves*K),G2(ncurves*K);

    /* multi-GPU: split the curve batch across all visible devices, launch each
     * asynchronously, then synchronize — so multiple GPUs run concurrently.
     * Degenerates to a single launch on a one-GPU box. */
    int ndev=1; if(cudaGetDeviceCount(&ndev)!=cudaSuccess||ndev<1) ndev=1;
    int per=(ncurves+ndev-1)/ndev;
    struct Slice{ int start,count;
        u64 *dN,*dNP,*dR1,*dR2,*dX0,*dZ0,*dA24,*ds,*dpr,*dZ1,*dG2; };
    std::vector<Slice> sl(ndev);
    bool ok=true;
    auto t0=std::chrono::steady_clock::now();
    for(int d=0;d<ndev;d++){
        int start=d*per, count=ncurves-start; if(count>per) count=per;
        sl[d].start=start; sl[d].count=count; if(count<=0) continue;
        if(cudaSetDevice(d)!=cudaSuccess){ ok=false; break; }
        size_t cb=(size_t)count*K*8; Slice&S=sl[d];
        cudaMalloc(&S.dN,cb);cudaMalloc(&S.dNP,count*8);cudaMalloc(&S.dR1,cb);
        cudaMalloc(&S.dR2,cb);cudaMalloc(&S.dX0,cb);cudaMalloc(&S.dZ0,cb);
        cudaMalloc(&S.dA24,cb);cudaMalloc(&S.dZ1,cb);cudaMalloc(&S.dG2,cb);
        cudaMalloc(&S.ds,ns>0?ns*8:8);cudaMalloc(&S.dpr,npr>0?npr*8:8);
        cudaMemcpyAsync(S.dN,&Nv[start*K],cb,cudaMemcpyHostToDevice);
        cudaMemcpyAsync(S.dNP,&NPv[start],count*8,cudaMemcpyHostToDevice);
        cudaMemcpyAsync(S.dR1,&R1v[start*K],cb,cudaMemcpyHostToDevice);
        cudaMemcpyAsync(S.dR2,&R2v[start*K],cb,cudaMemcpyHostToDevice);
        cudaMemcpyAsync(S.dX0,&X0[start*K],cb,cudaMemcpyHostToDevice);
        cudaMemcpyAsync(S.dZ0,&Z0[start*K],cb,cudaMemcpyHostToDevice);
        cudaMemcpyAsync(S.dA24,&A24[start*K],cb,cudaMemcpyHostToDevice);
        if(ns>0) cudaMemcpyAsync(S.ds,spow.data(),ns*8,cudaMemcpyHostToDevice);
        if(npr>0) cudaMemcpyAsync(S.dpr,pr.data(),npr*8,cudaMemcpyHostToDevice);
        int tpb=64, blk=(count+tpb-1)/tpb;
        ecm_kernel2<K><<<blk,tpb>>>(S.dN,S.dNP,S.dR1,S.dR2,S.dX0,S.dZ0,S.dA24,
                                    S.ds,ns,S.dpr,npr,S.dZ1,S.dG2,count);
        cudaMemcpyAsync(&Z1[start*K],S.dZ1,cb,cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(&G2[start*K],S.dG2,cb,cudaMemcpyDeviceToHost);
    }
    for(int d=0;d<ndev;d++){
        if(sl[d].count<=0) continue;
        cudaSetDevice(d); cudaDeviceSynchronize();
        if(cudaGetLastError()!=cudaSuccess) ok=false;
        Slice&S=sl[d];
        cudaFree(S.dN);cudaFree(S.dNP);cudaFree(S.dR1);cudaFree(S.dR2);cudaFree(S.dX0);
        cudaFree(S.dZ0);cudaFree(S.dA24);cudaFree(S.dZ1);cudaFree(S.dG2);cudaFree(S.ds);cudaFree(S.dpr);
    }
    cudaSetDevice(0);
    auto t1=std::chrono::steady_clock::now();
    double sec=std::chrono::duration<double>(t1-t0).count();
    cps = sec>0 ? ncurves/sec : 0;
    if(ndev>1) printf("# using %d GPUs (%d curves each)\n", ndev, per);
    /* bit-exact self-check: re-run a subset on the CPU (ecm_run2 is host+device)
     * and compare z1/g2. The stage-2 BSGS is a new composition of the validated
     * primitives, so verify it every run (cheap; honors the fork's ethos). */
    if(ok){
        int chk = ncurves<32?ncurves:32; long mis=0;
        for(int i=0;i<chk;i++){
            u64 z1[K],g2[K];
            ecm_run2<K>(z1,g2,&Nv[i*K],NPv[i],&R1v[i*K],&R2v[i*K],
                        &X0[i*K],&Z0[i*K],&A24[i*K],spow.data(),ns,pr.data(),npr);
            for(int j=0;j<K;j++) if(z1[j]!=Z1[i*K+j]||g2[j]!=G2[i*K+j]){ mis++; break; }
        }
        printf("# selfcheck: %s (%ld/%d GPU lanes differ from CPU)\n",
               mis==0?"PASS":"FAIL", mis, chk);
        if(mis) ok=false;
    }
    if(ok){
        for(int i=0;i<ncurves;i++){
            from_limbs(z,&Z1[(size_t)i*K],K); mpz_gcd(g,z,N); add_factor(g);
            from_limbs(z,&G2[(size_t)i*K],K); mpz_gcd(g,z,N); add_factor(g);
        }
    }
    /* per-slice device buffers were freed in the synchronize loop above */
    mpz_clears(t,R1m,R2m,g,z,NULL);
    return any;
}

int main(int argc, char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <N> [B1=50000] [curves=4096] [B2=100*B1]\n",argv[0]); return 2; }
    mpz_t N; mpz_init(N);
    if(mpz_set_str(N, argv[1], 10)!=0){ fprintf(stderr,"bad N\n"); return 2; }
    unsigned long B1 = argc>2 ? strtoul(argv[2],0,10) : 50000;
    int ncurves      = argc>3 ? atoi(argv[3]) : 4096;
    unsigned long B2 = argc>4 ? strtoul(argv[4],0,10) : 100UL*B1;
    if(B2<B1) B2=B1;

    size_t bits = mpz_sizeinbase(N,2);
    int needK = (int)((bits+2+63)/64);
    int K = needK<=2?2: needK<=4?4: needK<=8?8: needK<=16?16: 0;
    if(K==0){ fprintf(stderr,"N too large (%zu bits); supported up to 1022 bits (~307 digits)\n",bits); return 2; }
    if(mpz_even_p(N)){ fprintf(stderr,"N is even; strip factors of 2 first\n"); return 2; }

    /* sieve to B2: stage-1 prime powers <= B1, stage-2 primes in (B1,B2] */
    std::vector<u64> spow, pr; std::vector<char> comp(B2+1,0);
    for(unsigned long p=2;p<=B2;p++) if(!comp[p]){
        for(unsigned long q=p*p;q<=B2;q+=p) comp[q]=1;
        if(p<=B1){ u64 pe=p; while(pe*p<=B1) pe*=p; spow.push_back(pe); }
        else pr.push_back(p);
    }

    printf("# GPU ECM pre-factor: %zu-bit N (~%zu digits), width K=%d (%d-bit), B1=%lu, B2=%lu, curves=%d (Suyama+stage2)\n",
           bits, mpz_sizeinbase(N,10), K, 64*K, B1, B2, ncurves);
    printf("#   %zu stage-1 multipliers, %zu stage-2 primes\n", spow.size(), pr.size());

    std::vector<std::string> found; double cps=0; bool any=false;
    switch(K){
        case 2:  any=ecm_pass<2 >(N,ncurves,spow,pr,found,cps); break;
        case 4:  any=ecm_pass<4 >(N,ncurves,spow,pr,found,cps); break;
        case 8:  any=ecm_pass<8 >(N,ncurves,spow,pr,found,cps); break;
        case 16: any=ecm_pass<16>(N,ncurves,spow,pr,found,cps); break;
    }
    printf("# throughput: %.0f curves/s on the GPU (stage1+stage2)\n", cps);

    /* divide out everything found; report factors + remaining cofactor */
    mpz_t cof; mpz_init_set(cof,N);
    for(auto &f: found){ mpz_t fz; mpz_init_set_str(fz,f.c_str(),10);
        while(mpz_divisible_p(cof,fz)) mpz_divexact(cof,cof,fz);
        mpz_clear(fz); }
    if(any){
        printf("factors stripped:");
        for(auto &f: found) printf(" %s", f.c_str());
        printf("\n");
        char *c=mpz_get_str(NULL,10,cof);
        printf("remaining cofactor: %s%s\n", c,
               mpz_probab_prime_p(cof,25)? "  (prime)" :
               (mpz_cmp_ui(cof,1)==0? "  (fully factored)" : "  (composite -> hand to NFS)"));
        free(c);
    } else {
        printf("no factor found at B1=%lu with %d curves (try a larger B1 / more curves)\n",
               B1, ncurves);
    }
    mpz_clear(cof); mpz_clear(N);
    return any?0:1;
}
