/*
 * gpu-xdotprod-bench.cu — GPU x_dotprod for the Block Wiedemann inner loop, the
 * second compute primitive (after the SpMV) needed to keep BWC vectors resident
 * on the GPU (v3.1.0-modern, Track 2.2, full-residency port step 1).
 *
 * krylov.cpp calls x_dotprod(dst, xv, j0, j1, nx, v, sign) every inner iteration
 * (linalg/bwc/xdotprod.cpp): for each output row j in [j0,j1), and each of the m
 * "blocking" x-vectors, it XORs (GF(2): add==sub) the K-limb block-of-vectors
 * element v[i - i0] for the nx sparse positions i = xv[j*nx + t] that fall in the
 * local range [i0,i1). Today that reads v on the HOST every iteration — the main
 * reason a device-resident vector still has to come back to the CPU. This kernel
 * does the same gather on the GPU, reading a device-resident v.
 *
 * Same __host__ __device__ gather on CPU and GPU => validated bit-exact.
 *   nvcc -arch=sm_86 -O3 bench/gpu-xdotprod-bench.cu -o /tmp/gpu-xdot && /tmp/gpu-xdot
 */
#include <cstdio>
#include <cstdint>
#include <vector>
typedef uint64_t u64;
typedef uint32_t u32;
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

static u64 xr(u64*s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

/* dst[(j-j0)*K + k] = XOR over t<nx of v[(xv[j*nx+t]-i0)*K + k], for i in [i0,i1).
 * Mirrors xdotprod.cpp exactly for GF(2) (sign is XOR either way). One thread per
 * output row j (j in [j0,j1)); j-j0 indexes dst. */
template<int K>
HD static void xdot_one(u64 *dst, const u32 *xv, unsigned j, unsigned nx,
                        const u64 *v, unsigned i0, unsigned i1){
    u64 acc[K];
    for(int k=0;k<K;k++) acc[k]=0;
    for(unsigned t=0;t<nx;t++){
        u32 i = xv[(size_t)j*nx + t];
        if(i<i0 || i>=i1) continue;
        const u64 *e = v + (size_t)(i-i0)*K;
        for(int k=0;k<K;k++) acc[k]^=e[k];
    }
    for(int k=0;k<K;k++) dst[(size_t)j*K + k]=acc[k];
}

template<int K>
__global__ void xdot_kernel(u64 *dst, const u32 *xv, unsigned m, unsigned nx,
                            const u64 *v, unsigned i0, unsigned i1){
    unsigned j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=m) return;
    xdot_one<K>(dst, xv, j, nx, v, i0, i1);
}

template<int K>
static int run(const char* label, unsigned n, unsigned m, unsigned nx){
    u64 st=0xD07ULL+K;
    std::vector<u64> v((size_t)n*K); for(auto&x:v) x=xr(&st);
    std::vector<u32> xv((size_t)m*nx); for(auto&x:xv) x=(u32)(xr(&st)%n);
    unsigned i0=0, i1=n;                          /* single owner: whole vector local */
    std::vector<u64> dg((size_t)m*K), dc((size_t)m*K);

    u64 *dv,*ddst; u32 *dxv;
    if(cudaMalloc(&dv,v.size()*8)!=cudaSuccess){ printf("  [%s] malloc fail\n",label); return 1; }
    cudaMalloc(&dxv,xv.size()*4); cudaMalloc(&ddst,dg.size()*8);
    cudaMemcpy(dv,v.data(),v.size()*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dxv,xv.data(),xv.size()*4,cudaMemcpyHostToDevice);
    int tpb=128, blk=(int)((m+tpb-1)/tpb);
    xdot_kernel<K><<<blk,tpb>>>(ddst,dxv,m,nx,dv,i0,i1);
    cudaDeviceSynchronize();
    cudaError_t e=cudaGetLastError();
    cudaMemcpy(dg.data(),ddst,dg.size()*8,cudaMemcpyDeviceToHost);
    cudaFree(dv);cudaFree(dxv);cudaFree(ddst);

    for(unsigned j=0;j<m;j++) xdot_one<K>(dc.data(),xv.data(),j,nx,v.data(),i0,i1);
    long mis=0; for(size_t i=0;i<dc.size();i++) if(dc[i]!=dg[i]) mis++;
    printf("  [%s] n=%u m=%u nx=%u : %s (%ld/%zu words differ)%s\n",
           label, n, m, nx, mis==0?"PASS":"FAIL", mis, dc.size(), e?"  CUDAERR":"");
    return mis!=0;
}

int main(){
    printf("GPU x_dotprod (BWC inner-loop gather) — GPU validated bit-exact vs CPU\n");
    int f=0;
    f += run<1>("b64 ", 2000000, 64,  256);   /* m=64 blocking vectors, nx=256 sparse positions */
    f += run<2>("b128", 2000000, 128, 256);
    f += run<4>("b256", 1000000, 256, 384);
    printf("%s\n", f==0?"ALL PASS":"FAILURES");
    return f!=0;
}
