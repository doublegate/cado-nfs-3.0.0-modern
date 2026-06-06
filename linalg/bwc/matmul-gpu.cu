#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <utility>
#include <map>

#include <cuda_runtime.h>

#include "arith-generic.hpp"
#include "arith-hard.hpp"
#include "macros.h"
#include "matmul.hpp"
#include "matmul-common.hpp"
#include "matrix_u32.hpp"
#include "params.hpp"

/* GPU GF(2) SpMV backend for Block Wiedemann (v3.1.0-modern, Track 2.2).
 *
 * Same cache format and semantics as matmul-basic (per nonzero (i,j):
 * dst[i] ^= src[j], with each element a bitsliced block of K 64-bit limbs), but
 * the SpMV runs on the GPU. The matrix is kept resident on the device across the
 * thousands of BWC iterations; only the src/dst vectors cross the bus per call.
 * Both directions are served as *gathers* (one thread per output row) by storing
 * the matrix as CSR and its transpose on the device. The kernel is bit-exact
 * with the CPU path (validated in bench/gpu-spmv-bench.cu and by bench_matcache's
 * (M v1).v2 == (M^T v2).v1 check). See docs/gpu-linalg.md. */

#define MM_EXTENSION   "-gpu"
#define MM_MAGIC_FAMILY        0xa011UL
#define MM_MAGIC_VERSION       0x1001UL
#define MM_MAGIC (MM_MAGIC_FAMILY << 16 | MM_MAGIC_VERSION)

/* Follow matmul-basic's storage convention (so store_transposed matches the
 * caller's expectation / the bench_matcache check). We keep both the matrix and
 * its transpose on the device anyway, so both directions are fast gathers. */
#define MM_DIR0_PREFERS_TRANSP_MULT   1

#define CUCHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "matmul-gpu: %s: %s\n", #call, cudaGetErrorString(e_)); \
    abort(); } } while (0)

/* one thread per output row: dst[i] = XOR over row i of src[col], K limbs each */
template<int K>
__global__ void spmv_gather(const uint32_t * rowptr, const uint32_t * col,
                            const uint64_t * src, uint64_t * dst, unsigned int nrows)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows) return;
    uint64_t acc[K];
#pragma unroll
    for (int k = 0; k < K; k++) acc[k] = 0;
    uint32_t lo = rowptr[i], hi = rowptr[i + 1];
    for (uint32_t p = lo; p < hi; p++) {
        const uint64_t * s = src + (size_t) col[p] * K;
#pragma unroll
        for (int k = 0; k < K; k++) acc[k] ^= s[k];
    }
    uint64_t * d = dst + (size_t) i * K;
#pragma unroll
    for (int k = 0; k < K; k++) d[k] = acc[k];
}

/* coalesced variant: one warp per output row. Lanes stride the row's nonzeros
 * (col[] reads coalesce), src is gathered through the read-only cache (__ldg),
 * then the K-limb accumulator is warp-reduced. Bit-exact with spmv_gather; ~1.8-3x
 * faster (validated in bench/gpu-spmv-bench.cu). */
template<int K>
__global__ void spmv_warp(const uint32_t * rowptr, const uint32_t * col,
                          const uint64_t * src, uint64_t * dst, unsigned int nrows)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = gid >> 5, lane = gid & 31;
    if (row >= nrows) return;
    uint64_t acc[K];
#pragma unroll
    for (int k = 0; k < K; k++) acc[k] = 0;
    uint32_t lo = rowptr[row], hi = rowptr[row + 1];
    for (uint32_t p = lo + lane; p < hi; p += 32) {
        const uint64_t * s = src + (size_t) __ldg(&col[p]) * K;
#pragma unroll
        for (int k = 0; k < K; k++) acc[k] ^= __ldg(&s[k]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int k = 0; k < K; k++) acc[k] ^= __shfl_down_sync(0xffffffffu, acc[k], off);
    if (lane == 0) { uint64_t * d = dst + (size_t) row * K;
#pragma unroll
        for (int k = 0; k < K; k++) d[k] = acc[k]; }
}

static void launch_spmv(int K, const uint32_t * rp, const uint32_t * col,
                        const uint64_t * src, uint64_t * dst, unsigned int nrows)
{
    int tpb = 128; int blk = (int)(((size_t) nrows * 32 + tpb - 1) / tpb);
    switch (K) {
        case 1: spmv_warp<1><<<blk, tpb>>>(rp, col, src, dst, nrows); break;
        case 2: spmv_warp<2><<<blk, tpb>>>(rp, col, src, dst, nrows); break;
        case 4: spmv_warp<4><<<blk, tpb>>>(rp, col, src, dst, nrows); break;
        default:
            fprintf(stderr, "matmul-gpu: unsupported block width K=%d\n", K);
            abort();
    }
}

template<typename Arith>
struct matmul_gpu : public matmul_interface {
    Arith * xab;
    std::vector<uint32_t> q;            /* same flattened [len,col,...] cache as basic */
    int K = (int)(sizeof(typename Arith::elt) / sizeof(uint64_t));

    /* device residency, built lazily on first mul(); index 0 = stored direction
     * (the one matching d == !store_transposed), 1 = its transpose */
    bool dev_ready = false;
    uint32_t * d_rp[2] = { nullptr, nullptr };
    uint32_t * d_col[2] = { nullptr, nullptr };
    unsigned int nrows_dir[2] = { 0, 0 };
    /* persistent per-host-vector device buffer pool. With CADO_GPU_VECRESIDENT
     * set, a buffer flagged `current` (device == host) lets mul() skip the H2D
     * upload; host_vector_modified() clears the flag when the caller writes the
     * host buffer (the comm / twist). Default (flag unset): always upload, so
     * behaviour is bit-identical to before. */
    struct DV { uint64_t * d = nullptr; size_t bytes = 0; bool current = false; };
    std::map<const void *, DV> pool;
    DV & dv_for(const void * host, size_t bytes) {
        DV & x = pool[host];
        if (x.bytes < bytes) { if (x.d) cudaFree(x.d); CUCHECK(cudaMalloc(&x.d, bytes));
            x.bytes = bytes; x.current = false; }
        return x;
    }

    /* BWC reuses the same host vector buffers across the thousands of SpMV
     * iterations, so we page-lock (pin) each one the first time we see it ->
     * every H2D/D2H transfer then runs at full PCIe bandwidth instead of the
     * staged pageable path. (Full device residency -- vectors never leaving the
     * GPU -- would need changes in the mmt_vec layer above; this is the safe,
     * contained win at the backend level.) */
    /* split-timing accumulators (ms), used when CADO_GPU_TIMING is set */
    double t_h2d = 0, t_ker = 0, t_d2h = 0; long t_n = 0;

    std::map<const void *, size_t> pinned;
    void ensure_pinned(const void * p, size_t bytes) {
        auto it = pinned.find(p);
        if (it != pinned.end()) {
            if (bytes <= it->second) return;          /* already pinned big enough */
            cudaHostUnregister((void *) p);           /* grow the registration */
            pinned.erase(it);
        }
        if (cudaHostRegister((void *) p, bytes, cudaHostRegisterDefault) == cudaSuccess)
            pinned[p] = bytes;
        else
            cudaGetLastError();                       /* not pinnable -> pageable, still correct */
    }

    void build_cache(matrix_u32 &&) override;
    int reload_cache_private() override;
    void save_cache_private() override;
    void mul(void *, const void *, int) override;
    void host_vector_modified(void const *) override;
    void ensure_device();
    ~matmul_gpu() override;

    matmul_gpu(matmul_public && P, arith_concrete_base * pxx, cxx_param_list & pl, int optimized_direction)
        : matmul_interface(std::move(P))
        , xab((Arith *) pxx) // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    {
        int const suggest = optimized_direction ^ MM_DIR0_PREFERS_TRANSP_MULT;
        store_transposed = suggest;
        param_list_parse(pl, "mm_store_transposed", store_transposed);
    }
    matmul_gpu(matmul_gpu const &) = delete;
    matmul_gpu& operator=(matmul_gpu const &) = delete;
    matmul_gpu(matmul_gpu &&) noexcept = default;
    matmul_gpu& operator=(matmul_gpu &&) noexcept = default;
};

template<typename Arith>
void matmul_gpu<Arith>::build_cache(matrix_u32 && m) { q = std::move(m.p); }

template<typename Arith>
int matmul_gpu<Arith>::reload_cache_private()
{
    auto f = matmul_common_reload_cache_fopen(sizeof(typename Arith::elt), *this, MM_MAGIC);
    if (!f) return 0;
    uint32_t datasize;
    MATMUL_COMMON_READ_ONE32(datasize, f.get());
    resize_and_check_meaningful(q, datasize, f.get());
    MATMUL_COMMON_READ_MANY32(q.data(), datasize, f.get());
    return 1;
}

template<typename Arith>
void matmul_gpu<Arith>::save_cache_private()
{
    auto f = matmul_common_save_cache_fopen(sizeof(typename Arith::elt), *this, MM_MAGIC);
    if (!f) return;
    MATMUL_COMMON_WRITE_ONE32(q.size(), f.get());
    MATMUL_COMMON_WRITE_MANY32(q.data(), q.size(), f.get());
}

/* Parse q into CSR for the stored direction, build its transpose, upload both. */
template<typename Arith>
void matmul_gpu<Arith>::ensure_device()
{
    if (dev_ready) return;

    /* stored direction matches "d == !store_transposed" in matmul-basic: it
     * iterates dim[store_transposed] rows, columns index a dim[!store_transposed]
     * source. */
    unsigned int nr = dim[store_transposed];
    unsigned int nc = dim[!store_transposed];

    std::vector<uint32_t> rp0(nr + 1, 0), col0;
    const uint32_t * qq = q.data();
    col0.reserve(q.size());
    for (unsigned int i = 0; i < nr; i++) {
        uint32_t len = *qq++;
        rp0[i + 1] = rp0[i] + len;
        for (uint32_t t = 0; t < len; t++) col0.push_back(*qq++);
    }
    size_t nnz = col0.size();

    /* transpose: counting sort of (i -> j) into (j -> i) */
    std::vector<uint32_t> rp1(nc + 1, 0), col1(nnz);
    for (size_t p = 0; p < nnz; p++) rp1[col0[p] + 1]++;
    for (unsigned int j = 0; j < nc; j++) rp1[j + 1] += rp1[j];
    {
        std::vector<uint32_t> cur(rp1.begin(), rp1.end() - 1);
        for (unsigned int i = 0; i < nr; i++)
            for (uint32_t p = rp0[i]; p < rp0[i + 1]; p++)
                col1[cur[col0[p]]++] = i;
    }

    nrows_dir[0] = nr; nrows_dir[1] = nc;
    auto up = [](uint32_t * & dptr, const std::vector<uint32_t> & h) {
        CUCHECK(cudaMalloc(&dptr, h.size() * sizeof(uint32_t)));
        CUCHECK(cudaMemcpy(dptr, h.data(), h.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    };
    up(d_rp[0], rp0); up(d_col[0], col0);
    up(d_rp[1], rp1); up(d_col[1], col1);
    dev_ready = true;
}

template<typename Arith>
void matmul_gpu<Arith>::mul(void * xdst, void const * xsrc, int d)
{
    ensure_device();
    int dir = (d == !store_transposed) ? 0 : 1;
    unsigned int nrows = nrows_dir[dir];        /* == dim[!d] */
    unsigned int ncols = nrows_dir[!dir];       /* == dim[d]  */
    ASSERT_ALWAYS(nrows == dim[!d]);

    size_t srcbytes = (size_t) ncols * K * sizeof(uint64_t);
    size_t dstbytes = (size_t) nrows * K * sizeof(uint64_t);
    DV & sv = dv_for(xsrc, srcbytes);
    DV & wv = dv_for(xdst, dstbytes);

    ensure_pinned(xsrc, srcbytes);
    ensure_pinned(xdst, dstbytes);

    static const bool resident = getenv("CADO_GPU_VECRESIDENT") != nullptr;
    static const bool timing = getenv("CADO_GPU_TIMING") != nullptr;
    bool skip_h2d = resident && sv.current;     /* device copy of src is up to date */

    if (!timing) {
        if (!skip_h2d) { CUCHECK(cudaMemcpy(sv.d, xsrc, srcbytes, cudaMemcpyHostToDevice)); sv.current = true; }
        launch_spmv(K, d_rp[dir], d_col[dir], sv.d, wv.d, nrows);
        CUCHECK(cudaGetLastError());
        CUCHECK(cudaMemcpy(xdst, wv.d, dstbytes, cudaMemcpyDeviceToHost));
        wv.current = true;                      /* device buffer == host buffer now */
    } else {
        cudaEvent_t e0, e1, e2, e3;
        cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
        cudaEventRecord(e0);
        if (!skip_h2d) { CUCHECK(cudaMemcpy(sv.d, xsrc, srcbytes, cudaMemcpyHostToDevice)); sv.current = true; }
        cudaEventRecord(e1);
        launch_spmv(K, d_rp[dir], d_col[dir], sv.d, wv.d, nrows);
        cudaEventRecord(e2);
        CUCHECK(cudaMemcpy(xdst, wv.d, dstbytes, cudaMemcpyDeviceToHost));
        wv.current = true;
        cudaEventRecord(e3);
        cudaEventSynchronize(e3);
        float h2d, ker, d2h;
        cudaEventElapsedTime(&h2d, e0, e1);
        cudaEventElapsedTime(&ker, e1, e2);
        cudaEventElapsedTime(&d2h, e2, e3);
        t_h2d += h2d; t_ker += ker; t_d2h += d2h; t_n++;
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    }

    iteration[d]++;
}

template<typename Arith>
void matmul_gpu<Arith>::host_vector_modified(void const * hostvec)
{
    auto it = pool.find(hostvec);
    if (it != pool.end()) it->second.current = false;
}

template<typename Arith>
matmul_gpu<Arith>::~matmul_gpu()
{
    if (t_n) {
        fprintf(stderr, "# matmul-gpu split over %ld SpMV: H2D %.3f ms, kernel "
                "%.3f ms, D2H %.3f ms (per call); transfers %.0f%%\n",
                t_n, t_h2d / t_n, t_ker / t_n, t_d2h / t_n,
                100.0 * (t_h2d + t_d2h) / (t_h2d + t_ker + t_d2h));
    }
    for (int i = 0; i < 2; i++) { if (d_rp[i]) cudaFree(d_rp[i]); if (d_col[i]) cudaFree(d_col[i]); }
    for (auto const & kv : pool) if (kv.second.d) cudaFree(kv.second.d);
    for (auto const & kv : pinned) cudaHostUnregister((void *) kv.first);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
matmul_interface * CADO_CONCATENATE4(new_matmul_, ARITH_LAYER, _, MM_IMPL)(
        matmul_public && P,
        arith_generic * arith,
        cxx_param_list & pl,
        int optimized_direction)
{
    return new matmul_gpu<arith_hard>(std::move(P), arith->concrete(), pl, optimized_direction);
}

/* vim: set sw=4: */
