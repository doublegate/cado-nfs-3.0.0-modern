#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <utility>
#include <map>
#include <mutex>

#include <cuda_runtime.h>

#include "arith-generic.hpp"
#include "arith-hard.hpp"
#include "macros.h"
#include "matmul.hpp"
#include "matmul-common.hpp"
#include "matmul-gpu-hooks.h"
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

/* ---- process-global device-vector registry (Track 2.2 residency wiring) ----
 * Keyed by host vector pointer. Process-global (not per-mm) because the BWC comm
 * reduces across *sibling threads'* vectors, which live in other threads' mm; all
 * threads share the one CUDA context, so a sibling's device buffer is reachable
 * here by its host pointer. Thread-safe map ops (the CUDA work on a given buffer
 * is single-threaded: each vector belongs to one thread, and the comm is
 * serialized by serialize_threads). Buffers are freed at process exit. */
namespace {
/* current  : device buffer holds the latest data (uploaded or just computed)
 * host_dirty: the host buffer is stale w.r.t. the device buffer (comm-on-device
 *             left the result only on the device; materialise via sync_to_host) */
struct GDV { uint64_t * d = nullptr; size_t bytes = 0; bool current = false; bool host_dirty = false; };
std::mutex g_mtx;
std::map<const void *, GDV> g_pool;
std::map<const void *, size_t> g_pinned;

/* device buffer for a host vector (created/grown as needed) */
GDV & g_dv(const void * host, size_t bytes) {
    std::lock_guard<std::mutex> lk(g_mtx);
    GDV & x = g_pool[host];
    if (x.bytes < bytes) { if (x.d) cudaFree(x.d);
        if (cudaMalloc(&x.d, bytes) != cudaSuccess) { x.d = nullptr; x.bytes = 0; }
        else x.bytes = bytes;
        x.current = false; }
    return x;
}
void g_pin(const void * p, size_t bytes) {
    /* In comm-on-device mode, host buffers are copied at different (larger) sizes
     * by the comm than by mul(); cudaHostRegister enforces the registered region
     * size and conflicts on overlap/aliasing, which corrupts the CUDA context.
     * Pinning is a transfer-speed optimisation that residency makes moot anyway,
     * so we simply don't pin when device comm is active (copies fall back to
     * pageable — correct, and the transfers are what residency removes). The
     * default (non-DEVCOMM) path keeps pinning and its measured speedup. */
    static const bool devcomm = getenv("CADO_GPU_DEVCOMM") != nullptr;
    if (devcomm) return;
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_pinned.find(p);
    if (it != g_pinned.end()) { if (bytes <= it->second) return;
        cudaHostUnregister((void *) p); g_pinned.erase(it); }
    if (cudaHostRegister((void *) p, bytes, cudaHostRegisterDefault) == cudaSuccess)
        g_pinned[p] = bytes;
    else cudaGetLastError();
}
void g_invalidate(const void * host) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_pool.find(host);
    if (it != g_pool.end()) it->second.current = false;
}
} // namespace

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

/* ---- comm-on-device (Track 2.2): GF(2) reduce + broadcast of sibling vectors --
 * For the single-node BWC comm, all T sibling vectors must become the XOR-sum of
 * the originals (mmt_vec_allreduce). We do it in place on the device-resident
 * copies: reduce all T into sibling[0], then broadcast sibling[0] back over the
 * rest. Each output word g touches only index g across the T buffers, so the
 * in-place reduce into p[0] is race-free. Bit-identical to vec_add_and_reduce on
 * the host (validated standalone in bench/gpu-vecreduce-bench.cu). */
namespace {
#define GPU_COMM_MAX_SIB 64
struct PtrPack { uint64_t * p[GPU_COMM_MAX_SIB]; unsigned int T; };

__global__ void vecreduce_inplace(PtrPack pk, size_t words) {
    size_t g = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= words) return;
    uint64_t acc = 0;
    for (unsigned int t = 0; t < pk.T; t++) acc ^= pk.p[t][g];
    pk.p[0][g] = acc;
}
__global__ void vecbroadcast_n(PtrPack pk, size_t words) {
    size_t g = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= words) return;
    uint64_t v = pk.p[0][g];
    for (unsigned int t = 1; t < pk.T; t++) pk.p[t][g] = v;
}

/* The comm-on-device hook (installed into cado_gpu_comm_reduce_bcast). Returns 1
 * when it handled the comm. Single-threaded per call (one row-communicator leader
 * drives its own disjoint sibling set; see mmt_vec_allreduce). */
int gpu_comm_reduce_bcast_impl(void * const * host_ptrs,
                               unsigned int T, size_t bytes)
{
    if (T < 2) {            /* nothing to reduce; still mark the lone buffer */
        if (T == 1) { GDV & g = g_dv(host_ptrs[0], bytes); g.current = true; }
        return 1;
    }
    if (T > GPU_COMM_MAX_SIB) return 0;     /* unsupported width -> host fallback */
    size_t const words = bytes / sizeof(uint64_t);

    PtrPack pk; pk.T = T;
    for (unsigned int k = 0; k < T; k++) {
        GDV & g = g_dv(host_ptrs[k], bytes);
        if (!g.d) return 0;                 /* allocation failed -> host fallback */
        /* A2a: host is authoritative here (mul still does D2H), so always upload
         * the current host data before reducing — this guarantees the device
         * reduce operates on exactly the data the host comm would, regardless of
         * any stale device-buffer state outside the krylov loop. */
        CUCHECK(cudaMemcpy(g.d, host_ptrs[k], bytes, cudaMemcpyHostToDevice));
        pk.p[k] = g.d;
    }

    int tpb = 256; size_t blk = (words + tpb - 1) / tpb;
    vecreduce_inplace<<<blk, tpb>>>(pk, words);
    vecbroadcast_n<<<blk, tpb>>>(pk, words);
    CUCHECK(cudaGetLastError());

    /* A2a: keep the host buffers authoritative by writing the result back, and
     * do NOT trust the device copy afterwards (current=false) — outside the
     * krylov main loop (prep/secure/twist) the host buffer can be overwritten
     * without an invalidation, so a later skip-H2D must not reuse this device
     * data. The residency win (A2b) keeps current=true + host_dirty and relies on
     * full host-read sync coverage. */
    for (unsigned int k = 0; k < T; k++) {
        GDV & g = g_dv(host_ptrs[k], bytes);
        CUCHECK(cudaMemcpy(host_ptrs[k], g.d, bytes, cudaMemcpyDeviceToHost));
        g.current = false; g.host_dirty = false;
    }
    return 1;
}

/* Materialise a device-resident vector to host if the device copy is newer. */
int gpu_sync_to_host_impl(void const * host_ptr)
{
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_pool.find(host_ptr);
    if (it == g_pool.end() || !it->second.host_dirty || !it->second.d) return 0;
    CUCHECK(cudaMemcpy((void *) host_ptr, it->second.d, it->second.bytes,
                       cudaMemcpyDeviceToHost));
    it->second.host_dirty = false;
    return 1;
}

/* ---- low-level device-buffer ops for the 2D comm port (reduce+broadcast) ----
 * Each mirrors one host vec operation on the device-resident buffers at identical
 * byte offsets, so the comm result is bit-for-bit the host comm's. */

/* dst[off..+len) = XOR over k<nsrc of src_k[off..+len), in 64-bit words. */
__global__ void xor_block_kernel(PtrPack src, uint64_t * dst,
                                 size_t off_words, size_t words) {
    size_t g = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= words) return;
    uint64_t acc = 0;
    for (unsigned int k = 0; k < src.T; k++) acc ^= src.p[k][off_words + g];
    dst[off_words + g] = acc;
}

int gpu_dev_xor_block_impl(void * dst_host, void * const * src_hosts,
                           unsigned int nsrc, size_t off_bytes,
                           size_t len_bytes, size_t buf_bytes)
{
    if (nsrc == 0 || nsrc > GPU_COMM_MAX_SIB) return 0;
    GDV & gd = g_dv(dst_host, buf_bytes);
    if (!gd.d) return 0;
    PtrPack src; src.T = nsrc;
    for (unsigned int k = 0; k < nsrc; k++) {
        GDV & gs = g_dv(src_hosts[k], buf_bytes);
        if (!gs.d) return 0;
        src.p[k] = gs.d;
    }
    size_t words = len_bytes / sizeof(uint64_t);
    size_t off_words = off_bytes / sizeof(uint64_t);
    int tpb = 256; size_t blk = (words + tpb - 1) / tpb;
    xor_block_kernel<<<blk, tpb>>>(src, gd.d, off_words, words);
    CUCHECK(cudaGetLastError());
    return 1;
}

int gpu_dev_copy_block_impl(void * dst_host, size_t dst_off_bytes,
                            void const * src_host, size_t src_off_bytes,
                            size_t len_bytes, size_t dst_buf_bytes,
                            size_t src_buf_bytes)
{
    GDV & gd = g_dv(dst_host, dst_buf_bytes);
    GDV & gs = g_dv(src_host, src_buf_bytes);
    if (!gd.d || !gs.d) return 0;
    CUCHECK(cudaMemcpy((char *) gd.d + dst_off_bytes,
                       (const char *) gs.d + src_off_bytes,
                       len_bytes, cudaMemcpyDeviceToDevice));
    return 1;
}

int gpu_dev_upload_impl(void const * host, size_t buf_bytes)
{
    GDV & g = g_dv(host, buf_bytes);
    if (!g.d) return 0;
    CUCHECK(cudaMemcpy(g.d, host, buf_bytes, cudaMemcpyHostToDevice));
    g.current = true; g.host_dirty = false;
    return 1;
}

int gpu_dev_download_impl(void * host, size_t buf_bytes)
{
    GDV & g = g_dv(host, buf_bytes);
    if (!g.d) return 0;
    CUCHECK(cudaDeviceSynchronize());
    CUCHECK(cudaMemcpy(host, g.d, buf_bytes, cudaMemcpyDeviceToHost));
    g.current = false; g.host_dirty = false;
    return 1;
}

int gpu_dev_sync_impl(void) { CUCHECK(cudaDeviceSynchronize()); return 1; }

int gpu_dev_ensure_impl(void const * host, size_t buf_bytes)
{
    GDV & g = g_dv(host, buf_bytes);   /* allocates/grows under g_mtx, no data op */
    return g.d != nullptr;
}

int gpu_dev_mark_resident_impl(void const * host, size_t buf_bytes)
{
    GDV & g = g_dv(host, buf_bytes);
    if (!g.d) return 0;
    g.current = true;       /* device buffer is the authoritative copy ... */
    g.host_dirty = true;    /* ... and the host copy is now stale */
    return 1;
}

/* ---- GPU x_dotprod (gather the BW sequence off a device-resident vector) ---- */
std::map<const void *, std::pair<uint32_t *, size_t>> g_xv;   /* uploaded x-index cache */
uint32_t * g_xv_dev(const uint32_t * host, size_t n) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto & e = g_xv[host];
    if (e.second < n) {
        if (e.first) cudaFree(e.first);
        if (cudaMalloc(&e.first, n * sizeof(uint32_t)) != cudaSuccess) { e.first = nullptr; e.second = 0; return nullptr; }
        e.second = n;
        if (cudaMemcpy(e.first, host, n * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    }
    return e.first;
}

/* one thread per output row j in [j0,j1): dst[(j-j0)] = XOR over the nx sparse
 * positions i=xv[j*nx+t] in [vi0,vi1) of v[i - v_i0]. Mirrors xdotprod.cpp (GF2). */
template<int K>
__global__ void xdot_kernel(uint64_t * dst, const uint32_t * xv,
                            unsigned int j0, unsigned int j1, unsigned int nx,
                            const uint64_t * v, unsigned int v_i0,
                            unsigned int vi0, unsigned int vi1)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + j0;
    if (j >= j1) return;
    uint64_t acc[K];
#pragma unroll
    for (int k = 0; k < K; k++) acc[k] = 0;
    for (unsigned int t = 0; t < nx; t++) {
        uint32_t i = xv[(size_t) j * nx + t];
        if (i < vi0 || i >= vi1) continue;
        const uint64_t * e = v + (size_t) (i - v_i0) * K;
#pragma unroll
        for (int k = 0; k < K; k++) acc[k] ^= e[k];
    }
    uint64_t * d = dst + (size_t) (j - j0) * K;
#pragma unroll
    for (int k = 0; k < K; k++) d[k] = acc[k];
}

int gpu_x_dotprod_impl(void * dst, uint32_t const * xv,
                       unsigned int j0, unsigned int j1, unsigned int nx,
                       void const * v_host, size_t v_bytes,
                       unsigned int v_i0, unsigned int vi0, unsigned int vi1, int K)
{
    if (j1 <= j0 || (K != 1 && K != 2 && K != 4)) return 0;
    GDV & g = g_dv(v_host, v_bytes);
    if (!g.d || !g.current) return 0;        /* need a device-resident v */
    uint32_t * xvd = g_xv_dev(xv, (size_t) j1 * nx);
    if (!xvd) return 0;
    unsigned int m = j1 - j0;

    /* per-thread device scratch for the (small) result, grown as needed */
    static thread_local uint64_t * scratch = nullptr;
    static thread_local size_t scratch_n = 0;
    if (scratch_n < (size_t) m * K) {
        if (scratch) cudaFree(scratch);
        if (cudaMalloc(&scratch, (size_t) m * K * sizeof(uint64_t)) != cudaSuccess) { scratch = nullptr; scratch_n = 0; return 0; }
        scratch_n = (size_t) m * K;
    }

    int tpb = 64; int blk = (int) ((m + tpb - 1) / tpb);
    switch (K) {
        case 1: xdot_kernel<1><<<blk, tpb>>>(scratch, xvd, j0, j1, nx, g.d, v_i0, vi0, vi1); break;
        case 2: xdot_kernel<2><<<blk, tpb>>>(scratch, xvd, j0, j1, nx, g.d, v_i0, vi0, vi1); break;
        case 4: xdot_kernel<4><<<blk, tpb>>>(scratch, xvd, j0, j1, nx, g.d, v_i0, vi0, vi1); break;
    }
    CUCHECK(cudaGetLastError());
    /* XOR the (small) device result into the host dst, matching add_and_reduce. */
    std::vector<uint64_t> tmp((size_t) m * K);
    CUCHECK(cudaMemcpy(tmp.data(), scratch, (size_t) m * K * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t * hd = (uint64_t *) dst;
    for (size_t x = 0; x < (size_t) m * K; x++) hd[x] ^= tmp[x];
    return 1;
}

/* install the hooks exactly once */
void gpu_install_hooks() {
    static bool done = false;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (done) return;
    cado_gpu_comm_reduce_bcast = gpu_comm_reduce_bcast_impl;
    cado_gpu_sync_to_host = gpu_sync_to_host_impl;
    cado_gpu_dev_xor_block = gpu_dev_xor_block_impl;
    cado_gpu_dev_copy_block = gpu_dev_copy_block_impl;
    cado_gpu_dev_upload = gpu_dev_upload_impl;
    cado_gpu_dev_download = gpu_dev_download_impl;
    cado_gpu_dev_sync = gpu_dev_sync_impl;
    cado_gpu_dev_ensure = gpu_dev_ensure_impl;
    cado_gpu_dev_mark_resident = gpu_dev_mark_resident_impl;
    cado_gpu_x_dotprod = gpu_x_dotprod_impl;
    /* residency is genuinely active only with both flags set */
    cado_gpu_residency_available =
        (getenv("CADO_GPU_VECRESIDENT") && getenv("CADO_GPU_DEVCOMM")) ? 1 : 0;
    done = true;
}

/* Multi-GPU: the standard HPC model is one MPI rank per GPU. Bind this process to
 * a GPU chosen by its node-local MPI rank (round-robin over the visible devices);
 * CADO_GPU_DEVICE overrides explicitly. cudaSetDevice is per host thread, so each
 * BWC thread calls this (they share one rank, hence one device). A no-op with a
 * single visible GPU. Composes with MPI: each rank's vectors/SpMV live on its own
 * GPU and the (host) comm carries data between ranks. */
void gpu_select_device() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) return;
    int dev = 0;
    const char * ov = getenv("CADO_GPU_DEVICE");
    if (ov) {
        dev = atoi(ov) % ndev;
    } else if (ndev > 1) {
        const char * e = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (!e) e = getenv("MV2_COMM_WORLD_LOCAL_RANK");
        if (!e) e = getenv("SLURM_LOCALID");
        if (!e) e = getenv("PMI_LOCAL_RANK");
        if (e) dev = atoi(e) % ndev;
    }
    cudaSetDevice(dev);
}
} // namespace

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
    /* device vectors live in the process-global registry (g_pool/g_pin above),
     * so the comm can reach sibling threads' buffers. With CADO_GPU_VECRESIDENT
     * set, a buffer flagged `current` lets mul() skip the H2D; default: always
     * upload (bit-identical). */
    /* split-timing accumulators (ms), used when CADO_GPU_TIMING is set */
    double t_h2d = 0, t_ker = 0, t_d2h = 0; long t_n = 0;
    /* transfer counters (CADO_GPU_STATS): how often residency skips H2D/D2H */
    long n_mul = 0, n_h2d = 0, n_d2h = 0;

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
        gpu_select_device();    /* bind this thread/rank to its GPU (multi-GPU) */
        gpu_install_hooks();    /* make comm-on-device reachable from bwc_base */
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
    GDV & sv = g_dv(xsrc, srcbytes);
    GDV & wv = g_dv(xdst, dstbytes);

    g_pin(xsrc, srcbytes);
    g_pin(xdst, dstbytes);

    static const bool resident_env = getenv("CADO_GPU_VECRESIDENT") != nullptr;
    static const bool timing = getenv("CADO_GPU_TIMING") != nullptr;
    /* Residency is only active inside the krylov inner loop (cado_gpu_residency_active,
     * set by krylov.cpp) so prep/secure/twist stay host-authoritative. When active:
     * skip H2D if the device src is current, and skip D2H entirely — the dst stays
     * device-resident (host copy marked stale for cado_gpu_sync_to_host). */
    bool const residency = resident_env && cado_gpu_residency_active;
    bool skip_h2d = residency && sv.current;     /* device copy of src is up to date */

    n_mul++;
    if (!timing) {
        if (!skip_h2d) { CUCHECK(cudaMemcpy(sv.d, xsrc, srcbytes, cudaMemcpyHostToDevice)); sv.current = true; n_h2d++; }
        launch_spmv(K, d_rp[dir], d_col[dir], sv.d, wv.d, nrows);
        CUCHECK(cudaGetLastError());
        if (residency) {
            wv.current = true; wv.host_dirty = true;   /* dst left on device, host stale */
        } else {
            CUCHECK(cudaMemcpy(xdst, wv.d, dstbytes, cudaMemcpyDeviceToHost));
            wv.current = true; wv.host_dirty = false;  /* device buffer == host buffer now */
            n_d2h++;
        }
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
    g_invalidate(hostvec);
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
    if (n_mul && getenv("CADO_GPU_STATS")) {
        fprintf(stderr, "# matmul-gpu transfers over %ld SpMV: H2D %ld (%.0f%% skipped), "
                "D2H %ld (%.0f%% skipped)\n", n_mul,
                n_h2d, 100.0 * (n_mul - n_h2d) / n_mul,
                n_d2h, 100.0 * (n_mul - n_d2h) / n_mul);
    }
    for (int i = 0; i < 2; i++) { if (d_rp[i]) cudaFree(d_rp[i]); if (d_col[i]) cudaFree(d_col[i]); }
    /* g_pool / g_pinned are process-global (shared across sibling-thread mm's);
     * they are freed at process exit rather than per-mm. */
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
