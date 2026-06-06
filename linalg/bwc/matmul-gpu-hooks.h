#ifndef CADO_MATMUL_GPU_HOOKS_H
#define CADO_MATMUL_GPU_HOOKS_H

/* Cross-TU hooks for keeping BWC vectors device-resident across the comm
 * (v3.1.0-modern, Track 2.2 — comm-on-device, the full-vector-residency win).
 *
 * The BWC comm (mmt_vec_allreduce, in matmul_top_comm.cpp, part of bwc_base)
 * lives in a CUDA-free translation unit. The GPU matmul backend (matmul-gpu.cu)
 * installs the function pointers below in its constructor; bwc_base calls them
 * through the pointers, so the comm can run the GF(2) reduce/broadcast directly
 * on the device-resident sibling vectors instead of bouncing through host memory
 * every iteration. When no GPU backend is loaded the pointers stay null and the
 * comm takes its ordinary host path — zero behavioural change. The whole device
 * path is additionally gated at the call site on CADO_GPU_VECRESIDENT and on the
 * single-node case (no MPI); see matmul_top_comm.cpp. */

#include <cstddef>

/* Single-node GF(2) allreduce on device: given the host pointers of the T
 * sibling vectors (each `bytes` long), reduce them with XOR and broadcast the
 * result back over all T, entirely on the device-resident copies. Returns 1 if
 * it handled the comm on device (all T buffers were present & current in the
 * device registry), 0 if the caller must fall back to the host path. On success
 * the device copies are left current and the host copies are marked stale (to be
 * materialised on demand by gpu_sync_to_host). */
extern int (*cado_gpu_comm_reduce_bcast)(void * const * host_ptrs,
                                         unsigned int T, size_t bytes);

/* Materialise a device-resident vector back to its host buffer if the device
 * copy is newer (a no-op otherwise). Called at host-read sites (inner products,
 * checkpoints, twist) so a resident vector is correct on the host when actually
 * read. Returns 1 if a copy happened, 0 otherwise. */
extern int (*cado_gpu_sync_to_host)(void const * host_ptr);

/* ---- low-level device-buffer ops for the 2D comm port (reduce+broadcast) ----
 * These mirror, on the device-resident copies, the exact host operations of
 * mmt_vec_reduce + mmt_vec_broadcast at identical offsets, so the result is
 * bit-for-bit the host comm's. All sizes/offsets are in BYTES; buf_bytes is the
 * full buffer length of the host pointer (its registry key). All return 1 on
 * success, 0 to force a host fallback. The device kernels run on the default
 * stream; callers serialize across threads (serialize_threads) and must call
 * cado_gpu_dev_sync once per phase so cross-thread reads see completed writes. */

/* dst_buf[off .. off+len) = XOR over k<nsrc of src_bufs[k][off .. off+len).
 * (mirrors mmt_vec_reduce_inner's per-thread vec_add_and_reduce; src list
 * includes the accumulator dst as src_bufs[0].) */
extern int (*cado_gpu_dev_xor_block)(void * dst_host, void * const * src_hosts,
                                     unsigned int nsrc, size_t off_bytes,
                                     size_t len_bytes, size_t buf_bytes);

/* dst_buf[dst_off .. +len) = src_buf[src_off .. +len)  (device-to-device). */
extern int (*cado_gpu_dev_copy_block)(void * dst_host, size_t dst_off_bytes,
                                      void const * src_host, size_t src_off_bytes,
                                      size_t len_bytes,
                                      size_t dst_buf_bytes, size_t src_buf_bytes);

/* H2D the whole host buffer into its device copy (mark device current). */
extern int (*cado_gpu_dev_upload)(void const * host, size_t buf_bytes);

/* D2H the whole device copy back into the host buffer (no-trust: current=false).
 * Implies a device sync first. */
extern int (*cado_gpu_dev_download)(void * host, size_t buf_bytes);

/* Block until all issued device work has completed (one call per comm phase). */
extern int (*cado_gpu_dev_sync)(void);

/* Ensure the device buffer for `host` exists at >= buf_bytes WITHOUT copying any
 * data. Call this (barriered) before concurrent ops on a buffer that may need
 * growing, so the grow (cudaFree+cudaMalloc) cannot race a sibling thread's
 * in-flight copy into the same buffer. Returns 1 on success, 0 on alloc failure. */
extern int (*cado_gpu_dev_ensure)(void const * host, size_t buf_bytes);

/* ---- full vector residency (Track 2.2, the transfer-eliminating win) ----
 * When nonzero, the GPU backend keeps the BWC vectors device-resident across the
 * steady krylov iteration: mul() skips its H2D (src already on device) and D2H
 * (dst left on device), and the device comm skips its host upload/writeback. The
 * host copies go stale (host_dirty); host-read sites must call cado_gpu_sync_to_host
 * first. This is scoped to the krylov inner loop (set/cleared by krylov.cpp) so
 * prep/secure/twist — which overwrite host buffers without invalidation — stay on
 * the safe host-authoritative path. Requires CADO_GPU_VECRESIDENT + CADO_GPU_DEVCOMM. */
extern int cado_gpu_residency_active;

/* Mark the device buffer for `host` as the authoritative copy (current, and the
 * host copy stale) without any transfer — used by the device comm to leave its
 * result device-resident in residency mode. Returns 1 on success, 0 otherwise. */
extern int (*cado_gpu_dev_mark_resident)(void const * host, size_t buf_bytes);

#endif /* CADO_MATMUL_GPU_HOOKS_H */
