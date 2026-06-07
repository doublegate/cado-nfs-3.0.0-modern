# Multi-node device-resident BWC via NVSHMEM / GPUDirect (Roadmap D2)

Per the roadmap, D2 is a **hardware-gated design** — documented, not committed as
unvalidated code. The reference box has one GPU and no multi-node fabric, so the
NVSHMEM/GPUDirect path cannot be validated here; the **degenerate single-rank path
is already the validated device-resident loop** (`CADO_GPU_VECRESIDENT=1`,
`product == N`, see `docs/gpu-linalg.md`). This records the concrete plan so it can
be implemented + validated when CUDA-aware-MPI hardware is available.

## The problem D2 solves

3.1.0 made the BWC working vectors **device-resident within one rank**: the
krylov/mksol/secure loop runs on device buffers, eliminating the per-iteration
PCIe round-trip (the measured ~60 % of SpMV time at c70/c80). But Block Wiedemann
at scale runs on an `nh×nv` **MPI grid** (`balancing_workhorse` splits the matrix;
each rank owns a submatrix). Every iteration the partial SpMV results are combined
across ranks by collectives in `linalg/bwc/matmul_top_comm.cpp`:

- **broadcast across the row** — `MPI_Allgather` (`matmul_top_comm.cpp:151`),
- **reduce across the column** — `MPI_Reduce_scatter` /
  `MPI_Reduce_scatter_block` (`:551`–`:573`),
- **dot products** — `MPI_Allreduce` (`:804`).

These run on **host vector buffers**. So with >1 rank, a device-resident vector
must be copied **device→host every iteration**, exchanged on the host, and copied
**host→device** again — which caps the residency win at multi-node exactly where
the matrix no longer fits one GPU and the comm matters most. D2 keeps the vectors
on the device *through* the collective.

## Design — two levels, comm on the device

### Level 1 — intra-node, on-device reduce (NVLink)

Within a node, the `nh×nv` sub-grid's ranks share GPUs over **NVLink**. Replace
the host-buffer reduce/broadcast of the *local* communicator with a **device
collective**:

- **CUDA-aware MPI** (GPUDirect P2P): pass the **device** pointer to
  `MPI_Reduce_scatter` / `MPI_Allgather` / `MPI_Allreduce`; a CUDA-aware MPI
  (OpenMPI+UCX or MPICH+CUDA) moves data GPU↔GPU over NVLink with **no host
  staging**. This is the *minimal* change — the same collective calls in
  `matmul_top_comm.cpp`, but on `vec->v_dev` instead of `vec->v`, gated by a
  runtime probe (`MPIX_Query_cuda_support()`).
- **NVSHMEM** (GPU-initiated PGAS): allocate the working vectors on the NVSHMEM
  **symmetric heap**; the reduce/broadcast become `nvshmemx_*` put/reduce issued
  **from the GPU**, which lets them be launched in the SpMV kernel's epilogue and
  overlap with compute (Level 3). Higher performance, more invasive.

### Level 2 — inter-node, RDMA boundary exchange (InfiniBand)

Across nodes there is no NVLink; data crosses **InfiniBand**. With **GPUDirect
RDMA**, the same CUDA-aware-MPI / NVSHMEM calls do device→NIC→device RDMA with no
host bounce. The decomposition (the 3.1.0 "local-device-reduce / MPI-boundary-
exchange split"): do the intra-node reduction on-device first (Level 1), so only
the **reduced per-node partial** crosses the network — minimizing the inter-node
volume to one contribution per node instead of per rank.

### Level 3 — overlap comm with the SpMV

The single biggest lever. Split each rank's local SpMV into row-chunks (the D1
`CADO_GPU_NPART` machinery already produces independent chunks on per-device
streams). As soon as chunk *k*'s partial is on the device, start its **boundary
exchange asynchronously** on a dedicated comm stream (CUDA-aware-MPI persistent
request, or NVSHMEM from the kernel) **while chunk *k+1* computes** — hiding the
network latency behind compute. This turns the per-iteration comm from a serial
tax into (ideally) free, the only way a multi-node GPU BWC beats a single big GPU
per the 3.1.0 transfer accounting.

## Where to wire it (concrete)

| collective | call site | D2 replacement |
|------------|-----------|----------------|
| broadcast (row) | `matmul_top_comm.cpp:151` `MPI_Allgather` | device-buffer Allgather (CUDA-aware) / `nvshmemx_collect` |
| reduce (col) | `matmul_top_comm.cpp:551–573` `MPI_Reduce_scatter[_block]` | device-buffer Reduce_scatter / NVSHMEM reduce |
| dotprod | `matmul_top_comm.cpp:804` `MPI_Allreduce` | device-buffer Allreduce / NVSHMEM reduce |

Gate the device path on a runtime probe (`MPIX_Query_cuda_support()` for CUDA-aware
MPI, `nvshmem_init` success for NVSHMEM); fall back to the existing host-staged
collective when unavailable, so non-GPUDirect builds/clusters are unaffected.

## Honest scope & validation plan

- **HW-gated.** Needs CUDA-aware MPI (OpenMPI+UCX / MPICH+CUDA), a GPUDirect-RDMA
  fabric (NVLink intra-node, IB inter-node), and ≥2 GPUs / ≥2 nodes. None of this
  exists on the reference box (1 RTX 3090, no IB), so the NVSHMEM/GPUDirect path is
  **not validatable here** and **no unvalidated NVSHMEM code is committed** (per the
  fork's HW-gating ethos).
- **Degenerate path already validated.** Single rank (`nh=nv=1`) has no cross-rank
  collective, so the resident loop is exactly the 3.1.0 `CADO_GPU_VECRESIDENT`
  path — `product == N`, validated. D1 (this cycle) added the per-device-stream
  partition that Level 3's overlap reuses.
- **Validation plan on real HW:** (1) build CUDA-aware MPI, confirm
  `MPIX_Query_cuda_support()`; (2) switch the three collectives to device buffers,
  re-run a 2-rank c90 → `product == N` (correctness); (3) add the async overlap,
  measure krylov iter time vs the host-staged baseline on ≥2 nodes; (4) only then
  ship, with the measured number — never an unvalidated claim.

## Sources

- 3.1.0 residency analysis + the MPI-boundary decision: `docs/gpu-linalg.md`
  ("Where the time goes", "device-residency control plane").
- BWC comm layer: `linalg/bwc/matmul_top_comm.cpp` (the collectives above).
- NVSHMEM / GPUDirect RDMA: NVIDIA developer docs; RDMA SpMV on GPUs
  (arXiv:2311.18141). Multi-GPU BWC over GF(2): Schmidt et al., Concurrency &
  Computation 2013 (DOI 10.1002/cpe.2896).
