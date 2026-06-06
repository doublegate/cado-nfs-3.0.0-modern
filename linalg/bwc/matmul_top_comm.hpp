#ifndef CADO_MATMUL_TOP_COMM_HPP
#define CADO_MATMUL_TOP_COMM_HPP

#include "matmul_top_vec.hpp"

extern void mmt_vec_allreduce(mmt_vec & v);
extern void mmt_vec_broadcast(mmt_vec & v);
extern void mmt_vec_reduce(mmt_vec & w, mmt_vec & v);
extern void mmt_vec_reduce_mod_p(mmt_vec & v);
extern void mmt_vec_reduce_sameside(mmt_vec & v);

/* GPU device (comm-on-device) path for matmul_top_mul_comm = mmt_vec_reduce +
 * mmt_vec_broadcast (Track 2.2). Mirrors the host algorithm op-for-op on the
 * device-resident buffers for the single-node case. Returns 1 if it handled the
 * comm, 0 to fall back to the host path. */
extern int matmul_top_mul_comm_gpu(mmt_vec & v, mmt_vec & w);


#endif	/* MATMUL_TOP_COMM_HPP_ */
