#include "Block3d_cuda.h"
#include "runge_kutta_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void rk3_kernel(const value_type dt,
			     const size_type stage,
			     const value_type* Jac,
			     const value_type* Ep,
			     const value_type* Fp,
			     const value_type* Gp,
			     const value_type* diff_flux_vis,
			     value_type* Q,
			     value_type* Q_p
			     ) {
    
    const size_type i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_type j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const size_type k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    const size_type IM = blk_info.IM - 1;
    const size_type JM = blk_info.JM - 1;
    const size_type KM = blk_info.KM - 1;
    
    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }

    const size_type NEQ = blk_info.NEQ;
    
    const value_type Jac_a = Jac[blk_info.get_idx(i, j, k)];

    size_type idx1 = 0;    
    size_type idx2 = 0;    

    value_type df = 0.0;    

    for(size_type i_eq = 0; i_eq < NEQ; i_eq++) {
      
      idx1 = blk_info.get_idx_Ep(i_eq, i, j-1, k-1);
      idx2 = blk_info.get_idx_Ep(i_eq, i-1, j-1, k-1);

      df = Ep[idx1] - Ep[idx2];

      idx1 = blk_info.get_idx_Fp(i_eq, i-1, j, k-1);
      idx2 = blk_info.get_idx_Fp(i_eq, i-1, j-1, k-1);

      df += Fp[idx1] - Fp[idx2];

      idx1 = blk_info.get_idx_Gp(i_eq, i-1, j-1, k);
      idx2 = blk_info.get_idx_Gp(i_eq, i-1, j-1, k-1);

      df += Gp[idx1] - Gp[idx2];

#ifndef IS_INVISCID
      if (i_eq > 0) df -= diff_flux_vis[blk_info.get_idx_dfv(i_eq-1, i-1, j-1, k-1)];
#endif
      
      df /= Jac_a;

      idx1 = blk_info.get_idx_Q(i_eq, i, j, k);

      if (1 == stage) {
	Q_p[idx1] = Q[idx1] - dt * df;
      } else if (2 == stage) {
	Q_p[idx1] = 0.75 * Q[idx1] + 0.25 * (Q_p[idx1] - dt * df);
      } else if (3 == stage) {
	Q[idx1] = (Q[idx1] + 2.0 * (Q_p[idx1] - dt * df)) / 3.0;
      }

    }

  }

}
