#include "Block3d_cuda.h"
#include "initial_condition_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void initial_condition_kernel(value_type* rho,
					   value_type* u,
					   value_type* v,
					   value_type* w,
					   value_type* p,
					   value_type* T
					   ) {
    
    const size_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const size_type k = blockIdx.z * blockDim.z + threadIdx.z;
    
    const size_type IM = blk_info.IM_G;
    const size_type JM = blk_info.JM_G;
    const size_type KM = blk_info.KM_G;
    
    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }

    const size_type idx = blk_info.get_idx_u(i, j, k);

    const value_type AoA = blk_info.angle_attack * constant::PI / 180.0;

    rho[idx] = 1.0;
    u[idx] = std::cos(AoA);
    v[idx] = std::sin(AoA);
    w[idx] = 0.0;
    p[idx] = blk_info.p_inf;

#ifndef IS_INVISCID
    T[idx] = 1.0;
#endif
    
  }

}
