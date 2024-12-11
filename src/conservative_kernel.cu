#include "Block3d_cuda.h"
#include "conservative_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void conservative_kernel(const value_type* rho,
				      const value_type* u,
				      const value_type* v,
				      const value_type* w,
				      const value_type* p,
				      value_type* Q
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

    const size_type idx1 = blk_info.get_idx_u(i, j, k);
    const size_type idx2 = blk_info.get_idx_Qa(0, i, j, k);

    const value_type rr = rho[idx1];
    const value_type uu = u[idx1];
    const value_type vv = v[idx1];
    const value_type ww = w[idx1];
    const value_type pp = p[idx1];

    Q[idx2  ] = rr;
    Q[idx2+1] = rr * uu;
    Q[idx2+2] = rr * vv;
    Q[idx2+3] = rr * ww;
    Q[idx2+4] = pp / blk_info.gam1 + 0.5 * rr * (uu * uu + vv * vv + ww * ww);
    
  }

}
