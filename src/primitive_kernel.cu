#include "Block3d_cuda.h"
#include "primitive_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void primitive_kernel(const value_type* Q,
				   value_type* rho,
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

    const size_type idx1 = blk_info.get_idx_u(i, j, k);
    const size_type idx2 = blk_info.get_idx_Qa(0, i, j, k);

    const value_type rr = Q[idx2  ];
    const value_type uu = Q[idx2+1] / rr;
    const value_type vv = Q[idx2+2] / rr;
    const value_type ww = Q[idx2+3] / rr;
    const value_type pp = blk_info.gam1 * (Q[idx2+4] - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

    rho[idx1] = rr;	
    u[idx1] = uu;
    v[idx1] = vv;
    w[idx1] = ww;
    p[idx1] = pp;

    T[idx1] = blk_info.gM2 * pp / rr;
    
  }

}
