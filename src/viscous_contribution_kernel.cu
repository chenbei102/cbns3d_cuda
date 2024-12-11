#include "Block3d_cuda.h"
#include "viscous_contribution_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void viscous_contribution_kernel(const value_type* Ev,
					      const value_type* Fv,
					      const value_type* Gv,
					      value_type* diff_flux_vis
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
    const size_type NG = blk_info.NG;
    
    value_type df = 0.0;    

    static const value_type coeff_c[2] = {1.0/12.0, -2.0/3.0};

    size_type idx1 = 0;    
    size_type idx2 = 0;    
    size_type idx3 = 0;    
    size_type idx4 = 0;    

    for(size_type i_eq = 0; i_eq < NEQ-1; i_eq++) {

      idx1 = blk_info.get_idx_Ev(i_eq, i+NG-3, j, k);
      idx2 = blk_info.get_idx_Ev(i_eq, i+NG-2, j, k);
      idx3 = blk_info.get_idx_Ev(i_eq, i+NG  , j, k);
      idx4 = blk_info.get_idx_Ev(i_eq, i+NG+1, j, k);

      df = coeff_c[0]*Ev[idx1] + coeff_c[1]*Ev[idx2] - coeff_c[1]*Ev[idx3] - coeff_c[0]*Ev[idx4];

      idx1 = blk_info.get_idx_Ev(i_eq, i, j+NG-3, k);
      idx2 = blk_info.get_idx_Ev(i_eq, i, j+NG-2, k);
      idx3 = blk_info.get_idx_Ev(i_eq, i, j+NG  , k);
      idx4 = blk_info.get_idx_Ev(i_eq, i, j+NG+1, k);

      df += coeff_c[0]*Fv[idx1] + coeff_c[1]*Fv[idx2] - coeff_c[1]*Fv[idx3] - coeff_c[0]*Fv[idx4];

      idx1 = blk_info.get_idx_Ev(i_eq, i, j, k+NG-3);
      idx2 = blk_info.get_idx_Ev(i_eq, i, j, k+NG-2);
      idx3 = blk_info.get_idx_Ev(i_eq, i, j, k+NG  );
      idx4 = blk_info.get_idx_Ev(i_eq, i, j, k+NG+1);

      df += coeff_c[0]*Gv[idx1] + coeff_c[1]*Gv[idx2] - coeff_c[1]*Gv[idx3] - coeff_c[0]*Gv[idx4];

      diff_flux_vis[blk_info.get_idx_dfv(i_eq, i-1, j-1, k-1)] = df;

    }

  }

}
