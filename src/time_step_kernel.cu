#include "Block3d_cuda.h"
#include "time_step_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void time_step_kernel(const value_type* rho,
				   const value_type* u,
				   const value_type* v,
				   const value_type* w,
				   const value_type* p,
				   const value_type* mu,
				   const value_type* xi_x,
				   const value_type* xi_y,
				   const value_type* xi_z,
				   const value_type* eta_x,
				   const value_type* eta_y,
				   const value_type* eta_z,
				   const value_type* zeta_x,
				   const value_type* zeta_y,
				   const value_type* zeta_z,
				   value_type* dt
				   ) {
    
    const size_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const size_type k = blockIdx.z * blockDim.z + threadIdx.z;
    
    const size_type IM = blk_info.IM;
    const size_type JM = blk_info.JM;
    const size_type KM = blk_info.KM;
    
    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }

    size_type idx1 = blk_info.get_idx(i, j, k);
    
    value_type rr = rho[idx1];
    value_type uu = u[idx1];
    value_type vv = v[idx1];
    value_type ww = w[idx1];
    value_type pp = p[idx1];

    value_type c = std::sqrt(blk_info.gamma * pp / rr);

#ifndef IS_INVISCID
    value_type factor_v = 2.0 * mu[idx1] * blk_info.C_dt_v * std::sqrt(uu*uu + vv*vv + ww*ww) / c / rr;
#endif

    value_type n_x = xi_x[idx1];
    value_type n_y = xi_y[idx1];
    value_type n_z = xi_z[idx1];

    value_type qn = uu * n_x + vv * n_y + ww * n_z;
    value_type n_abs = std::sqrt(n_x * n_x + n_y * n_y + n_z * n_z);

    value_type t1 = std::abs(qn) + c * n_abs;
#ifndef IS_INVISCID
    t1 += n_abs * n_abs * factor_v;
#endif

    n_x = eta_x[idx1];
    n_y = eta_y[idx1];
    n_z = eta_z[idx1];

    qn = uu * n_x + vv * n_y + ww * n_z;
    n_abs = std::sqrt(n_x * n_x + n_y * n_y + n_z * n_z);

    value_type t2 = std::abs(qn) + c * n_abs;
#ifndef IS_INVISCID
    t2 += n_abs * n_abs * factor_v;
#endif

    n_x = zeta_x[idx1];
    n_y = zeta_y[idx1];
    n_z = zeta_z[idx1];

    qn = uu * n_x + vv * n_y + ww * n_z;
    n_abs = std::sqrt(n_x * n_x + n_y * n_y + n_z * n_z);

    value_type t3 = std::abs(qn) + c * n_abs;
#ifndef IS_INVISCID
    t3 += n_abs * n_abs * factor_v;
#endif

    idx1 = blk_info.get_idx_x(i, j, k);
    dt[idx1] = t1 + t2 + t3;

  }

}
