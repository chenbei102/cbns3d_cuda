#include "Block3d_cuda.h"
#include "viscous_flux_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void viscous_flux_kernel(const value_type* u,
				      const value_type* v,
				      const value_type* w,
				      const value_type* Jac,
				      const value_type* xi_x,
				      const value_type* xi_y,
				      const value_type* xi_z,
				      const value_type* eta_x,
				      const value_type* eta_y,
				      const value_type* eta_z,
				      const value_type* zeta_x,
				      const value_type* zeta_y,
				      const value_type* zeta_z,
				      const value_type* tau_xx,
				      const value_type* tau_yy,
				      const value_type* tau_zz,
				      const value_type* tau_xy,
				      const value_type* tau_xz,
				      const value_type* tau_yz,
				      const value_type* q_x,
				      const value_type* q_y,
				      const value_type* q_z,
				      value_type* Ev,
				      value_type* Fv,
				      value_type* Gv
				      ) {
    
    const index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const index_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const index_type k = blockIdx.z * blockDim.z + threadIdx.z;

    const size_type IM = blk_info.IM_G - 2;
    const size_type JM = blk_info.JM_G - 2;
    const size_type KM = blk_info.KM_G - 2;
    
    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }

    const index_type NG = blk_info.NG;
    
    const size_type idx1 = blk_info.get_idx(i-NG+1, j-NG+1, k-NG+1);
    const size_type idx2 = blk_info.get_idx_Ev(0, i, j, k);

    value_type Jac_a = Jac[idx1];

    value_type uu = u[idx1];
    value_type vv = v[idx1];
    value_type ww = w[idx1];

    value_type tau_xx_l = tau_xx[idx1];
    value_type tau_yy_l = tau_yy[idx1];
    value_type tau_zz_l = tau_zz[idx1];
    value_type tau_xy_l = tau_xy[idx1];
    value_type tau_xz_l = tau_xz[idx1];
    value_type tau_yz_l = tau_yz[idx1];

    // ---------------------------------------------------------------------
    // Viscous flux in xi-direction 

    value_type n_x = xi_x[idx1] * Jac_a;
    value_type n_y = xi_y[idx1] * Jac_a;
    value_type n_z = xi_z[idx1] * Jac_a;

    Ev[idx2  ] = n_x * tau_xx_l + n_y * tau_xy_l + n_z * tau_xz_l;
    Ev[idx2+1] = n_x * tau_xy_l + n_y * tau_yy_l + n_z * tau_yz_l;
    Ev[idx2+2] = n_x * tau_xz_l + n_y * tau_yz_l + n_z * tau_zz_l;
    Ev[idx2+3] = n_x * ( uu * tau_xx_l + vv * tau_xy_l + ww * tau_xz_l - q_x[idx1] )
      + n_y * ( uu * tau_xy_l + vv * tau_yy_l + ww * tau_yz_l - q_y[idx1] )
      + n_z * ( uu * tau_xz_l + vv * tau_yz_l + ww * tau_zz_l - q_z[idx1] );

    // ---------------------------------------------------------------------
    // Viscous flux in eta-direction

    n_x = eta_x[idx1] * Jac_a;
    n_y = eta_y[idx1] * Jac_a;
    n_z = eta_z[idx1] * Jac_a;

    Fv[idx2  ] = n_x * tau_xx_l + n_y * tau_xy_l + n_z * tau_xz_l;
    Fv[idx2+1] = n_x * tau_xy_l + n_y * tau_yy_l + n_z * tau_yz_l;
    Fv[idx2+2] = n_x * tau_xz_l + n_y * tau_yz_l + n_z * tau_zz_l;
    Fv[idx2+3] = n_x * ( uu * tau_xx_l + vv * tau_xy_l + ww * tau_xz_l - q_x[idx1] )
      + n_y * ( uu * tau_xy_l + vv * tau_yy_l + ww * tau_yz_l - q_y[idx1] )
      + n_z * ( uu * tau_xz_l + vv * tau_yz_l + ww * tau_zz_l - q_z[idx1] );
	
    // ---------------------------------------------------------------------
    // Viscous flux in zeta-direction

    n_x = zeta_x[idx1] * Jac_a;
    n_y = zeta_y[idx1] * Jac_a;
    n_z = zeta_z[idx1] * Jac_a;

    Gv[idx2  ] = n_x * tau_xx_l + n_y * tau_xy_l + n_z * tau_xz_l;
    Gv[idx2+1] = n_x * tau_xy_l + n_y * tau_yy_l + n_z * tau_yz_l;
    Gv[idx2+2] = n_x * tau_xz_l + n_y * tau_yz_l + n_z * tau_zz_l;
    Gv[idx2+3] = n_x * ( uu * tau_xx_l + vv * tau_xy_l + ww * tau_xz_l - q_x[idx1] )
      + n_y * ( uu * tau_xy_l + vv * tau_yy_l + ww * tau_yz_l - q_y[idx1] )
      + n_z * ( uu * tau_xz_l + vv * tau_yz_l + ww * tau_zz_l - q_z[idx1] );

  }

}
