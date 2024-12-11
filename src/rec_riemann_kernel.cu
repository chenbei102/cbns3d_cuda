#include "Block3d_cuda.h"
#include "eigenvectors_roe.h"
#include "weno.h"
#include "lf_flux.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void rec_riemann_xi_kernel(const value_type* Q,
					const value_type* rho,
					const value_type* u,
					const value_type* v,
					const value_type* w,
					const value_type* p,
					const value_type* xi_x,
					const value_type* xi_y,
					const value_type* xi_z,
					const value_type* Jac,
					value_type* Ep
					) {
    
    const size_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const size_type k = blockIdx.z * blockDim.z + threadIdx.z;
    
    const size_type IM = blk_info.IM - 1;
    const size_type JM = blk_info.JM - 2;
    const size_type KM = blk_info.KM - 2;
    
    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }

    size_type idx1 = blk_info.get_idx_Ep(0, i, j, k);
    size_type idx2 = blk_info.get_idx(i  , j+1, k+1);
    size_type idx3 = blk_info.get_idx(i+1, j+1, k+1);

    value_type Jac_a = Jac[idx2];
    value_type Jac_b = Jac[idx3];

    value_type n_x = 0.5 * (xi_x[idx2] * Jac_a + xi_x[idx3] * Jac_b);
    value_type n_y = 0.5 * (xi_y[idx2] * Jac_a + xi_y[idx3] * Jac_b);
    value_type n_z = 0.5 * (xi_z[idx2] * Jac_a + xi_z[idx3] * Jac_b);

    value_type rho_L = rho[idx2];
    value_type u_L = u[idx2];
    value_type v_L = v[idx2];
    value_type w_L = w[idx2];
    value_type p_L = p[idx2];

    value_type rho_R = rho[idx3];
    value_type u_R = u[idx3];
    value_type v_R = v[idx3];
    value_type w_R = w[idx3];
    value_type p_R = p[idx3];

    static const size_type NEQ = blk_info.NEQ;
    
    value_type R_l[NEQ][NEQ];
    value_type L_l[NEQ][NEQ];

    calc_eigenvectors_roe(rho_L, u_L, v_L, w_L, p_L, rho_R, u_R, v_R, w_R, p_R,
			  n_x, n_y, n_z, R_l, L_l);

    rec_weno5(Q + blk_info.get_idx_Q(0, i-2, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i-1, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i  , j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i+2, j+1, k+1),
	      R_l, L_l,
	      rho_L, u_L, v_L, w_L, p_L);

    rec_weno5(Q + blk_info.get_idx_Q(0, i+3, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i+2, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i  , j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i-1, j+1, k+1),
	      R_l, L_l,
	      rho_R, u_R, v_R, w_R, p_R);

    lf_flux(rho_R, u_R, v_R, w_R, p_R,
	    rho_L, u_L, v_L, w_L, p_L,
	    n_x, n_y, n_z, &Ep[idx1]);

  }

  __global__ void rec_riemann_eta_kernel(const value_type* Q,
					 const value_type* rho,
					 const value_type* u,
					 const value_type* v,
					 const value_type* w,
					 const value_type* p,
					 const value_type* eta_x,
					 const value_type* eta_y,
					 const value_type* eta_z,
					 const value_type* Jac,
					 value_type* Fp
					 ) {
    
    const size_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const size_type k = blockIdx.z * blockDim.z + threadIdx.z;
    
    const size_type IM = blk_info.IM - 2;
    const size_type JM = blk_info.JM - 1;
    const size_type KM = blk_info.KM - 2;
    
    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }

    size_type idx1 = blk_info.get_idx_Fp(0, i, j, k);
    size_type idx2 = blk_info.get_idx(i+1, j  , k+1);
    size_type idx3 = blk_info.get_idx(i+1, j+1, k+1);

    value_type Jac_a = Jac[idx2];
    value_type Jac_b = Jac[idx3];

    value_type n_x = 0.5 * (eta_x[idx2] * Jac_a + eta_x[idx3] * Jac_b);
    value_type n_y = 0.5 * (eta_y[idx2] * Jac_a + eta_y[idx3] * Jac_b);
    value_type n_z = 0.5 * (eta_z[idx2] * Jac_a + eta_z[idx3] * Jac_b);

    value_type rho_L = rho[idx2];
    value_type u_L = u[idx2];
    value_type v_L = v[idx2];
    value_type w_L = w[idx2];
    value_type p_L = p[idx2];

    value_type rho_R = rho[idx3];
    value_type u_R = u[idx3];
    value_type v_R = v[idx3];
    value_type w_R = w[idx3];
    value_type p_R = p[idx3];

    static const size_type NEQ = blk_info.NEQ;
    
    value_type R_l[NEQ][NEQ];
    value_type L_l[NEQ][NEQ];

    calc_eigenvectors_roe(rho_L, u_L, v_L, w_L, p_L, rho_R, u_R, v_R, w_R, p_R,
			  n_x, n_y, n_z, R_l, L_l);

    rec_weno5(Q + blk_info.get_idx_Q(0, i+1, j-2, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j-1, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j  , k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j+2, k+1),
	      R_l, L_l,
	      rho_L, u_L, v_L, w_L, p_L);

    rec_weno5(Q + blk_info.get_idx_Q(0, i+1, j+3, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j+2, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j  , k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j-1, k+1),
	      R_l, L_l,
	      rho_R, u_R, v_R, w_R, p_R);

    lf_flux(rho_R, u_R, v_R, w_R, p_R,
	    rho_L, u_L, v_L, w_L, p_L,
	    n_x, n_y, n_z, &Fp[idx1]);

  }

  __global__ void rec_riemann_zeta_kernel(const value_type* Q,
					  const value_type* rho,
					  const value_type* u,
					  const value_type* v,
					  const value_type* w,
					  const value_type* p,
					  const value_type* zeta_x,
					  const value_type* zeta_y,
					  const value_type* zeta_z,
					  const value_type* Jac,
					  value_type* Gp
					  ) {
    
    const size_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const size_type k = blockIdx.z * blockDim.z + threadIdx.z;
    
    const size_type IM = blk_info.IM - 2;
    const size_type JM = blk_info.JM - 2;
    const size_type KM = blk_info.KM - 1;
    
    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }

    size_type idx1 = blk_info.get_idx_Gp(0, i, j, k);
    size_type idx2 = blk_info.get_idx(i+1, j+1, k  );
    size_type idx3 = blk_info.get_idx(i+1, j+1, k+1);

    value_type Jac_a = Jac[idx2];
    value_type Jac_b = Jac[idx3];

    value_type n_x = 0.5 * (zeta_x[idx2] * Jac_a + zeta_x[idx3] * Jac_b);
    value_type n_y = 0.5 * (zeta_y[idx2] * Jac_a + zeta_y[idx3] * Jac_b);
    value_type n_z = 0.5 * (zeta_z[idx2] * Jac_a + zeta_z[idx3] * Jac_b);

    value_type rho_L = rho[idx2];
    value_type u_L = u[idx2];
    value_type v_L = v[idx2];
    value_type w_L = w[idx2];
    value_type p_L = p[idx2];

    value_type rho_R = rho[idx3];
    value_type u_R = u[idx3];
    value_type v_R = v[idx3];
    value_type w_R = w[idx3];
    value_type p_R = p[idx3];

    static const size_type NEQ = blk_info.NEQ;
    
    value_type R_l[NEQ][NEQ];
    value_type L_l[NEQ][NEQ];

    calc_eigenvectors_roe(rho_L, u_L, v_L, w_L, p_L, rho_R, u_R, v_R, w_R, p_R,
			  n_x, n_y, n_z, R_l, L_l);

    rec_weno5(Q + blk_info.get_idx_Q(0, i+1, j+1, k-2),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k-1),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k  ),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k+2),
	      R_l, L_l,
	      rho_L, u_L, v_L, w_L, p_L);

    rec_weno5(Q + blk_info.get_idx_Q(0, i+1, j+1, k+3),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k+2),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k+1),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k  ),
	      Q + blk_info.get_idx_Q(0, i+1, j+1, k-1),
	      R_l, L_l,
	      rho_R, u_R, v_R, w_R, p_R);

    lf_flux(rho_R, u_R, v_R, w_R, p_R,
	    rho_L, u_L, v_L, w_L, p_L,
	    n_x, n_y, n_z, &Gp[idx1]);

  }

}
