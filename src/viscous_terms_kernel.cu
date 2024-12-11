#include "Block3d_cuda.h"
#include "viscous_terms_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void viscous_terms_kernel(const value_type* T,
				       const value_type* xi_x,
				       const value_type* xi_y,
				       const value_type* xi_z,
				       const value_type* eta_x,
				       const value_type* eta_y,
				       const value_type* eta_z,
				       const value_type* zeta_x,
				       const value_type* zeta_y,
				       const value_type* zeta_z,
				       const value_type* u_xi,
				       const value_type* u_eta,
				       const value_type* u_zeta,
				       const value_type* v_xi,
				       const value_type* v_eta,
				       const value_type* v_zeta,
				       const value_type* w_xi,
				       const value_type* w_eta,
				       const value_type* w_zeta,
				       const value_type* T_xi,
				       const value_type* T_eta,
				       const value_type* T_zeta,
				       value_type* mu,
				       value_type* tau_xx,
				       value_type* tau_yy,
				       value_type* tau_zz,
				       value_type* tau_xy,
				       value_type* tau_xz,
				       value_type* tau_yz,
				       value_type* q_x,
				       value_type* q_y,
				       value_type* q_z
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

    const value_type T_l = T[idx1];
    const value_type C_T_inf = blk_info.C_T_inf;
    
    value_type mu_l = (1.0 + C_T_inf) / (T_l + C_T_inf) * std::sqrt(T_l * T_l * T_l);

    mu[idx1] = mu_l;

    const value_type xi_x_l = xi_x[idx1];
    const value_type xi_y_l = xi_y[idx1];
    const value_type xi_z_l = xi_z[idx1];
    const value_type eta_x_l = eta_x[idx1];
    const value_type eta_y_l = eta_y[idx1];
    const value_type eta_z_l = eta_z[idx1];
    const value_type zeta_x_l = zeta_x[idx1];
    const value_type zeta_y_l = zeta_y[idx1];
    const value_type zeta_z_l = zeta_z[idx1];

    value_type u_xi_l = u_xi[idx1];
    value_type u_eta_l = u_eta[idx1];
    value_type u_zeta_l = u_zeta[idx1];
    const value_type v_xi_l = v_xi[idx1];
    const value_type v_eta_l = v_eta[idx1];
    const value_type v_zeta_l = v_zeta[idx1];
    const value_type w_xi_l = w_xi[idx1];
    const value_type w_eta_l = w_eta[idx1];
    const value_type w_zeta_l = w_zeta[idx1];

    mu_l *= blk_info.Re_inv;

    tau_xx[idx1] = (2.0/3.0)*mu_l*(2*eta_x_l*u_eta_l - eta_y_l*v_eta_l - eta_z_l*w_eta_l + 2*u_xi_l*xi_x_l + 2*u_zeta_l*zeta_x_l - v_xi_l*xi_y_l - v_zeta_l*zeta_y_l - w_xi_l*xi_z_l - w_zeta_l*zeta_z_l);
    tau_xy[idx1] = mu_l*(eta_x_l*v_eta_l + eta_y_l*u_eta_l + u_xi_l*xi_y_l + u_zeta_l*zeta_y_l + v_xi_l*xi_x_l + v_zeta_l*zeta_x_l);
    tau_yy[idx1] = (2.0/3.0)*mu_l*(-eta_x_l*u_eta_l + 2*eta_y_l*v_eta_l - eta_z_l*w_eta_l - u_xi_l*xi_x_l - u_zeta_l*zeta_x_l + 2*v_xi_l*xi_y_l + 2*v_zeta_l*zeta_y_l - w_xi_l*xi_z_l - w_zeta_l*zeta_z_l);
    tau_xz[idx1] = mu_l*(eta_x_l*w_eta_l + eta_z_l*u_eta_l + u_xi_l*xi_z_l + u_zeta_l*zeta_z_l + w_xi_l*xi_x_l + w_zeta_l*zeta_x_l);
    tau_yz[idx1] = mu_l*(eta_y_l*w_eta_l + eta_z_l*v_eta_l + v_xi_l*xi_z_l + v_zeta_l*zeta_z_l + w_xi_l*xi_y_l + w_zeta_l*zeta_y_l);
    tau_zz[idx1] = (2.0/3.0)*mu_l*(-eta_x_l*u_eta_l - eta_y_l*v_eta_l + 2*eta_z_l*w_eta_l - u_xi_l*xi_x_l - u_zeta_l*zeta_x_l - v_xi_l*xi_y_l - v_zeta_l*zeta_y_l + 2*w_xi_l*xi_z_l + 2*w_zeta_l*zeta_z_l);

    mu_l *= -blk_info.Pr_inv * blk_info.gam1_inv / blk_info.Mach2;

    u_xi_l = T_xi[idx1];
    u_eta_l = T_eta[idx1];
    u_zeta_l = T_zeta[idx1];

    q_x[idx1] = mu_l * (xi_x_l * u_xi_l + eta_x_l * u_eta_l + zeta_x_l * u_zeta_l); 
    q_y[idx1] = mu_l * (xi_y_l * u_xi_l + eta_y_l * u_eta_l + zeta_y_l * u_zeta_l);
    q_z[idx1] = mu_l * (xi_z_l * u_xi_l + eta_z_l * u_eta_l + zeta_z_l * u_zeta_l); 
    
  }

}
