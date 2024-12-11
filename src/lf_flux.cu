#include "Block3d_cuda.h"
#include "lf_flux.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------
  
  __device__ void lf_flux(value_type rho_R,
			  value_type u_R, value_type v_R, value_type w_R,
			  value_type p_R,
			  value_type rho_L,
			  value_type u_L, value_type v_L, value_type w_L,
			  value_type p_L,
			  value_type nx, value_type ny, value_type nz,
			  value_type *flux) {

    const value_type gamma = blk_info.gamma;
    const value_type gam1 = blk_info.gam1;

    value_type c_R = std::sqrt(gamma * p_R / rho_R);
    value_type H_R = c_R * c_R / gam1 + 0.5 * (u_R * u_R + v_R * v_R + w_R * w_R);
    value_type rE_R = p_R / gam1 + 0.5 * rho_R * (u_R * u_R + v_R * v_R + w_R * w_R);

    value_type c_L = std::sqrt(gamma * p_L / rho_L);
    value_type H_L = c_L * c_L / gam1 + 0.5 * (u_L * u_L + v_L * v_L + w_L * w_L);
    value_type rE_L = p_L / gam1 + 0.5 * rho_L * (u_L * u_L + v_L * v_L + w_L * w_L);

    value_type q_R = u_R * nx + v_R * ny + w_R * nz;
    value_type q_L = u_L * nx + v_L * ny + w_L * nz;

    // ---------------------------------------------------------------------------
    value_type ws[3];
    value_type ws_max;
  
    ws[0] = std::abs(u_R) + c_R;
    ws_max = std::abs(u_L) + c_L;
    if (ws_max > ws[0]) ws[0] = ws_max;

    ws[1] = std::abs(v_R) + c_R;
    ws_max = std::abs(v_L) + c_L;
    if (ws_max > ws[1]) ws[1] = ws_max;

    ws[2] = std::abs(w_R) + c_R;
    ws_max = std::abs(w_L) + c_L;
    if (ws_max > ws[2]) ws[2] = ws_max;

    ws_max = std::abs(ws[0] * nx + ws[1] * ny + ws[2] * nz);
    // ---------------------------------------------------------------------------

    value_type diss[constant::NEQ];
    diss[0] = ws_max * (rho_R - rho_L);
    diss[1] = ws_max * (rho_R * u_R - rho_L * u_L);
    diss[2] = ws_max * (rho_R * v_R - rho_L * v_L);
    diss[3] = ws_max * (rho_R * w_R - rho_L * w_L);
    diss[4] = ws_max * (rE_R - rE_L);

    value_type f_L[constant::NEQ];
    f_L[0] = rho_L * q_L;
    f_L[1] = rho_L * q_L * u_L + p_L * nx;
    f_L[2] = rho_L * q_L * v_L + p_L * ny;
    f_L[3] = rho_L * q_L * w_L + p_L * nz;
    f_L[4] = rho_L * q_L * H_L;

    value_type f_R[constant::NEQ];
    f_R[0] = rho_R * q_R;
    f_R[1] = rho_R * q_R * u_R + p_R * nx;
    f_R[2] = rho_R * q_R * v_R + p_R * ny;
    f_R[3] = rho_R * q_R * w_R + p_R * nz;
    f_R[4] = rho_R * q_R * H_R;

    for (size_type li = 0; li < constant::NEQ; li++) {
      flux[li] = 0.5 * (f_L[li] + f_R[li] - diss[li]);
    }

  }
  
}
