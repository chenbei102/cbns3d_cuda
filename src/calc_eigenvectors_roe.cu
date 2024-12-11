#include "Block3d_cuda.h"
#include "eigenvectors_roe.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __device__ void calc_eigenvectors_roe(const value_type rho_L,
					const value_type u_L,
					const value_type v_L,
					const value_type w_L,
					const value_type p_L,
					const value_type rho_R,
					const value_type u_R,
					const value_type v_R,
					const value_type w_R,
					const value_type p_R,
					value_type n_x, value_type n_y, value_type n_z, 
					value_type R[constant::NEQ][constant::NEQ],
					value_type L[constant::NEQ][constant::NEQ]
					) {

    const value_type gamma = blk_info.gamma;
    const value_type gam1 = blk_info.gam1;
    
    value_type l_x = 0.0;
    value_type l_y = n_z;
    value_type l_z = -n_y;

    value_type m_x, m_y, m_z;
  
    value_type n_abs = std::sqrt(l_x * l_x + l_y * l_y + l_z * l_z);

    if (n_abs > 1.0e-8) {
      m_x = -n_y*n_y - n_z*n_z;
      m_y = n_x*n_y;
      m_z = n_x*n_z;
    } else {
      l_x = -n_z;
      l_y = 0.0;
      l_z = n_x;
    
      m_x = n_x*n_y;
      m_y = -n_x*n_x - n_z*n_z;
      m_z = n_y*n_z;
    }

    n_abs = std::sqrt(n_x * n_x + n_y * n_y + n_z * n_z);
    n_x /= n_abs;
    n_y /= n_abs;
    n_z /= n_abs;

    n_abs = std::sqrt(l_x * l_x + l_y * l_y + l_z * l_z);
    l_x /= n_abs;
    l_y /= n_abs;
    l_z /= n_abs;

    n_abs = std::sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
    m_x /= n_abs;
    m_y /= n_abs;
    m_z /= n_abs;

    value_type c_L = std::sqrt(gamma * p_L / rho_L);
    value_type H_L = c_L * c_L / gam1 + 0.5 * (u_L * u_L + v_L * v_L + w_L * w_L);

    value_type c_R = std::sqrt(gamma * p_R / rho_R);
    value_type H_R = c_R * c_R / gam1 + 0.5 * (u_R * u_R + v_R * v_R + w_R * w_R);

    value_type ratio = std::sqrt(rho_R / rho_L);

    value_type uu = (u_L + ratio * u_R) / (1.0 + ratio);
    value_type vv = (v_L + ratio * v_R) / (1.0 + ratio);
    value_type ww = (w_L + ratio * w_R) / (1.0 + ratio);
    value_type H = (H_L + ratio * H_R) / (1.0 + ratio);

    value_type q2 = uu * uu + vv * vv + ww * ww;

    value_type c = std::sqrt(gam1 * (H - 0.5 * q2));

    value_type qn = uu * n_x + vv * n_y + ww * n_z;
    value_type ql = uu * l_x + vv * l_y + ww * l_z;
    value_type qm = uu * m_x + vv * m_y + ww * m_z;

    R[0][0] = 1.0;
    R[1][0] = uu - c * n_x;
    R[2][0] = vv - c * n_y;
    R[3][0] = ww - c * n_z;
    R[4][0] = H - qn * c;
  
    R[0][1] = 1.0;
    R[1][1] = uu;
    R[2][1] = vv;
    R[3][1] = ww;
    R[4][1] = 0.5 * q2;

    R[0][2] = 1.0;
    R[1][2] = uu + c * n_x;
    R[2][2] = vv + c * n_y;
    R[3][2] = ww + c * n_z;
    R[4][2] = H + qn * c;

    R[0][3] = 0.0;
    R[1][3] = l_x;
    R[2][3] = l_y;
    R[3][3] = l_z;
    R[4][3] = ql;

    R[0][4] = 0.0;
    R[1][4] = m_x;
    R[2][4] = m_y;
    R[3][4] = m_z;
    R[4][4] = qm;

    value_type K_2c2 = gam1 / (2.0 * c * c);
    value_type one_2c = 0.5 / c;
  
    L[0][0] = 0.5 * K_2c2 * q2 + one_2c * qn;
    L[1][0] = 1.0 - K_2c2 * q2;
    L[2][0] = 0.5 * K_2c2 * q2 - one_2c * qn;
    L[3][0] = -ql;
    L[4][0] = -qm;

    L[0][1] = -(K_2c2 * uu + one_2c * n_x);
    L[1][1] = 2.0 * K_2c2 * uu;
    L[2][1] = -(K_2c2 * uu - one_2c * n_x);
    L[3][1] = l_x;
    L[4][1] = m_x;

    L[0][2] = -(K_2c2 * vv + one_2c * n_y);
    L[1][2] = 2.0 * K_2c2 * vv;
    L[2][2] = -(K_2c2 * vv - one_2c * n_y);
    L[3][2] = l_y;
    L[4][2] = m_y;

    L[0][3] = -(K_2c2 * ww + one_2c * n_z);
    L[1][3] = 2.0 * K_2c2 * ww;
    L[2][3] = -(K_2c2 * ww - one_2c * n_z);
    L[3][3] = l_z;
    L[4][3] = m_z;

    L[0][4] = K_2c2;
    L[1][4] = -2.0 * K_2c2;
    L[2][4] = K_2c2;
    L[3][4] = 0.0;
    L[4][4] = 0.0;

  }

}
