#include "Block3d_cuda.h"
#include "metrics_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void metrics_kernel(const value_type* x_xi,
				 const value_type* x_eta,
				 const value_type* x_zeta,
				 const value_type* y_xi,
				 const value_type* y_eta,
				 const value_type* y_zeta,
				 const value_type* z_xi,
				 const value_type* z_eta,
				 const value_type* z_zeta,
				 value_type* xi_x,
				 value_type* xi_y,
				 value_type* xi_z,
				 value_type* eta_x,
				 value_type* eta_y,
				 value_type* eta_z,
				 value_type* zeta_x,
				 value_type* zeta_y,
				 value_type* zeta_z,
				 value_type* Jac
				 ) {
    
    // Compute the elements of the Jacobian matrix for the coordinate
    // transformation from physical space to computational space, along with the
    // determinant of the Jacobian.
    
    const index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const index_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const index_type k = blockIdx.z * blockDim.z + threadIdx.z;
    
    const size_type IM = blk_info.IM;
    const size_type JM = blk_info.JM;
    const size_type KM = blk_info.KM;

    const size_type NG = blk_info.NG;

    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }
    
    size_type idx1 = blk_info.get_idx_x(i, j, k);
    size_type idx2 = blk_info.get_idx(i, j, k);

    value_type Jac_a =
      -x_eta[idx1]*y_xi[idx1]*z_zeta[idx1] + x_eta[idx1]*y_zeta[idx1]*z_xi[idx1]
      + x_xi[idx1]*y_eta[idx1]*z_zeta[idx1] - x_xi[idx1]*y_zeta[idx1]*z_eta[idx1]
      - x_zeta[idx1]*y_eta[idx1]*z_xi[idx1] + x_zeta[idx1]*y_xi[idx1]*z_eta[idx1];

    Jac[idx2] = Jac_a;

    xi_x[idx2] = (y_eta[idx1]*z_zeta[idx1] - y_zeta[idx1]*z_eta[idx1])/Jac_a;
    xi_y[idx2] = (-x_eta[idx1]*z_zeta[idx1] + x_zeta[idx1]*z_eta[idx1])/Jac_a;
    xi_z[idx2] = (x_eta[idx1]*y_zeta[idx1] - x_zeta[idx1]*y_eta[idx1])/Jac_a;
    eta_x[idx2] = (-y_xi[idx1]*z_zeta[idx1] + y_zeta[idx1]*z_xi[idx1])/Jac_a;
    eta_y[idx2] = (x_xi[idx1]*z_zeta[idx1] - x_zeta[idx1]*z_xi[idx1])/Jac_a;
    eta_z[idx2] = (-x_xi[idx1]*y_zeta[idx1] + x_zeta[idx1]*y_xi[idx1])/Jac_a;
    zeta_x[idx2] = (-y_eta[idx1]*z_xi[idx1] + y_xi[idx1]*z_eta[idx1])/Jac_a;
    zeta_y[idx2] = (x_eta[idx1]*z_xi[idx1] - x_xi[idx1]*z_eta[idx1])/Jac_a;
    zeta_z[idx2] = (-x_eta[idx1]*y_xi[idx1] + x_xi[idx1]*y_eta[idx1])/Jac_a;

    // Extend geometric metrics to include ghost points along the xi-direction
    if (0 == i) {
      for(index_type li = 0; li < NG; li++) {
	idx1 = blk_info.get_idx(-li-1, j, k);

	Jac[idx1] = Jac_a;

	xi_x[idx1] = xi_x[idx2];
	xi_y[idx1] = xi_y[idx2];
	xi_z[idx1] = xi_z[idx2];
	eta_x[idx1] = eta_x[idx2];
	eta_y[idx1] = eta_y[idx2];
	eta_z[idx1] = eta_z[idx2];
	zeta_x[idx1] = zeta_x[idx2];
	zeta_y[idx1] = zeta_y[idx2];
	zeta_z[idx1] = zeta_z[idx2];
      }
    }
    if (IM-1 == i) {
      for(index_type li = 0; li < NG; li++) {
	idx1 = blk_info.get_idx(IM+li, j, k);

	Jac[idx1] = Jac_a;

	xi_x[idx1] = xi_x[idx2];
	xi_y[idx1] = xi_y[idx2];
	xi_z[idx1] = xi_z[idx2];
	eta_x[idx1] = eta_x[idx2];
	eta_y[idx1] = eta_y[idx2];
	eta_z[idx1] = eta_z[idx2];
	zeta_x[idx1] = zeta_x[idx2];
	zeta_y[idx1] = zeta_y[idx2];
	zeta_z[idx1] = zeta_z[idx2];
      }
    }
      
    // Extend geometric metrics to include ghost points along the eta-direction
    if (0 == j) {
      for(index_type li = 0; li < NG; li++) {
	idx1 = blk_info.get_idx(i, -li-1, k);

	Jac[idx1] = Jac_a;

	xi_x[idx1] = xi_x[idx2];
	xi_y[idx1] = xi_y[idx2];
	xi_z[idx1] = xi_z[idx2];
	eta_x[idx1] = eta_x[idx2];
	eta_y[idx1] = eta_y[idx2];
	eta_z[idx1] = eta_z[idx2];
	zeta_x[idx1] = zeta_x[idx2];
	zeta_y[idx1] = zeta_y[idx2];
	zeta_z[idx1] = zeta_z[idx2];
      }
    }
    if (JM-1 == j) {
      for(index_type li = 0; li < NG; li++) {
	idx1 = blk_info.get_idx(i, JM+li, k);

	Jac[idx1] = Jac_a;

	xi_x[idx1] = xi_x[idx2];
	xi_y[idx1] = xi_y[idx2];
	xi_z[idx1] = xi_z[idx2];
	eta_x[idx1] = eta_x[idx2];
	eta_y[idx1] = eta_y[idx2];
	eta_z[idx1] = eta_z[idx2];
	zeta_x[idx1] = zeta_x[idx2];
	zeta_y[idx1] = zeta_y[idx2];
	zeta_z[idx1] = zeta_z[idx2];
      }
    }
      
    // Extend geometric metrics to include ghost points along the zeta-direction
    if (0 == k) {
      for(index_type li = 0; li < NG; li++) {
	idx1 = blk_info.get_idx(i, j, -li-1);

	Jac[idx1] = Jac_a;

	xi_x[idx1] = xi_x[idx2];
	xi_y[idx1] = xi_y[idx2];
	xi_z[idx1] = xi_z[idx2];
	eta_x[idx1] = eta_x[idx2];
	eta_y[idx1] = eta_y[idx2];
	eta_z[idx1] = eta_z[idx2];
	zeta_x[idx1] = zeta_x[idx2];
	zeta_y[idx1] = zeta_y[idx2];
	zeta_z[idx1] = zeta_z[idx2];
      }
    }
    if (KM-1 == k) {
      for(index_type li = 0; li < NG; li++) {
	idx1 = blk_info.get_idx(i, j, KM+li);

	Jac[idx1] = Jac_a;

	xi_x[idx1] = xi_x[idx2];
	xi_y[idx1] = xi_y[idx2];
	xi_z[idx1] = xi_z[idx2];
	eta_x[idx1] = eta_x[idx2];
	eta_y[idx1] = eta_y[idx2];
	eta_z[idx1] = eta_z[idx2];
	zeta_x[idx1] = zeta_x[idx2];
	zeta_y[idx1] = zeta_y[idx2];
	zeta_z[idx1] = zeta_z[idx2];
      }
    }

    // -------------------------------------------------------------------------
    if ((NG > i) && (NG > j)) {

      idx1 = blk_info.get_idx(-1-i, -1-j, k);
      idx2 = blk_info.get_idx(0, 0, k);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((IM-NG <= i) && (NG > j)) {

      idx1 = blk_info.get_idx(i+NG, -1-j, k);
      idx2 = blk_info.get_idx(IM-1, 0, k);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((IM-NG <= i) && (JM-NG <= j)) {

      idx1 = blk_info.get_idx(i+NG, j+NG, k);
      idx2 = blk_info.get_idx(IM-1, JM-1, k);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((NG > i) && (JM-NG <= j)) {

      idx1 = blk_info.get_idx(-1-i, j+NG, k);
      idx2 = blk_info.get_idx(0, JM-1, k);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((NG > i) && (NG > k)) {

      idx1 = blk_info.get_idx(-1-i, j, -1-k);
      idx2 = blk_info.get_idx(0, j, 0);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((IM-NG <= i) && (NG > k)) {

      idx1 = blk_info.get_idx(i+NG, j, -1-k);
      idx2 = blk_info.get_idx(IM-1, j, 0);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((IM-NG <= i) && (KM-NG <= k)) {

      idx1 = blk_info.get_idx(i+NG, j, k+NG);
      idx2 = blk_info.get_idx(IM-1, j, KM-1);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((NG > i) && (KM-NG <= k)) {

      idx1 = blk_info.get_idx(-1-i, j, k+NG);
      idx2 = blk_info.get_idx(0, j, KM-1);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((NG > j) && (NG > k)) {

      idx1 = blk_info.get_idx(i, -1-j, -1-k);
      idx2 = blk_info.get_idx(i, 0, 0);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((JM-NG <= j) && (NG > k)) {

      idx1 = blk_info.get_idx(i, j+NG, -1-k);
      idx2 = blk_info.get_idx(i, JM-1, 0);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((JM-NG <= j) && (KM-NG <= k)) {

      idx1 = blk_info.get_idx(i, j+NG, k+NG);
      idx2 = blk_info.get_idx(i, JM-1, KM-1);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((NG > j) && (KM-NG <= k)) {

      idx1 = blk_info.get_idx(i, -1-j, k+NG);
      idx2 = blk_info.get_idx(i, 0, KM-1);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    // -------------------------------------------------------------------------
    if ((NG > i) && (NG > j) && (NG > k)) {

      idx1 = blk_info.get_idx(-1-i, -1-j, -1-k);
      idx2 = blk_info.get_idx(0, 0, 0);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }
    
    if ((IM-NG <= i) && (NG > j) && (NG > k)) {

      idx1 = blk_info.get_idx(i+NG, -1-j, -1-k);
      idx2 = blk_info.get_idx(IM-1, 0, 0);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }
    
    if ((IM-NG <= i) && (JM-NG <= j) && (NG > k)) {

      idx1 = blk_info.get_idx(i+NG, j+NG, -1-k);
      idx2 = blk_info.get_idx(IM-1, JM-1, 0);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((NG > i) && (JM-NG <= j) && (NG > k)) {

      idx1 = blk_info.get_idx(-1-i, j+NG, -1-k);
      idx2 = blk_info.get_idx(0, JM-1, 0);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }
      
    if ((NG > i) && (NG > j) && (KM-NG <= k)) {

      idx1 = blk_info.get_idx(-1-i, -1-j, k+NG);
      idx2 = blk_info.get_idx(0, 0, KM-1);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }
    
    if ((IM-NG <= i) && (NG > j) && (KM-NG <= k)) {

      idx1 = blk_info.get_idx(i+NG, -1-j, k+NG);
      idx2 = blk_info.get_idx(IM-1, 0, KM-1);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }
    
    if ((IM-NG <= i) && (JM-NG <= j) && (KM-NG <= k)) {

      idx1 = blk_info.get_idx(i+NG, j+NG, k+NG);
      idx2 = blk_info.get_idx(IM-1, JM-1, KM-1);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }

    if ((NG > i) && (JM-NG <= j) && (KM-NG <= k)) {

      idx1 = blk_info.get_idx(-1-i, j+NG, k+NG);
      idx2 = blk_info.get_idx(0, JM-1, KM-1);

      Jac[idx1] = Jac[idx2];

      xi_x[idx1] = xi_x[idx2];
      xi_y[idx1] = xi_y[idx2];
      xi_z[idx1] = xi_z[idx2];
      eta_x[idx1] = eta_x[idx2];
      eta_y[idx1] = eta_y[idx2];
      eta_z[idx1] = eta_z[idx2];
      zeta_x[idx1] = zeta_x[idx2];
      zeta_y[idx1] = zeta_y[idx2];
      zeta_z[idx1] = zeta_z[idx2];

    }
      
  }

}
