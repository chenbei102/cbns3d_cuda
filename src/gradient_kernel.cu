#include "Block3d_cuda.h"
#include "gradient_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void gradient_kernel(const value_type* f,
				  value_type* f_xi, value_type* f_eta, value_type* f_zeta) {
    
    const index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const index_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const index_type k = blockIdx.z * blockDim.z + threadIdx.z;
    
    const size_type IM = blk_info.IM;
    const size_type JM = blk_info.JM;
    const size_type KM = blk_info.KM;
    

    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }
    
    size_type idx0 = blk_info.get_idx_x(i, j, k);
 
    size_type idx_1 = blk_info.get_idx_x(i - 1, j, k);
    size_type idx_2 = blk_info.get_idx_x(i - 2, j, k);
    size_type idx_3 = blk_info.get_idx_x(i - 3, j, k);
    size_type idx_4 = blk_info.get_idx_x(i - 4, j, k);
    size_type idx1  = blk_info.get_idx_x(i + 1, j, k);
    size_type idx2  = blk_info.get_idx_x(i + 2, j, k);
    size_type idx3  = blk_info.get_idx_x(i + 3, j, k);
    size_type idx4  = blk_info.get_idx_x(i + 4, j, k);

    // xi=0 
    if ( 0 == i ) {
      f_xi[idx0] = -25.0/12.0*f[idx0] + 4.0*f[idx1] - 3.0*f[idx2] + (4.0/3.0)*f[idx3] - 0.25*f[idx4];
    }

    // xi=1 
    if ( 1 == i ) {
      f_xi[idx0] = -5.0/6.0*f[idx0] - 0.25*f[idx_1] + 1.5*f[idx1] - 0.5*f[idx2] + (1.0/12.0)*f[idx3];
    }

    // xi=i 
    if ( (1 < i) && (IM-2 > i) ) {
      f_xi[idx0] = (1.0/12.0)*f[idx_2] - 2.0/3.0*f[idx_1] + (2.0/3.0)*f[idx1] - 1.0/12.0*f[idx2];
    }

    // xi=IM - 2 
    if ( IM-2 == i ) {
      f_xi[idx0] = (5.0/6.0)*f[idx0] - 1.0/12.0*f[idx_3] + 0.5*f[idx_2] - 1.5*f[idx_1] + 0.25*f[idx1];
    }

    // xi=IM - 1 
    if ( IM-1 == i ) {
      f_xi[idx0] = (25.0/12.0)*f[idx0] + 0.25*f[idx_4] - 4.0/3.0*f[idx_3] + 3.0*f[idx_2] - 4.0*f[idx_1];
    }

    idx_1 = blk_info.get_idx_x(i, j - 1, k);
    idx_2 = blk_info.get_idx_x(i, j - 2, k);
    idx_3 = blk_info.get_idx_x(i, j - 3, k);
    idx_4 = blk_info.get_idx_x(i, j - 4, k);
    idx1  = blk_info.get_idx_x(i, j + 1, k);
    idx2  = blk_info.get_idx_x(i, j + 2, k);
    idx3  = blk_info.get_idx_x(i, j + 3, k);
    idx4  = blk_info.get_idx_x(i, j + 4, k);

    // eta=0 
    if ( 0 == j ) {
      f_eta[idx0] = -25.0/12.0*f[idx0] + 4.0*f[idx1] - 3.0*f[idx2] + (4.0/3.0)*f[idx3] - 0.25*f[idx4];
    }

    // eta=1 
    if ( 1 == j ) {
      f_eta[idx0] = -5.0/6.0*f[idx0] - 0.25*f[idx_1] + 1.5*f[idx1] - 0.5*f[idx2] + (1.0/12.0)*f[idx3];
    }

    // eta=j 
    if ( (1 < j) && (JM-2 > j) ) {
      f_eta[idx0] = (1.0/12.0)*f[idx_2] - 2.0/3.0*f[idx_1] + (2.0/3.0)*f[idx1] - 1.0/12.0*f[idx2];
    }

    // eta=JM - 2 
    if ( JM-2 == j ) {
      f_eta[idx0] = (5.0/6.0)*f[idx0] - 1.0/12.0*f[idx_3] + 0.5*f[idx_2] - 1.5*f[idx_1] + 0.25*f[idx1];
    }

    // eta=JM - 1 
    if ( JM-1 == j ) {
      f_eta[idx0] = (25.0/12.0)*f[idx0] + 0.25*f[idx_4] - 4.0/3.0*f[idx_3] + 3.0*f[idx_2] - 4.0*f[idx_1];
    }

    idx_1 = blk_info.get_idx_x(i, j, k - 1);
    idx_2 = blk_info.get_idx_x(i, j, k - 2);
    idx_3 = blk_info.get_idx_x(i, j, k - 3);
    idx_4 = blk_info.get_idx_x(i, j, k - 4);
    idx1  = blk_info.get_idx_x(i, j, k + 1);
    idx2  = blk_info.get_idx_x(i, j, k + 2);
    idx3  = blk_info.get_idx_x(i, j, k + 3);
    idx4  = blk_info.get_idx_x(i, j, k + 4);

    // zeta=0 
    if ( 0 == k ) {
      f_zeta[idx0] = -25.0/12.0*f[idx0] + 4.0*f[idx1] - 3.0*f[idx2] + (4.0/3.0)*f[idx3] - 0.25*f[idx4];
    }

    // zeta=1 
    if ( 1 == k ) {
      f_zeta[idx0] = -5.0/6.0*f[idx0] - 0.25*f[idx_1] + 1.5*f[idx1] - 0.5*f[idx2] + (1.0/12.0)*f[idx3];
    }

    // zeta=k 
    if ( (1 < k) && (KM-2 > k) ) {
      f_zeta[idx0] = (1.0/12.0)*f[idx_2] - 2.0/3.0*f[idx_1] + (2.0/3.0)*f[idx1] - 1.0/12.0*f[idx2];
    }

    // zeta=KM - 2 
    if ( KM-2 == k ) {
      f_zeta[idx0] = (5.0/6.0)*f[idx0] - 1.0/12.0*f[idx_3] + 0.5*f[idx_2] - 1.5*f[idx_1] + 0.25*f[idx1];
    }

    // zeta=KM - 1 
    if ( KM-1 == k ) {
      f_zeta[idx0] = (25.0/12.0)*f[idx0] + 0.25*f[idx_4] - 4.0/3.0*f[idx_3] + 3.0*f[idx_2] - 4.0*f[idx_1];
    }

  }

  __global__ void gradient_wg_kernel(const value_type* f,
				     value_type* f_xi, value_type* f_eta, value_type* f_zeta) {
    
    const index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const index_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const index_type k = blockIdx.z * blockDim.z + threadIdx.z;
    
    const size_type IM = blk_info.IM_G;
    const size_type JM = blk_info.JM_G;
    const size_type KM = blk_info.KM_G;
    

    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }
    
    size_type idx0 = blk_info.get_idx_u(i, j, k);
 
    size_type idx_1 = blk_info.get_idx_u(i - 1, j, k);
    size_type idx_2 = blk_info.get_idx_u(i - 2, j, k);
    size_type idx_3 = blk_info.get_idx_u(i - 3, j, k);
    size_type idx_4 = blk_info.get_idx_u(i - 4, j, k);
    size_type idx1  = blk_info.get_idx_u(i + 1, j, k);
    size_type idx2  = blk_info.get_idx_u(i + 2, j, k);
    size_type idx3  = blk_info.get_idx_u(i + 3, j, k);
    size_type idx4  = blk_info.get_idx_u(i + 4, j, k);

    // xi=0 
    if ( 0 == i ) {
      f_xi[idx0] = -25.0/12.0*f[idx0] + 4.0*f[idx1] - 3.0*f[idx2] + (4.0/3.0)*f[idx3] - 0.25*f[idx4];
    }

    // xi=1 
    if ( 1 == i ) {
      f_xi[idx0] = -5.0/6.0*f[idx0] - 0.25*f[idx_1] + 1.5*f[idx1] - 0.5*f[idx2] + (1.0/12.0)*f[idx3];
    }

    // xi=i 
    if ( (1 < i) && (IM-2 > i) ) {
      f_xi[idx0] = (1.0/12.0)*f[idx_2] - 2.0/3.0*f[idx_1] + (2.0/3.0)*f[idx1] - 1.0/12.0*f[idx2];
    }

    // xi=IM - 2 
    if ( IM-2 == i ) {
      f_xi[idx0] = (5.0/6.0)*f[idx0] - 1.0/12.0*f[idx_3] + 0.5*f[idx_2] - 1.5*f[idx_1] + 0.25*f[idx1];
    }

    // xi=IM - 1 
    if ( IM-1 == i ) {
      f_xi[idx0] = (25.0/12.0)*f[idx0] + 0.25*f[idx_4] - 4.0/3.0*f[idx_3] + 3.0*f[idx_2] - 4.0*f[idx_1];
    }

    idx_1 = blk_info.get_idx_u(i, j - 1, k);
    idx_2 = blk_info.get_idx_u(i, j - 2, k);
    idx_3 = blk_info.get_idx_u(i, j - 3, k);
    idx_4 = blk_info.get_idx_u(i, j - 4, k);
    idx1  = blk_info.get_idx_u(i, j + 1, k);
    idx2  = blk_info.get_idx_u(i, j + 2, k);
    idx3  = blk_info.get_idx_u(i, j + 3, k);
    idx4  = blk_info.get_idx_u(i, j + 4, k);

    // eta=0 
    if ( 0 == j ) {
      f_eta[idx0] = -25.0/12.0*f[idx0] + 4.0*f[idx1] - 3.0*f[idx2] + (4.0/3.0)*f[idx3] - 0.25*f[idx4];
    }

    // eta=1 
    if ( 1 == j ) {
      f_eta[idx0] = -5.0/6.0*f[idx0] - 0.25*f[idx_1] + 1.5*f[idx1] - 0.5*f[idx2] + (1.0/12.0)*f[idx3];
    }

    // eta=j 
    if ( (1 < j) && (JM-2 > j) ) {
      f_eta[idx0] = (1.0/12.0)*f[idx_2] - 2.0/3.0*f[idx_1] + (2.0/3.0)*f[idx1] - 1.0/12.0*f[idx2];
    }

    // eta=JM - 2 
    if ( JM-2 == j ) {
      f_eta[idx0] = (5.0/6.0)*f[idx0] - 1.0/12.0*f[idx_3] + 0.5*f[idx_2] - 1.5*f[idx_1] + 0.25*f[idx1];
    }

    // eta=JM - 1 
    if ( JM-1 == j ) {
      f_eta[idx0] = (25.0/12.0)*f[idx0] + 0.25*f[idx_4] - 4.0/3.0*f[idx_3] + 3.0*f[idx_2] - 4.0*f[idx_1];
    }

    idx_1 = blk_info.get_idx_u(i, j, k - 1);
    idx_2 = blk_info.get_idx_u(i, j, k - 2);
    idx_3 = blk_info.get_idx_u(i, j, k - 3);
    idx_4 = blk_info.get_idx_u(i, j, k - 4);
    idx1  = blk_info.get_idx_u(i, j, k + 1);
    idx2  = blk_info.get_idx_u(i, j, k + 2);
    idx3  = blk_info.get_idx_u(i, j, k + 3);
    idx4  = blk_info.get_idx_u(i, j, k + 4);

    // zeta=0 
    if ( 0 == k ) {
      f_zeta[idx0] = -25.0/12.0*f[idx0] + 4.0*f[idx1] - 3.0*f[idx2] + (4.0/3.0)*f[idx3] - 0.25*f[idx4];
    }

    // zeta=1 
    if ( 1 == k ) {
      f_zeta[idx0] = -5.0/6.0*f[idx0] - 0.25*f[idx_1] + 1.5*f[idx1] - 0.5*f[idx2] + (1.0/12.0)*f[idx3];
    }

    // zeta=k 
    if ( (1 < k) && (KM-2 > k) ) {
      f_zeta[idx0] = (1.0/12.0)*f[idx_2] - 2.0/3.0*f[idx_1] + (2.0/3.0)*f[idx1] - 1.0/12.0*f[idx2];
    }

    // zeta=KM - 2 
    if ( KM-2 == k ) {
      f_zeta[idx0] = (5.0/6.0)*f[idx0] - 1.0/12.0*f[idx_3] + 0.5*f[idx_2] - 1.5*f[idx_1] + 0.25*f[idx1];
    }

    // zeta=KM - 1 
    if ( KM-1 == k ) {
      f_zeta[idx0] = (25.0/12.0)*f[idx0] + 0.25*f[idx_4] - 4.0/3.0*f[idx_3] + 3.0*f[idx_2] - 4.0*f[idx_1];
    }

  }

}
