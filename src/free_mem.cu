#include "free_mem.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------


  void free_mem(Block3dData *block_data) {

    // Release device memory 
  
    cudaFree(block_data->dt);

    cudaFree(block_data->xi_x);
    cudaFree(block_data->xi_y);
    cudaFree(block_data->xi_z);
    cudaFree(block_data->eta_x);
    cudaFree(block_data->eta_y);
    cudaFree(block_data->eta_z);
    cudaFree(block_data->zeta_x);
    cudaFree(block_data->zeta_y);
    cudaFree(block_data->zeta_z);

    cudaFree(block_data->Jac);
  
    cudaFree(block_data->rho);
    cudaFree(block_data->u);
    cudaFree(block_data->v);
    cudaFree(block_data->w);
    cudaFree(block_data->p);

#ifndef IS_INVISCID
    cudaFree(block_data->T);
    cudaFree(block_data->mu);

    cudaFree(block_data->u_xi);
    cudaFree(block_data->v_xi);
    cudaFree(block_data->w_xi);
    cudaFree(block_data->u_eta);
    cudaFree(block_data->v_eta);
    cudaFree(block_data->w_eta);
    cudaFree(block_data->u_zeta);
    cudaFree(block_data->v_zeta);
    cudaFree(block_data->w_zeta);

    cudaFree(block_data->T_xi);
    cudaFree(block_data->T_eta);
    cudaFree(block_data->T_zeta);

    cudaFree(block_data->tau_xx);
    cudaFree(block_data->tau_yy);
    cudaFree(block_data->tau_zz);
    cudaFree(block_data->tau_xy);
    cudaFree(block_data->tau_xz);
    cudaFree(block_data->tau_yz);

    cudaFree(block_data->q_x);
    cudaFree(block_data->q_y);
    cudaFree(block_data->q_z);

    cudaFree(block_data->Ev);
    cudaFree(block_data->Fv);
    cudaFree(block_data->Gv);

    cudaFree(block_data->diff_flux_vis);
#endif

    cudaFree(block_data->Q);
    cudaFree(block_data->Q_p);
  
    cudaFree(block_data->Ep);
    cudaFree(block_data->Fp);
    cudaFree(block_data->Gp);
  
  }

}
