#include "allocate_mem.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------


  void allocate_mem(const Block3dInfo *block_info, Block3dData *block_data) {

    // Allocates device memory 

    const size_type NEQ = block_info->NEQ;

    const size_type IM = block_info->IM;
    const size_type JM = block_info->JM;
    const size_type KM = block_info->KM;

    const size_type IM_G = block_info->IM_G;
    const size_type JM_G = block_info->JM_G;
    const size_type KM_G = block_info->KM_G;
  
    size_type array_size = IM * JM * KM;
    size_t d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(block_data->dt), d_size) );

    array_size = IM_G * JM_G * KM_G;
    d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(block_data->xi_x), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->xi_y), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->xi_z), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->eta_x), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->eta_y), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->eta_z), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->zeta_x), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->zeta_y), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->zeta_z), d_size) );

    ERROR_CHECK( cudaMalloc((void **)&(block_data->Jac), d_size) );
  
    ERROR_CHECK( cudaMalloc((void **)&(block_data->rho), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->u), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->v), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->w), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->p), d_size) );

#ifndef IS_INVISCID
    ERROR_CHECK( cudaMalloc((void **)&(block_data->T), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->mu), d_size) );

    ERROR_CHECK( cudaMalloc((void **)&(block_data->u_xi), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->v_xi), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->w_xi), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->u_eta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->v_eta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->w_eta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->u_zeta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->v_zeta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->w_zeta), d_size) );

    ERROR_CHECK( cudaMalloc((void **)&(block_data->T_xi), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->T_eta), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->T_zeta), d_size) );

    ERROR_CHECK( cudaMalloc((void **)&(block_data->tau_xx), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->tau_yy), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->tau_zz), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->tau_xy), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->tau_xz), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->tau_yz), d_size) );

    ERROR_CHECK( cudaMalloc((void **)&(block_data->q_x), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->q_y), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->q_z), d_size) );

    array_size = (NEQ - 1) * (IM_G - 2) * (JM_G - 2) * (KM_G - 2);
    d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(block_data->Ev), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->Fv), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->Gv), d_size) );
  
    array_size = (NEQ - 1) * (IM - 2) * (JM - 2) * (KM - 2);
    d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(block_data->diff_flux_vis), d_size) );
#endif
    
    array_size = NEQ * IM_G * JM_G * KM_G;
    d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(block_data->Q), d_size) );
    ERROR_CHECK( cudaMalloc((void **)&(block_data->Q_p), d_size) );
  
    array_size = NEQ * (IM - 1) * (JM - 2) * (KM - 2);
    d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(block_data->Ep), d_size) );

    array_size = NEQ * (IM - 2) * (JM - 1) * (KM - 2);
    d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(block_data->Fp), d_size) );

    array_size = NEQ * (IM - 2) * (JM - 2) * (KM - 1);
    d_size = array_size * sizeof(value_type);

    ERROR_CHECK( cudaMalloc((void **)&(block_data->Gp), d_size) );
  
  }

}
