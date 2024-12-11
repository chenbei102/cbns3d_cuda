#ifndef _CALC_VISCOUS_FLUX_CONTRIBUTION_H_
#define _CALC_VISCOUS_FLUX_CONTRIBUTION_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void calc_viscous_flux_contribution(const Block3dInfo *block_info, Block3dData *block_data);
  
}

#endif /* _CALC_VISCOUS_FLUX_CONTRIBUTION_H_ */
