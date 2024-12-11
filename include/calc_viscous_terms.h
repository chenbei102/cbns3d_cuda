#ifndef _CALC_VISCOUS_TERMS_H_
#define _CALC_VISCOUS_TERMS_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void calc_viscous_terms(const Block3dInfo *block_info, Block3dData *block_data);
  
}

#endif /* _CALC_VISCOUS_TERMS_H_ */
