#ifndef _CALC_DT_H_
#define _CALC_DT_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  value_type calc_dt(const Block3dInfo *block_info, Block3dData *block_data);
  
}

#endif /* _CALC_DT_H_ */
