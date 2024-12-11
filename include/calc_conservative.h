#ifndef _CALC_CONSERVATIVE_H_
#define _CALC_CONSERVATIVE_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void calc_conservative(const Block3dInfo *block_info, Block3dData *block_data,
			 value_type *Q);
  
}

#endif /* _CALC_CONSERVATIVE_H_ */
