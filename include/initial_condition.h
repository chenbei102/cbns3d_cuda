#ifndef _INITIAL_CONDITION_H_
#define _INITIAL_CONDITION_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void initial_condition(const Block3dInfo *block_info, Block3dData *block_data);

}

#endif /* _INITIAL_CONDITION_H_ */
