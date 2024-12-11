#ifndef _ALLOCATE_MEM_H_
#define _ALLOCATE_MEM_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void allocate_mem(const Block3dInfo *block_info, Block3dData *block_data);

}
  
#endif /* _ALLOCATE_MEM_H_ */
