#ifndef _SET_CONSERVATIVE_H_
#define _SET_CONSERVATIVE_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void set_conservative(const Block3dInfo *block_info, Block3dData *block_data,
			const value_type *Q);
  
}

#endif /* _SET_CONSERVATIVE_H_ */
