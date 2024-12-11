#ifndef _CALC_METRICS_H_
#define _CALC_METRICS_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void calc_metrics(const Block3dInfo *block_info, Block3dData *block_data,
		    const value_type *x, const value_type *y, const value_type *z);

}

#endif /* _CALC_METRICS_H_ */
