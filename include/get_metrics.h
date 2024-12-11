#ifndef _GET_METRICS_H_
#define _GET_METRICS_H_

#include "Block3d_cuda.h"


namespace block3d_cuda {

  void get_metrics(const Block3dInfo *block_info, const Block3dData *block_data,
		   value_type* xi_x, value_type* xi_y, value_type* xi_z,
		   value_type* eta_x, value_type* eta_y, value_type* eta_z,
		   value_type* zeta_x, value_type* zeta_y, value_type* zeta_z,
		   value_type* Jac);

}

#endif /* _GET_METRICS_H_ */
