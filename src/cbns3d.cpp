/**
 * @brief High-Order GPU-Accelerated 3D Compressible Navier-Stokes Solver
 *
 * This program solves the 3D compressible Navier-Stokes equations on curvilinear 
 * structured meshes using high-order finite difference methods. By leveraging 
 * CUDA for GPU acceleration, it achieves significantly enhanced computational 
 * efficiency.
 *
 * This file contains the implementation of the program's main function, which 
 * serves as the application entry point. 
 *
 * @author Bei Chen
 */

#include "Block3d.h"

int main() {

  Block3d block {};

  block.solve();

  return 0;

}
