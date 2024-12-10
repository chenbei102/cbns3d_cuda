# cbns3d_cuda
This repository provides a **High-Order GPU-Accelerated 3D Compressible Navier-Stokes Solver**. The solver employs high-order finite difference methods to solve the 3D compressible Navier-Stokes equations on curvilinear structured meshes. Developed in C++ with CUDA, it leverages GPU acceleration to significantly enhance computational efficiency. 

This code is derived from my research experience in numerical methods for CFD, and it emphasizes clarity and maintainability, prioritizing transparent algorithmic logic over micro-level code optimization.

<div style="display: flex; align-items: center; justify-content: space-between;">
    <img src="fig/rae2822.png" alt="Image 1" style="width: 49%; height: auto;">
    <img src="fig/cp_rae2822.png" alt="Image 2" style="width: 49%; height: auto;">
</div>

## Features:
- **High-Order Finite Difference Schemes**: Implements a 5th-order WENO scheme for inviscid fluxes and a 4th-order central finite difference scheme for viscous fluxes, ensuring high accuracy in simulations.
- **Curvilinear Structured Mesh Support**: Flexible support for non-Cartesian geometries.
- **CUDA Acceleration**: Leverages GPU computing to significantly enhance computational efficiency.
- **CMake Build System**: Simplified build and cross-platform compatibility.
- **Unit Testing**: Developed with **Google Test** to ensure code reliability.
