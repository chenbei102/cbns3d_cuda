/**
 * @brief Header file defining utilities and structures for CUDA-based simulations.
 *
 * This file contains:
 * - Error checking utilities for CUDA API calls.
 * - A structure to encapsulate computational block information for simulations.
 * - A structure to manage device memory pointers for flow field data.
 */

#ifndef _BLOCK3D_CUDA_H_
#define _BLOCK3D_CUDA_H_

#include <cstdio>
#include <cuda_runtime.h>

#include "constants.h"

// Macro to check the return status of CUDA API calls and handle errors.
#define ERROR_CHECK(call)                                                   \
{                                                                           \
  const cudaError_t error = call;                                           \
  if (error != cudaSuccess) {                                               \
    std::printf("Error: %s:%d, ", __FILE__, __LINE__);			    \
    std::printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE);                                                \
  }                                                                         \
}


namespace block3d_cuda {

  struct Block3dInfo {

    // Contains essential data for computational blocks in a simulation.

    // Constant to define the number of equations to solve
    static const size_type NEQ {constant::NEQ};

    // Number of ghost points (used in boundary conditions)
    static const size_type NG {constant::NG};

    // Number of points in the computational mesh 
    size_type IM; // Number of points in the i-direction
    size_type JM; // Number of points in the j-direction
    size_type KM; // Number of points in the k-direction

    size_type IM_G; 
    size_type JM_G; 
    size_type KM_G; 

    value_type CFL;

    // Test case parameters
    value_type angle_attack;
    
    size_type i_begin;
    size_type i_end;

    // Physical parameters relevant to the simulation
    value_type Mach;
    value_type Re;
    value_type Mach2;		
    value_type gM2;		
    value_type Re_inv;		

    value_type p_inf;		

    value_type gamma;		
    value_type Pr;
    value_type Pr_t;
    value_type gam1;		
    value_type gam1_inv;	
    value_type Pr_inv;		
    value_type Pr_t_inv;
    value_type gPr;

    value_type C_T_inf;		
    value_type C_dt_v;		

    /* ---------------------------------------------------------------------- */

    // Utility functions for converting multi-dimensional indices to one-dimensional indices.
    
    __forceinline__ __device__ size_type get_node_num() {
      return IM * JM * KM;
    }

    __forceinline__ __device__ size_type get_idx_x(size_type i, size_type j, size_type k) {
      return i + IM * (j + JM * k);
    }

    __forceinline__ __device__ size_type get_idx_u(size_type i, size_type j, size_type k) {
      return i + IM_G * (j + JM_G * k);
    }

    __forceinline__ __device__ size_type get_idx(index_type i, index_type j, index_type k) {
      return (NG+i) + IM_G * ((NG+j) + JM_G * (NG+k));
    }

    __forceinline__ __device__ size_type get_idx_Qa(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + NEQ * (i + IM_G * (j + JM_G * k));
    }

    __forceinline__ __device__ size_type get_idx_Q(size_type n_eq,
						   index_type i, index_type j, index_type k) {
      return n_eq + NEQ * ((NG+i) + IM_G * ((NG+j) + JM_G * (NG+k)));
    }

    __forceinline__ __device__ size_type get_idx_Ep(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + NEQ * (i + (IM - 1) * (j + (JM - 2) * k));
    }

    __forceinline__ __device__ size_type get_idx_Fp(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + NEQ * (i + (IM - 2) * (j + (JM - 1) * k));
    }

    __forceinline__ __device__ size_type get_idx_Gp(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + NEQ * (i + (IM - 2) * (j + (JM - 2) * k));
    }

    __forceinline__ __device__ size_type get_idx_Ev(size_type n_eq,
						    size_type i, size_type j, size_type k) {
      return n_eq + (NEQ - 1) * (i + (IM_G - 2) * (j + (JM_G - 2) * k));
    }

    __forceinline__ __device__ size_type get_idx_dfv(size_type n_eq,
						     size_type i, size_type j, size_type k) {
      return n_eq + (NEQ - 1) * (i + (IM - 2) * (j + (JM - 2) * k));
    }

    /* ---------------------------------------------------------------------- */
    
  };


  struct Block3dData {

    // Manages pointers to flow field data in device memory.

    // Mesh coordinate data
    value_type *x;
    value_type *y;
    value_type *z;

    // Geometric metrics for coordinate transformations
    value_type *x_xi;
    value_type *y_xi;
    value_type *z_xi;
    value_type *x_eta;
    value_type *y_eta;
    value_type *z_eta;
    value_type *x_zeta;
    value_type *y_zeta;
    value_type *z_zeta;

    value_type *xi_x;
    value_type *xi_y;
    value_type *xi_z;
    value_type *eta_x;
    value_type *eta_y;
    value_type *eta_z;
    value_type *zeta_x;
    value_type *zeta_y;
    value_type *zeta_z;

    value_type *Jac;

    // Flow field data
    value_type *rho;
    value_type *u;
    value_type *v;
    value_type *w;
    value_type *p;
    
    value_type *Q;
    value_type *Q_p;

    value_type *T;
    value_type *mu;

#ifndef IS_INVISCID
    // Spatial derivatives of flow field variables
    value_type *u_xi;
    value_type *v_xi;
    value_type *w_xi;
    value_type *u_eta;
    value_type *v_eta;
    value_type *w_eta;
    value_type *u_zeta;
    value_type *v_zeta;
    value_type *w_zeta;

    value_type *T_xi;
    value_type *T_eta;
    value_type *T_zeta;

    // Stress tensor
    value_type *tau_xx;
    value_type *tau_yy;
    value_type *tau_zz;
    value_type *tau_xy;
    value_type *tau_xz;
    value_type *tau_yz;

    // Heat flux
    value_type *q_x;
    value_type *q_y;
    value_type *q_z;

    // Viscous flux 
    value_type *Ev;
    value_type *Fv;
    value_type *Gv;
#endif
    
    // Derivative of viscous fluxes
    value_type *diff_flux_vis;

    // Reconstructed numerical flux for inviscid term
    value_type *Ep;
    value_type *Fp;
    value_type *Gp;

    // Local time step
    value_type *dt;
    
  };
    
}

#endif /* _BLOCK3D_CUDA_H_ */
