#include "Block3d_cuda.h"
#include "primitive_bc_kernel.h"


namespace block3d_cuda {

  // ---------------------------------------------------------------------------

  extern __constant__ Block3dInfo blk_info;
  
  // ---------------------------------------------------------------------------

  __global__ void primitive_bc_kernel(value_type* Q,
				      value_type* rho,
				      value_type* u,
				      value_type* v,
				      value_type* w,
				      value_type* p,
				      value_type* T,
				      const value_type* xi_x,
				      const value_type* xi_y,
				      const value_type* xi_z,
				      const value_type* eta_x,
				      const value_type* eta_y,
				      const value_type* eta_z,
				      const value_type* zeta_x,
				      const value_type* zeta_y,
				      const value_type* zeta_z,
				      const value_type* Jac
				      ) {
    
    const size_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type j = blockIdx.y * blockDim.y + threadIdx.y;
    const size_type k = blockIdx.z * blockDim.z + threadIdx.z;

    const size_type IM = blk_info.IM;
    const size_type JM = blk_info.JM;
    const size_type KM = blk_info.KM;
    
    if ((IM <= i) || (JM <= j) || (KM <= k)) {
      return;
    }

    const size_type NEQ = blk_info.NEQ;
    const size_type NG = blk_info.NG;

    const size_type i_begin = blk_info.i_begin;
    const size_type i_end = blk_info.i_end;

    const value_type gam1 = blk_info.gam1;
    const value_type p_inf = blk_info.p_inf;

    // -------------------------------------------------------------------------
    // Compute primitive variables
    
    size_type idx1 = blk_info.get_idx(i, j, k);
    size_type idx2 = blk_info.get_idx_Q(0, i, j, k);

    if ((0 < i) && (IM-1 > i) &&
	(0 < j) && (JM-1 > j) &&
	(0 < k) && (KM-1 > k) ) {

      const value_type rr = Q[idx2  ];
      const value_type uu = Q[idx2+1] / rr;
      const value_type vv = Q[idx2+2] / rr;
      const value_type ww = Q[idx2+3] / rr;
      const value_type pp = gam1 * (Q[idx2+4] - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

      rho[idx1] = rr;	
      u[idx1] = uu;
      v[idx1] = vv;
      w[idx1] = ww;
      p[idx1] = pp;

#ifndef IS_INVISCID
      T[idx1] = blk_info.gM2 * pp / rr;
#endif
      
    }
    
    // -------------------------------------------------------------------------
    // Apply boundary conditions

    // xi=1, eta=j, zeta=k
    if (0 == i) {

      if ((0 < j) && (JM-1 > j) &&
	  (0 < k) && (KM-1 > k) ) {

	idx1 = blk_info.get_idx_Q(0, 1, j, k);

	value_type rho_1 = Q[idx1  ];
	value_type rho_u = Q[idx1+1];
	value_type rho_v = Q[idx1+2];
	value_type rho_w = Q[idx1+3];
	value_type rho_et = p_inf / gam1 + 0.5 * (rho_u * rho_u + rho_v * rho_v + rho_w * rho_w) / rho_1;

	for (index_type li = 0; li <= NG; li++) {

	  idx1 = blk_info.get_idx_Q(0, -li, j, k);

	  Q[idx1  ] = rho_1;
	  Q[idx1+1] = rho_u;
	  Q[idx1+2] = rho_v;
	  Q[idx1+3] = rho_w;
	  Q[idx1+4] = rho_et;
	
	}

	const value_type rr = rho_1;
	const value_type uu = rho_u / rr;
	const value_type vv = rho_v / rr;
	const value_type ww = rho_w / rr;
	const value_type pp = gam1 * (rho_et - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

	for (index_type li = 0; li <= NG; li++) {

	  idx1 = blk_info.get_idx(-li, j, k);

	  rho[idx1] = rr;	
	  u[idx1] = uu;
	  v[idx1] = vv;
	  w[idx1] = ww;
	  p[idx1] = pp;

#ifndef IS_INVISCID
	  T[idx1] = blk_info.gM2 * pp / rr;
#endif
	  
	}
	
      }
      
    }
    
    // xi=IM, eta=j, zeta=k
    if (IM-1 == i) {

      if ((0 < j) && (JM-1 > j) &&
	  (0 < k) && (KM-1 > k) ) {

	idx1 = blk_info.get_idx_Q(0, IM-2, j, k);

	value_type rho_1 = Q[idx1  ];
	value_type rho_u = Q[idx1+1];
	value_type rho_v = Q[idx1+2];
	value_type rho_w = Q[idx1+3];
	value_type rho_et = p_inf / gam1 + 0.5 * (rho_u * rho_u + rho_v * rho_v + rho_w * rho_w) / rho_1;

	for (size_type li = 0; li <= NG; li++) {

	  idx1 = blk_info.get_idx_Q(0, IM-1+li, j, k);

	  Q[idx1  ] = rho_1;
	  Q[idx1+1] = rho_u;
	  Q[idx1+2] = rho_v;
	  Q[idx1+3] = rho_w;
	  Q[idx1+4] = rho_et;

	}
      
	const value_type rr = rho_1;
	const value_type uu = rho_u / rr;
	const value_type vv = rho_v / rr;
	const value_type ww = rho_w / rr;
	const value_type pp = gam1 * (rho_et - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

	for (size_type li = 0; li <= NG; li++) {

	  idx1 = blk_info.get_idx(IM-1+li, j, k);

	  rho[idx1] = rr;	
	  u[idx1] = uu;
	  v[idx1] = vv;
	  w[idx1] = ww;
	  p[idx1] = pp;

#ifndef IS_INVISCID
	  T[idx1] = blk_info.gM2 * pp / rr;
#endif

	}

      }

    }
    __syncthreads();
    
    // xi=i, eta=1, zeta=k
    if (0 == j) {

      if ((0 < k) && (KM-1 > k)) {

	if (i_begin-1 > i) {

	  idx1 = blk_info.get_idx_Q(0, i, 1, k);

	  value_type rr = Q[idx1  ];
	  value_type uu = Q[idx1+1] / rr;
	  value_type vv = Q[idx1+2] / rr;
	  value_type ww = Q[idx1+3] / rr;
	  value_type pp = gam1 * (Q[idx1+4] - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

	  idx1 = blk_info.get_idx_Q(0, IM-1-i, 1, k);

	  value_type rr1 = Q[idx1  ];
	  value_type uu1 = Q[idx1+1] / rr1;
	  value_type vv1 = Q[idx1+2] / rr1;
	  value_type ww1 = Q[idx1+3] / rr1;
	  value_type pp1 = gam1 * (Q[idx1+4] - 0.5 * rr1 * (uu1 * uu1 + vv1 * vv1 + ww1 * ww1));

	  rr += rr1;
	  uu += uu1;
	  vv += vv1;
	  ww += ww1;
	  pp += pp1;

	  rr *= 0.5;
	  uu *= 0.5;
	  vv *= 0.5;
	  ww *= 0.5;
	  pp *= 0.5;

	  idx1 = blk_info.get_idx_Q(0, i, 0, k);

	  Q[idx1  ] = rr;
	  Q[idx1+1] = rr * uu;
	  Q[idx1+2] = rr * vv;
	  Q[idx1+3] = rr * ww;
	  Q[idx1+4] = pp / gam1 + 0.5 * rr * (uu * uu + vv * vv + ww * ww);

	  idx2 = blk_info.get_idx_Q(0, IM-1-i, 0, k);

	  Q[idx2  ] = Q[idx1  ];
	  Q[idx2+1] = Q[idx1+1];
	  Q[idx2+2] = Q[idx1+2];
	  Q[idx2+3] = Q[idx1+3];      
	  Q[idx2+4] = Q[idx1+4];
      
	  for(index_type lj = 1; lj <= NG; lj++) {

	    idx1 = blk_info.get_idx_Q(0, IM-1-i, lj, k);
	    idx2 = blk_info.get_idx_Q(0, i, -lj, k);

	    Q[idx2  ] = Q[idx1  ];
	    Q[idx2+1] = Q[idx1+1];
	    Q[idx2+2] = Q[idx1+2];
	    Q[idx2+3] = Q[idx1+3];      
	    Q[idx2+4] = Q[idx1+4];

	    idx1 = blk_info.get_idx_Q(0, i, lj, k);
	    idx2 = blk_info.get_idx_Q(0, IM-1-i, -lj, k);

	    Q[idx2  ] = Q[idx1  ];
	    Q[idx2+1] = Q[idx1+1];
	    Q[idx2+2] = Q[idx1+2];
	    Q[idx2+3] = Q[idx1+3];      
	    Q[idx2+4] = Q[idx1+4];

	  }

	} else if (i_end > i) {
#ifndef IS_INVISCID
	  idx1 = blk_info.get_idx_Q(0, i, 1, k);

	  value_type rho_1 = Q[idx1  ];
	  value_type rho_u = Q[idx1+1];
	  value_type rho_v = Q[idx1+2];
	  value_type rho_w = Q[idx1+3];
	  value_type rho_et = Q[idx1+4];

	  value_type pp = gam1 * (rho_et - 0.5 * (rho_u * rho_u + rho_v * rho_v + rho_w * rho_w) / rho_1);

	  idx1 = blk_info.get_idx_Q(0, i, 0, k);

	  Q[idx1  ] = rho_1;
	  Q[idx1+1] = 0.0;
	  Q[idx1+2] = 0.0;
	  Q[idx1+3] = 0.0;
	  Q[idx1+4] = pp / gam1;

	  // -----------------------------------------------------------------------
	  for(index_type lj = 1; lj <= NG; lj++) {

	    idx1 = blk_info.get_idx_Q(0, i, lj, k);
	    idx2 = blk_info.get_idx_Q(0, i, -lj, k);

	    Q[idx2  ] =  Q[idx1  ];
	    Q[idx2+1] = -Q[idx1+1];
	    Q[idx2+2] = -Q[idx1+2];
	    Q[idx2+3] = -Q[idx1+3];      
	    Q[idx2+4] =  Q[idx1+4];

	  }
	  // -----------------------------------------------------------------------
#else
	  idx1 = blk_info.get_idx_Q(0, i, 1, k);

	  value_type rr = Q[idx1  ];
	  value_type uu = Q[idx1+1] / rr;
	  value_type vv = Q[idx1+2] / rr;
	  value_type ww = Q[idx1+3] / rr;
	  value_type pp = gam1 * (Q[idx1+4] - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

	  idx1 = blk_info.get_idx(i, 0, k);

	  value_type t_x = eta_y[idx1];
	  value_type t_y = -eta_x[idx1];
	  value_type t_abs = Jac[idx1];

	  t_x *= t_abs;
	  t_y *= t_abs;

	  t_abs = std::sqrt(t_x * t_x + t_y * t_y);

	  t_x /= t_abs;
	  t_y /= t_abs;

	  // -----------------------------------------------------------------------
	  for(index_type lj = 1; lj <= NG; lj++) {

	    idx1 = blk_info.get_idx_Q(0, i, lj, k);
	    idx2 = blk_info.get_idx_Q(0, i, -lj, k);

	    t_abs = Q[idx1+1] * t_x + Q[idx1+2] * t_y;
	
	    Q[idx2  ] = Q[idx1  ];
	    Q[idx2+1] = 2 * t_abs * t_x - Q[idx1+1];
	    Q[idx2+2] = 2 * t_abs * t_y - Q[idx1+2];
	    Q[idx2+3] = Q[idx1+3];      
	    Q[idx2+4] = Q[idx1+4]
	      - 0.5 * (Q[idx1+1]*Q[idx1+1] + Q[idx1+2]*Q[idx1+2] -
		       Q[idx2+1]*Q[idx2+1] + Q[idx2+2]*Q[idx2+2]) / Q[idx2];

	  }
	  // -----------------------------------------------------------------------

	  t_abs = uu * t_x + vv * t_y;

	  uu = t_abs * t_x;
	  vv = t_abs * t_y;
      
	  idx1 = blk_info.get_idx_Q(0, i, 0, k);
      
	  Q[idx1  ] = rr;
	  Q[idx1+1] = rr * uu;
	  Q[idx1+2] = rr * vv;
	  Q[idx1+3] = rr * ww;
	  Q[idx1+4] = pp / gam1 + 0.5 * rr * (uu * uu + vv * vv + ww * ww);
#endif

	}

	for(index_type lj = 0; lj <= NG; lj++) {

	  idx2 = blk_info.get_idx_Q(0, i, -lj, k);

	  value_type rr = Q[idx2  ];
	  value_type uu = Q[idx2+1] / rr;
	  value_type vv = Q[idx2+2] / rr;
	  value_type ww = Q[idx2+3] / rr;
	  value_type pp = gam1 * (Q[idx2+4] - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

	  idx1 = blk_info.get_idx(i, -lj, k);

	  rho[idx1] = rr;	
	  u[idx1] = uu;
	  v[idx1] = vv;
	  w[idx1] = ww;
	  p[idx1] = pp;

#ifndef IS_INVISCID
	  T[idx1] = blk_info.gM2 * pp / rr;
#endif

	}

      }

    }

    // xi=i, eta=JM, zeta=k
    if (JM-1 == j) {

      if ((0 < k) && (KM-1 > k)) {

	idx2 = blk_info.get_idx_Q(0, i, JM-1, k);

	value_type rr = Q[idx2  ];
	value_type uu = Q[idx2+1] / rr;
	value_type vv = Q[idx2+2] / rr;
	value_type ww = Q[idx2+3] / rr;
	value_type pp = gam1 * (Q[idx2+4] - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

	for(size_type lj = 0; lj < NG; lj++) {

	  idx1 = blk_info.get_idx(i, JM-1+lj, k);

	  rho[idx1] = rr;	
	  u[idx1] = uu;
	  v[idx1] = vv;
	  w[idx1] = ww;
	  p[idx1] = pp;

#ifndef IS_INVISCID
	  T[idx1] = blk_info.gM2 * pp / rr;
#endif

	}

      }
	
    }
    __syncthreads();

    // xi=i, eta=j, zeta=1
    if (0 == k) {

      idx2 = blk_info.get_idx_Q(0, i, j, 1);

      for(index_type lk = 0; lk <= NG; lk++) {

	idx1 = blk_info.get_idx_Q(0, i, j, -lk);

	for(size_type i_eq = 0; i_eq < NEQ; i_eq++) {
	  Q[idx1 + i_eq] = Q[idx2 + i_eq];
	}

      }

      value_type rr = Q[idx2  ];
      value_type uu = Q[idx2+1] / rr;
      value_type vv = Q[idx2+2] / rr;
      value_type ww = Q[idx2+3] / rr;
      value_type pp = gam1 * (Q[idx2+4] - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

      for(index_type lk = 0; lk <= NG; lk++) {

	idx1 = blk_info.get_idx(i, j, -lk);

	rho[idx1] = rr;	
	u[idx1] = uu;
	v[idx1] = vv;
	w[idx1] = ww;
	p[idx1] = pp;

#ifndef IS_INVISCID
	T[idx1] = blk_info.gM2 * pp / rr;
#endif

      }

    }
    
    // xi=i, eta=j, zeta=KM
    if (KM-1 == k) {

      idx2 = blk_info.get_idx_Q(0, i, j, KM-2);

      for(size_type lk = 0; lk <= NG; lk++) {

	idx1 = blk_info.get_idx_Q(0, i, j, KM-1+lk);

	for(size_type i_eq = 0; i_eq < NEQ; i_eq++) {
	  Q[idx1 + i_eq] = Q[idx2 + i_eq];
	}

      }

      value_type rr = Q[idx2  ];
      value_type uu = Q[idx2+1] / rr;
      value_type vv = Q[idx2+2] / rr;
      value_type ww = Q[idx2+3] / rr;
      value_type pp = gam1 * (Q[idx2+4] - 0.5 * rr * (uu * uu + vv * vv + ww * ww));

      for(size_type lk = 0; lk <= NG; lk++) {

	idx1 = blk_info.get_idx(i, j, KM-1+lk);

	rho[idx1] = rr;	
	u[idx1] = uu;
	v[idx1] = vv;
	w[idx1] = ww;
	p[idx1] = pp;

#ifndef IS_INVISCID
	T[idx1] = blk_info.gM2 * pp / rr;
#endif

      }

    }

  }

}
