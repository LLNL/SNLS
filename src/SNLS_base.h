// -*-c++-*-

#ifndef SNLS_BASE_H
#define SNLS_BASE_H

#include "SNLS_config.h"
#include "SNLS_cuda_portability.h"
#include "SNLS_port.h"

#if defined(SNLS_RAJA_PERF_SUITE)
#include "RAJA/RAJA.hpp"
#endif

#define SNLS_NN_INDX(p, q, nDim) (p) * (nDim) + (q)
#define SNLS_NM_INDX(p, q, pDim, qDim) (p) * (qDim) + (q)

namespace snls {

typedef enum {
   convByBracket      =  1,
   converged          =  0,
   initEvalFailure    = -2,
   evalFailure        = -3,
   unConverged        = -10,
   deltaFailure       = -20,
   unConvergedMaxIter = -30,
   slowJacobian       = -40,
   slowConvergence    = -50,
   algFailure         = -100,
   bracketFailure     = -101,
   unset              = -200
} SNLSStatus_t ;

__snls_hdev__
inline
bool
isConverged( SNLSStatus_t status ) { return (status >= 0 ? true : false) ; };

#if defined(SNLS_RAJA_PERF_SUITE)
// useful RAJA views for our needs
typedef RAJA::View<bool, RAJA::Layout<1> > rview1b;
typedef RAJA::View<double, RAJA::Layout<1> > rview1d;
typedef RAJA::View<double, RAJA::Layout<2> > rview2d;
typedef RAJA::View<double, RAJA::Layout<3> > rview3d;
#endif

} // namespace snls

#endif  // SNLS_BASE_H
