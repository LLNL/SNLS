// -*-c++-*-

#ifndef SNLS_BASE_H
#define SNLS_BASE_H

#include "SNLS_config.h"
#include "SNLS_gpu_portability.h"
#include "SNLS_port.h"

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
   linearSolveFailure = -110,
   unset              = -200
} SNLSStatus_t ;

__snls_hdev__
inline
bool
isConverged( SNLSStatus_t status ) { return (status >= 0 ? true : false) ; };

} // namespace snls

#endif  // SNLS_BASE_H
