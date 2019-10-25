// -*-c++-*-

#ifndef SNLS_BASE_H
#define SNLS_BASE_H

#include "SNLS_cuda_portability.h"
#include "SNLS_port.h"

namespace snls {

typedef enum {
   convByBracket      =  1,
   converged          =  0,
   initEvalFailure    = -2,
   evalFailure        = -3,
   unConverged        = -10,
   deltaFailure       = -20,
   algFailure         = -100,
   bracketFailure     = -101,
   unset              = -200
} SNLSStatus_t ;

__snls_hdev__
inline
bool
isConverged( SNLSStatus_t status ) { return (status >= 0 ? true : false) ; };

} // namespace snls

#endif  // SNLS_BASE_H
