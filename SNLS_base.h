// -*-c++-*-

#ifndef SNLS_BASE_H
#define SNLS_BASE_H

namespace snls {

typedef enum {
   converged          =  0,
   initEvalFailure    = -2,
   evalFailure        = -3,
   unConverged        = -10,
   deltaFailure       = -20,
   algFailure         = -100,
   unset              = -200
} SNLSStatus_t ;

} // namespace snls

#endif  // SNLS_BASE_H
