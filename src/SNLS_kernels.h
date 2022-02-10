#pragma once

#include "SNLS_base.h"
#include "SNLS_TrDelta.h"

#include <stdlib.h>
#include <iostream>
#ifdef __cuda_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

namespace snls {

/** Helper templates to ensure compliant CRJ implementations */
template<typename CRJ, typename = void>
struct has_valid_computeRJ : std::false_type { static constexpr bool value = false;};

template<typename CRJ>
struct has_valid_computeRJ <
   CRJ,typename std::enable_if<
       std::is_same<
           decltype(std::declval<CRJ>().computeRJ(std::declval<double* const>(), std::declval<double* const>(),std::declval<const double *>())),
           bool  
       >::value
       ,
       void
   >::type
>: std::true_type { static constexpr bool value = true;};

template<typename CRJ, typename = void>
struct has_ndim : std::false_type { static constexpr bool value = false;};

template<typename CRJ>
struct has_ndim <
   CRJ,typename std::enable_if<
       std::is_same<
           decltype(CRJ::nDimSys),
           const int  
       >::value
       ,
       void
   >::type
>: std::true_type { static constexpr bool value = true;};

template<int nDim>
__snls_hdev__
inline
void dogleg(const double delta,
            const double res_0,
            const double nr_norm,
            const double Jg_2,
            const double* const grad,
            const double* const nrStep,
            double* const delx,
            double* const x,
            double &pred_resid,
            bool &use_nr,
            #ifdef __cuda_host_only__
            std::ostream* _os
            #else
            char* _os // do not use
            #endif
            ) {

   // No need to do any other calculations if this condition is true
   if ( nr_norm <= delta ) {
      // use Newton step
      use_nr = true ;

      for (int iX = 0; iX < nDim; ++iX) {
         delx[iX] = nrStep[iX] ;
      }
      pred_resid = 0e0 ;

#ifdef __cuda_host_only__
      if ( _os != nullptr ) {
         *_os << "trying newton step" << std::endl ;
      }
#endif
   }
   // Find Cauchy point
   else {
      // If we didn't reject things this is the only thing that needs to be updated
      // everything else we should have the info to recompute
      // The nice thing about recomputing is that we can actually define the variables as const
      // to help the compiler out.

      const double norm2_grad = snls::linalg::dotProd<nDim>(grad, grad);
      const double norm_grad = sqrt( norm2_grad );

      const double alpha = (Jg_2 > 0.0) ? (norm2_grad / Jg_2) : 1.0;
      const double norm_grad_inv = (norm_grad > 0.0) ? (1.0 / norm_grad) : 1.0;
      const double norm_s_sd_opt = alpha * norm_grad;

      // step along the dogleg path
      if ( norm_s_sd_opt >= delta ) {

         // use step along steapest descent direction
         {
            const double factor = -delta * norm_grad_inv;
            for (int iX = 0; iX < nDim; ++iX) {
               delx[iX] = factor * grad[iX] ;
            }
         }

         {
            const double val = -(delta * norm_grad) + 0.5 * delta * delta * Jg_2 * (norm_grad_inv * norm_grad_inv);
            pred_resid = sqrt(fmax(2.0 * val + res_0 * res_0, 0.0));
         }

#ifdef __cuda_host_only__
         if ( _os != nullptr ) {
            *_os << "trying step along first leg" << std::endl ;
         }
#endif

      }
      else{

         double beta;
         // Scoping this set of calculations
         {
            double qb = 0.0;
            double qa = 0.0;
            for (int iX = 0; iX < nDim; ++iX) {
               double p_iX = nrStep[iX] + alpha * grad[iX];
               qa += p_iX * p_iX;
               qb += p_iX * grad[iX];
            }
            // Previously qb = (-p^t g / ||g||) * alpha * ||g|| * 2.0
            // However, we can see that this simplifies a bit and also with the beta term
            // down below we there's a common factor of 2.0 that we can eliminate from everything
            qb *= alpha;
            // qc and beta depend on delta
            //
            const double qc = norm_s_sd_opt * norm_s_sd_opt - delta * delta;
            beta = (qb + sqrt(qb * qb - qa * qc)) / qa;
         }
#if defined(SNLS_DEBUG) && defined(SNLS_EXTRA_DEBUG_CHECKS)
         if ( beta > 1.0 || beta < 0.0 ) {
            SNLS_FAIL(__func__, "beta not in [0,1]") ;
         }
#endif
         beta = fmax(0.0, fmin(1.0, beta)) ; // to deal with any roundoff

         // delx[iX] = alpha*ngrad[iX] + beta*p[iX] = beta*nrStep[iX] - (1.0-beta)*alpha*grad[iX]
         //
         {
            const double omb  = 1.0 - beta;
            const double omba = omb * alpha;
            for (int iX = 0; iX < nDim; ++iX) {
               delx[iX] = beta * nrStep[iX] - omba * grad[iX];
            }
            const double res_cauchy = (Jg_2 > 0.0) ? (sqrt(fmax(res_0 * res_0 - alpha * norm2_grad, 0.0))) : res_0;
            pred_resid = omb * res_cauchy;
         }

#ifdef __cuda_host_only__
         if ( _os != nullptr ) {
            *_os << "trying step along second leg" << std::endl ;
         }
#endif
      } // if norm_s_sd_opt >= delta
   } // use_nr

   // update x here to keep in line with batch version
   for (int iX = 0; iX < nDim; iX++) {
      x[iX] += delx[iX];
   }
}// end non-batch dogleg

template<int nDim>
__snls_hdev__
inline
void updateDelta(const TrDeltaControl* const deltaControl,
                 const double* const residual,
                 const double res_0,
                 const double pred_resid,
                 const double nr_norm,
                 const double tolerance,
                 const bool use_nr,
                 const bool rjSuccess,
                 double& delta,
                 double& res,
                 double& rhoLast,
                 bool& reject_prev,
                 SNLSStatus_t& status,
#ifdef __cuda_host_only__
                 std::ostream* os
#else
                 char* os // do not use
#endif
                 ) 
{

   if ( !(rjSuccess) ) {
      // got an error doing the evaluation
      // try to cut back step size and go again
      bool deltaSuccess = deltaControl->decrDelta(os, delta, nr_norm, use_nr );
      if ( ! deltaSuccess ) {
         status = deltaFailure ;
         return; // while ( _nIters < _maxIter )
      }
      reject_prev = true;
   }
   else {
      res = snls::linalg::norm<nDim>(residual);
#ifdef __cuda_host_only__
      if ( os != nullptr ) {
         *os << "res = " << res << std::endl ;
      }
#endif

      // allow to exit now, may have forced one iteration anyway, in which
      // case the delta update can do funny things if the residual was
      // already very small

      if ( res < tolerance ) {
#ifdef __cuda_host_only__
         if ( os != nullptr ) {
            *os << "converged" << std::endl ;
         }
#endif
         status = converged ;
         return; // while ( _nIters < _maxIter )
      }

      {
         bool deltaSuccess = deltaControl->updateDelta(os,
                                                      delta, res, res_0, pred_resid,
                                                      reject_prev, use_nr, nr_norm, rhoLast) ;
         if ( ! deltaSuccess ) {
            status = deltaFailure ;
            return; // while ( _nIters < _maxIter ) 
         }
      }
      // Could also potentially include the reject previous portion in here as well
      // if we want to keep this similar to the batch version of things
   }
} // end update delta

} // end snls namespace