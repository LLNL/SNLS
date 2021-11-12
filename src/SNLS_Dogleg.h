#pragma once

#include "SNLS_base.h"

#include <stdlib.h>
#include <iostream>
#ifdef __cuda_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

#if defined(SNLS_RAJA_PERF_SUITE)
// SNLS_base includes all the raja view typedefs
#include "SNLS_device_forall.h"
#endif


namespace snls {

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
#ifdef SNLS_DEBUG
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

   // update x here as we want to avoid additional kernel calls
   for (int iX = 0; iX < nDim; iX++) {
      x[iX] += delx[iX];
   }
}// end non-batch dogleg

#if defined(SNLS_RAJA_PERF_SUITE)
namespace batch {

// So this function computes the delta x and then updates the solution variable for a batch of data
// The user is responsible for providing a potentially updated gradient, Jg_2, nrStep terms.
template<int nDim>
__snls_hdev__
inline
void dogleg(const int offset,
            const int batch_size,
            const chai::managedArray<bool> &status,
            const rview1d &delta,
            const rview1d &res_0,
            const rview1d &nr_norm,
            const rview1d &Jg_2,
            const rview2d &grad,
            const rview2d &nrStep,
            rview2d &delx,
            rview2d &x,
            rview1d &pred_resid,
            rview1b &use_nr,
            #ifdef __cuda_host_only__
            std::ostream* _os
            #else
            char* _os // do not use
            #endif
            ) 
{
   SNLS_FORALL_T(i, 256, 0, batch_size,
   {
      // this breaks out of the internal lambda and is essentially a loop continue
      if( status[i + offset] != SNLSStatus_t::unConverged){ return; }
      // No need to do any other calculations if this condition is true
      use_nr(i) = false;

      if ( nr_norm(i) <= delta(i + offset) ) {
         // use Newton step
         use_nr(i) = true ;

         for (int iX = 0; iX < nDim; ++iX) {
            delx(i, iX) = nrStep(i, iX);
         }
         pred_resid(i) = 0e0 ;

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

         const double norm2_grad = snls::linalg::dotProd<nDim>(&grad.data[i * nDim], &grad.data[i * nDim]);
         const double norm_grad = sqrt( norm2_grad );

         const double alpha = (Jg_2(i) > 0.0) ? (norm2_grad / Jg_2(i)) : 1.0;
         const double norm_grad_inv = (norm_grad > 0.0) ? (1.0 / norm_grad) : 1.0;
         const double norm_s_sd_opt = alpha * norm_grad;

         // step along the dogleg path
         if ( norm_s_sd_opt >= delta(i + offset) ) {

            // use step along steapest descent direction
            {
               const double factor = -delta(i + offset) * norm_grad_inv;
               for (int iX = 0; iX < nDim; ++iX) {
                  delx(i, iX) = factor * grad(i, iX);
               }
            }

            {
               const double val = -(delta(i + offset) * norm_grad) + 0.5 * delta(i + offset) * delta(i + offset) * Jg_2(i) * (norm_grad_inv * norm_grad_inv);
               pred_resid = sqrt(fmax(2.0 * val + res_0(i) * res_0(i), 0.0));
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
                  double p_iX = nrStep(i, iX) + alpha * grad(i, iX);
                  qa += p_iX * p_iX;
                  qb += p_iX * grad(i, iX);
               }
               // Previously qb = (-p^t g / ||g||) * alpha * ||g|| * 2.0
               // However, we can see that this simplifies a bit and also with the beta term
               // down below we there's a common factor of 2.0 that we can eliminate from everything
               qb *= alpha;
               // qc and beta depend on delta
               //
               const double qc = norm_s_sd_opt * norm_s_sd_opt - delta(i + offset) * delta(i + offset);
               beta = (qb + sqrt(qb * qb - qa * qc)) / qa;
            }
   #ifdef SNLS_DEBUG
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
                  delx(i, iX) = beta * nrStep(i, iX) - omba * grad(i, iX);
               }
               const double res_cauchy = (Jg_2(i) > 0.0) ? (sqrt(fmax(res_0(i) * res_0(i) - alpha * norm2_grad, 0.0))) : res_0(i);
               pred_resid(i) = omb * res_cauchy;
            }

   #ifdef __cuda_host_only__
            if ( _os != nullptr ) {
               *_os << "trying step along second leg" << std::endl ;
            }
   #endif
         } // if norm_s_sd_opt >= delta
      } // use_nr

      // update x here as we want to avoid additional kernel calls
      for (int iX = 0; iX < nDim; iX++) {
         x(i + offset, iX) += delx(i, iX);
      }
   });
} // end batch dogleg
} // end batch namespace
#endif
} // end snls namespace