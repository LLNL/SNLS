#pragma once

#include "SNLS_base.h"

#include <stdlib.h>
#include <iostream>
#ifdef __cuda_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

#include "SNLS_lup_solve.h"

#if defined(SNLS_RAJA_PERF_SUITE)
#include "SNLS_device_forall.h"
#include "SNLS_memory_manager.h"
#include "RAJA/RAJA.hpp"
#include "chai/ManagedArray.hpp"
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

}

#if defined(SNLS_RAJA_PERF_SUITE)
namespace batch {
// This occurs outside the dogleg step as we want this to be generic across different methods whether
// they use something like a QR solver or LU solve for such things
// Additionally, this enables us if we have a QR decomposition of things to just make use of the 
// this->computeNewtonStep( &_Jacobian.data[i * _nXnDim], &_residual.data[i * _nDim], &nrStep.data[i * _nDim] );
__snls_hdev__
inline
void dogleg(const int offset,
            const int batch_size,
            const chai::managedArray<bool> &status,
            const rview3d &Jacobian,
            const rview2d &residual,
            rview2d &delx,
            rview2d &grad,
            rview2d &nrStep,
            rview1d &norm_grad,
            rview1d &Jg2,
            rview1d &alpha,
            rview1d &res_cauchy,
            rview1d &res_0,
            rview1d &norm_s_sd_opt,
            rview1d &norm_grad_inv,
            rview1d &nr2norm,
            rview1d &qa,
            rview1d &qb,
            rview1d &pred_resid,
            rview1b &reject_prev,
            rview1b &use_nr
            ) {
   // Start of compute kernel
   // This loop contains the 
   // cauchy point calculations, update delta x, and 
   // update of the solution steps.
   // These could be broken down to 2 different 
   // compute kernels. However, it is more performant to
   // fuse all of them into one kernel.
   SNLS_FORALL_T(i, 256, 0, batch_size,
   { // start of cauchy point calculations
      // this breaks out of the internal lambda and is essentially a loop continue
      if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
      if ( !reject_prev(i) ) {
         //
         // have a newly accepted solution point
         // compute information for step determination
         // fix me: this won't work currently
         this->computeSysMult( &_Jacobian.data[i * _nXnDim], &_residual.data[i * _nDim], &grad.data[i * _nDim], true );
         // used to keep :
         //	ngrad[iX] = -grad[iX] ;
         // 	nsd[iX]   = ngrad[iX] * norm_grad_inv ; 
         
         // find Cauchy point
         //
         {
            double norm2_grad = this->normvecSq( &grad.data[i * _nDim] ) ;
            norm_grad(i) = sqrt( norm2_grad ) ;
            {
               double ntemp[_nDim] ; 
               this->computeSysMult( &_Jacobian.data[i * _nXnDim], &grad.data[i * _nDim], &ntemp[0], false ) ; // was -grad in previous implementation, but sign does not matter
               Jg2(i) = this->normvecSq( &ntemp[0] ) ;
            }
            
            alpha(i) = 1e0 ;
            res_cauchy(i) = res_0(i);
            if ( Jg2(i) > 0e0 ) { 
               alpha(i) = norm2_grad / Jg2(i) ;
               res_cauchy(i) = sqrt(fmax(res_0(i) * res_0(i) - alpha(i) * norm2_grad, 0.0)) ;
            }

            // optimal step along steapest descent is alpha*ngrad
            norm_s_sd_opt(i) = alpha(i) * norm_grad(i);
            
            norm_grad_inv(i) = 1e0 ;
            if ( norm_grad(i) > 0e0 ) {
               norm_grad_inv(i) = 1e0 / norm_grad(i);
            }
            
         }

         // Newton Raphson Step occurs outside the dogleg step
         nr2norm(i) = this->normvec( &nrStep.data[i * _nDim] ) ;
         //
         // as currently implemented, J is not factored in-place and computeSysMult no loner works ;
         // or would have to be reworked for PLU=J

         {
            {
               double gam0 = 0e0 ;
               double norm2_p = 0.0 ;
               for (int iX = 0; iX < _nDim; ++iX) {
                  double p_iX = nrStep(i, iX) + alpha(i) * grad(i, iX);
                  norm2_p += p_iX * p_iX ;
                  gam0 += p_iX * grad(i, iX);
               }
               gam0 = -gam0 * norm_grad_inv(i);

               qb(i) = 2.0 * gam0 * norm_s_sd_opt(i);

               qa(i) = norm2_p ;
            }
         }

      } // !reject_prev
      // end cauchy point calculations

      // compute the step given _delta
      //
      // start of calculating size of delta x
      use_nr(i) = false;
      if ( nr2norm(i) <= _delta(i + offset) ) {
         // use Newton step
         use_nr(i) = true ;

         for (int iX = 0; iX < _nDim; ++iX) {
            delx(i, iX) = nrStep(i, iX);
         }
         pred_resid(i) = 0e0 ;

#ifdef __cuda_host_only__
         if ( _os != nullptr ) {
            *_os << "trying newton step" << std::endl ;
         }
#endif
      }
      else { // use_nr

         // step along dogleg path
         if ( norm_s_sd_opt(i) >= _delta(i + offset) ) {
            // use step along steapest descent direction
            {
               double factor = -_delta(i + offset) * norm_grad_inv(i);
               for (int iX = 0; iX < _nDim; ++iX) {
                  delx(i, iX) = factor * grad(i, iX);
               }
            }
            {
               double val = -(_delta(i + offset) * norm_grad(i)) 
                           + 0.5 * _delta(i + offset) * _delta(i + offset) * Jg2(i)
                           * (norm_grad_inv(i) * norm_grad_inv(i));
               pred_resid(i) = sqrt(fmax(2.0 * val + res_0(i) * res_0(i), 0.0)) ;
            }
         }
         else{
            // qc and beta depend on delta
            //
            double qc = norm_s_sd_opt(i) * norm_s_sd_opt(i) - _delta(i + offset) * _delta(i + offset);
            //
            double beta = (-qb(i) + sqrt(qb(i) * qb(i) - 4.0 * qa(i) * qc))/(2.0 * qa(i)) ;
#ifdef SNLS_DEBUG
            if ( beta > 1.0 || beta < 0.0 ) {
               SNLS_FAIL(__func__, "beta not in [0,1]") ;
            }
#endif
            beta = fmax(0.0, fmin(1.0, beta)) ; // to deal with and roundoff

            // delx[iX] = alpha*ngrad[iX] + beta*p[iX] = beta*nrStep[iX] - (1.0-beta)*alpha*grad[iX]
            //
            {
               double omb  = 1.0 - beta ;
               double omba = omb * alpha(i);
               for (int iX = 0; iX < _nDim; ++iX) {
                  delx(i, iX) = beta * nrStep(i, iX) - omba * grad(i, iX);
               }
               pred_resid(i) = omb * res_cauchy(i);
            }
         } // if norm_s_sd_opt >= delta
      } // use_nr
      // end of calculating delta x size
      this->update( &delx.data[i * _nDim], i, offset ) ;
      reject_prev(i) = false ;
   }); // end of batch compute kernel 1
}


}
#endif
} // end snls namespace