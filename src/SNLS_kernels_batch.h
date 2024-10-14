#pragma once

#include "SNLS_base.h"

#include <stdlib.h>
#include <iostream>
#ifdef __snls_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

#if defined(SNLS_RAJA_PORT_SUITE)
#include "SNLS_linalg.h"
#include "RAJA/RAJA.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/managed_ptr.hpp"
#include "SNLS_device_forall.h"
#include "SNLS_view_types.h"
#include "SNLS_memory_manager.h"
#include "SNLS_TrDelta.h"


namespace snls {
namespace batch {

/** Helper templates to ensure compliant CRJ implementations */
template<typename CRJ, typename = void>
struct has_valid_computeRJ : std::false_type { static constexpr bool value = false;};

template<typename CRJ>
struct has_valid_computeRJ <
   CRJ,typename std::enable_if<
       std::is_void<
           decltype(std::declval<CRJ>().computeRJ(std::declval<rview2d&>(), std::declval<rview3d&>(), std::declval<const rview2d&>(), 
                    std::declval<rview1b&>(), std::declval<const chai::ManagedArray<SNLSStatus_t>&>(),
                    std::declval<const int>(), std::declval<const int>())) 
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

// So this function computes the delta x and then updates the solution variable for a batch of data
// The user is responsible for providing a potentially updated gradient, Jg_2, nrStep terms.
template<int nDim>
__snls_host__
inline
void dogleg(const int offset,
            const int batch_size,
            const chai::ManagedArray<SNLSStatus_t> &status, // makes use of offset
            const rview1d &delta, // makes use of offset
            const rview1d &res_0,
            const rview1d &nr_norm,
            const rview1d &Jg_2,
            const rview2d &grad,
            const rview2d &nrStep,
            rview2d &delx,
            rview2d &x, // makes use of offset
            rview1d &pred_resid,
            rview1b &use_nr
            ) 
{
   snls::forall<SNLS_GPU_BLOCKS>(0, batch_size, [=] __snls_hdev__ (int i)
   {
      // this breaks out of the internal lambda and is essentially a loop continue
      // This check is to prevent the solver from further updating  the solution after a point has been solved/failed
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
      }
      // Find Cauchy point
      else {
         // If we didn't reject things this is the only thing that needs to be updated
         // everything else we should have the info to recompute
         // The nice thing about recomputing is that we can actually define the variables as const
         // to help the compiler out.

         const double norm2_grad = snls::linalg::dotProd<nDim>(&grad.get_data()[i * nDim], &grad.get_data()[i * nDim]);
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
               pred_resid(i) = sqrt(fmax(2.0 * val + res_0(i) * res_0(i), 0.0));
            }
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
                  delx(i, iX) = beta * nrStep(i, iX) - omba * grad(i, iX);
               }
               const double res_cauchy = (Jg_2(i) > 0.0) ? (sqrt(fmax(res_0(i) * res_0(i) - alpha * norm2_grad, 0.0))) : res_0(i);
               pred_resid(i) = omb * res_cauchy;
            }
         } // if norm_s_sd_opt >= delta
      } // use_nr

      // update x here as we want to avoid additional kernel calls
      for (int iX = 0; iX < nDim; iX++) {
         x(i + offset, iX) += delx(i, iX);
      }
   });
} // end batch dogleg

// For certain solvers we might want to eventually have a version of this that makes use of rhoLast
// as it can be used to determine if the solver is converging slow or not
template<int nDim>
__snls_host__
inline
void updateDelta(const int offset,
                 const int batch_size,
                 const int mfevals,
                 chai::managed_ptr<TrDeltaControl>& deltaControl, // makes use of offset
                 const rview2d &residual,
                 const rview1d &pred_resid,
                 const rview1d &nr_norm,
                 const rview1b &use_nr,
                 const rview1b &rjSuccess,
                 const double tolerance,
                 rview2d &delx,
                 rview2d &x, // makes use of offset
                 rview1d res_0,
                 rview1d res, // makes use of offset
                 rview1d delta, // makes use of offset
                 rview1b reject_prev,
                 chai::ManagedArray<int> fevals, // makes use of offset
                 chai::ManagedArray<SNLSStatus_t> &status // makes use of offset
                 ) 
{
   // The below set of fused kernels compute the updated delta for the step size,
   // reject the previous solution if the computeRJ up above failed,
   // and updates the res0 if the solution is still unconverged.
   snls::forall<SNLS_GPU_BLOCKS>(0, batch_size, [=] __snls_hdev__ (int i)
   {
      // Update the delta kernel
      // this breaks out of the internal lambda and is essentially a loop continue
      // This check is to prevent the solver from further updating  the solution after a point has been solved/failed
      if( status[i + offset] != SNLSStatus_t::unConverged){ return; }
      reject_prev(i) = false;
      if ( !(rjSuccess(i)) ) {
         // got an error doing the evaluation
         // try to cut back step size and go again
         bool deltaSuccess = deltaControl->decrDelta(nullptr, delta(i + offset), nr_norm(i), use_nr(i) ) ;
         if ( ! deltaSuccess ) {
            status[i + offset] = deltaFailure;
            fevals[i + offset] = mfevals;
            return; // equivalent to a continue in a while loop
         }
         reject_prev(i) = true;
      }
      else {
         res(i + offset) = snls::linalg::norm<nDim>(&residual.get_data()[i * nDim]);
         // allow to exit now, may have forced one iteration anyway, in which
         // case the delta update can do funny things if the residual was
         // already very small 
         if ( res(i + offset) < tolerance ) {
            status[i + offset] = converged ;
            fevals[i + offset] = mfevals;
            return; // equivalent to a continue in a while loop
         }
         {
            bool deltaSuccess = deltaControl->updateDelta(nullptr,
                                                         delta(i + offset), res(i + offset), res_0(i), pred_resid(i),
                                                         reject_prev(i), use_nr(i), nr_norm(i)) ;
            if ( ! deltaSuccess ) {
               status[i + offset] = deltaFailure;
               fevals[i + offset] = mfevals;
               return; // equivalent to a continue in a while loop
            }
         }
      }
         // end of updating the delta
      // rejects the previous solution if things failed earlier
      if ( reject_prev(i) ) { 
         res(i + offset) = res_0(i);
         for(int iX = 0; iX < nDim; iX++) {
            x(i + offset, iX) -= delx(i, iX);
         }
      }
      // update our res_0 for the next iteration
      res_0(i) = res(i + offset);
   });

} // end batch update delta

} // end batch namespace
#endif
} // end snls namespace
