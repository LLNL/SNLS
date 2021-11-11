// -*-c++-*-

#ifndef SNLS_TRDLDG_BATCH_H
#define SNLS_TRDLDG_BATCH_H

#include "SNLS_base.h"
#if defined(SNLS_RAJA_PERF_SUITE)
#include "SNLS_linalg.h"
#include "SNLS_lup_solve.h"
#include "SNLS_TrDelta.h"
#include "SNLS_device_forall.h"
#include "SNLS_memory_manager.h"
#include "RAJA/RAJA.hpp"
#include "chai/ManagedArray.hpp"

#include <stdlib.h>
#include <iostream>
#ifdef __cuda_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

#ifndef SNLS_USE_LAPACK
#define SNLS_USE_LAPACK 0
#endif

#if HAVE_LAPACK && SNLS_USE_LAPACK

extern "C" {
   int DGETRF(const int* m, const int* n, double* A, const int* lda, int* ipiv, int* info);
   int DGETRS(const char* trans, const int* n, const int* nrhs, const double* const A, const int* lda,
              const int* const ipiv, double* b, const int* ldb, int* info);
}
#endif

//////////////////////////////////////////////////////////////////////
/// This macro provides a simple way to set-up raja views given a chai
/// array (wdata), the chai execution space (es), and the offset into
/// that chai array for the raja view
#define SNLS_RSETUP(wdata, es, offset) &(wdata.data(es))[offset]

namespace snls {

// useful RAJA views for our needs
typedef RAJA::View<bool, RAJA::Layout<1> > rview1b;
typedef RAJA::View<double, RAJA::Layout<1> > rview1d;
typedef RAJA::View<double, RAJA::Layout<2> > rview2d;
typedef RAJA::View<double, RAJA::Layout<3> > rview3d;

namespace batch{

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

// trust region type solver, dogleg approximation
// for dense general Jacobian matrix
template< class CRJ >
class SNLSTrDlDenseG_Batch 
{
   public:
      static_assert(has_valid_computeRJ<CRJ>::value, "The CRJ implementation in SNLSTrDlDenseG_Batch needs to implement void computeRJ( rview2d &r, rview3d &J, const rview2d &x,"
                                                      " rview1b &rJSuccess, const chai::ManagedArray<SNLSStatus_t> &status, const int offset, const int nbatch )");
      static_assert(has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");

   public:
   /// constructor which requires the number of points to be set
   /// or else it defaults to just using 1 for batch solves
   SNLSTrDlDenseG_Batch(CRJ &crj, uint npts = 1, uint dflt_initial_batch = 50000) :
               _crj(crj),
               _mfevals(0), _nIters(0), _nJFact(0),
               _deltaControl(nullptr),
               _outputLevel(0),
               _os(nullptr),
               _npts(npts),
               _initial_batch_size(dflt_initial_batch),
               _x(nullptr, npts, CRJ::nDimSys),
               _res(nullptr, npts),
               _delta(nullptr, npts),
               _residual(nullptr, dflt_initial_batch, CRJ::nDimSys),
               _Jacobian(nullptr, dflt_initial_batch, CRJ::nDimSys, CRJ::nDimSys),
               _rjSuccess(nullptr, dflt_initial_batch)
   {
      init();
   };

   void init() {
      // Create all of the working arrays at initializations to reduce the number
      // of mallocs we need to create through out the life of the solver object
      memoryManager& mm = memoryManager::getInstance();
      const auto es = snls::Device::GetCHAIES();
      // Values multiplied by _initial_batch_size are related to the various
      // working arrays needed such as the residual and Jacobian matrices
      // Values multiplied _npts are those useful to the user such as the
      // solution variable _x, _delta, and l2 norm of residual (_res)
      // Working arrays used in the solve function are:
      // 1d arrays [_initial_batch_size]
      // res_0, nr2norm, alpha, norm_s_sd_opt, norm_grad
      // norm_grad_inv, qa, qb, Jg2, res_cauchy, pred_resid
      // 2d arrays [_initial_batch_size, CRJ::nDimSys]
      // nrStep, grad, delx, solx, _residual
      // 3d arrays [_initial_batch_size, CRJ::nDimSys, CRJ::nDimSys]
      // _Jacobian
      const int num_allocs = _initial_batch_size * (11 + (5 * CRJ::nDimSys) + CRJ::nDimSys * CRJ::nDimSys) + 
                             _npts * (2 + CRJ::nDimSys);
      wrk_data = mm.allocManagedArray<double>(num_allocs);
      _status = mm.allocManagedArray<SNLSStatus_t>(_npts);
      _fevals = mm.allocManagedArray<int>(_npts);

      // These are boolean working arrays used in the solve routine
      // _rjSuccess is the only one accessible for external uses
      // 1d arrays [_initial_batch_size]
      // _rjSuccess, use_nr, reject_prev
      wrkb_data = mm.allocManagedArray<bool>(3 * _initial_batch_size);
      _rjSuccess.set_data(SNLS_RSETUP(wrkb_data, es, 0));

      int offset = 0;
      _offset_work = _npts * (2 + CRJ::nDimSys)
                   + _initial_batch_size * (CRJ::nDimSys + CRJ::nDimSys * CRJ::nDimSys);
      _x.set_data(SNLS_RSETUP(wrk_data, es, offset));
      offset += _npts * CRJ::nDimSys;
      _res.set_data(SNLS_RSETUP(wrk_data, es, offset));
      offset += _npts;
      _delta.set_data(SNLS_RSETUP(wrk_data, es, offset));
      offset += _npts;

      _residual.set_data(SNLS_RSETUP(wrk_data, es, offset));
      offset += _initial_batch_size * CRJ::nDimSys;
      _Jacobian.set_data(SNLS_RSETUP(wrk_data, es, offset));

      SNLS_FORALL(i, 0, _npts, {
         _fevals[i] = 0;
         for (int j = 0; j < CRJ::nDimSys; j++){
            _x(i, j) = 0.0;
         }
         _delta(i) = 1e8;
         _res(i) = 1e20;
         _status[i] = SNLSStatus_t::unConverged;
      });
   };
   /// destructor needs to dealloc the wrk_data, wrkb_data, _feval, and _status variables
   ~SNLSTrDlDenseG_Batch() {
      wrk_data.free();
      wrkb_data.free();
      _fevals.free();
      _status.free();
   };

   public:
      CRJ &_crj ;
      static const int _nDim = CRJ::nDimSys ;
      /// The size of the nonlinear system of equations being solved for
      int     getNDim   () const { return(_nDim   ); };
      /// Returns the maximum of function evaluations across all the nonlinear system solves
      int     getMaxNFEvals() const { return(_mfevals ); };
      /// Returns the function evaluation array for each point
      const chai::ManagedArray<int> getNFEvals() const { return _fevals; };
      /// Returns the size of the delta step used as part of the dogleg solve of the
      /// PDE
      const rview1d& getDelta() const { return _delta; };
      /// Returns the L2 norm of the residual vector of the nonlinear systems being solved for
      const rview1d& getRes() const { return _res; };
      /// The working array for the residual vector
      /// Useful if one wants to do a computeRJ call outside of the solve func
      /// It has dimensions of (_intial_batch_size, _nDim) and follows c array striding
      rview2d& getResidualVec() { return _residual; }
      /// The working array for the jacobian matrix
      /// Useful if one wants to do a computeRJ call outside of the solve func
      /// It has dimensions of (_intial_batch_size, _nDim, _nDim) and follows c array striding
      rview3d& getJacobianMat() { return _Jacobian; }
      /// The working array for the rjSuccess vector
      /// Useful if one wants to do a computeRJ call outside of the solve func
      /// It has dimensions of (_intial_batch_size) and follows c array striding
      rview1b& getRJSuccessVec() { return _rjSuccess; }

      /// setX can be used to set the initial guess for all of the points used in the batch job
      inline void setX( const double* const x) {
         SNLS_FORALL(ipts, 0, _npts, {
            for (int iX = 0; iX < _nDim; ++iX) {
               _x(ipts, iX) = x[ipts * _nDim + iX] ;
            }
         });
      }

      /// setX can be used to set the initial guess for all of the points used in the batch job
      inline void setX(const chai::ManagedArray<double> &x) {
         SNLS_FORALL(ipts, 0, _npts, {
            for (int iX = 0; iX < _nDim; ++iX) {
               _x(ipts, iX) = x[ipts * _nDim + iX] ;
            }
         });
      }

      /// getX can be used to get solution for all of the points used in the batch job
      inline void getX( double* const x) const {
         SNLS_FORALL(ipts, 0, _npts, {
            for (int iX = 0; iX < _nDim; ++iX) {
               x[ipts * _nDim + iX] = _x(ipts, iX);
            }
         });
      };

      /// getX can be used to get solution for all of the points used in the batch job
      inline void getX( chai::ManagedArray<double> &x) const {
         SNLS_FORALL(ipts, 0, _npts, {
            for (int iX = 0; iX < _nDim; ++iX) {
               x[ipts * _nDim + iX] = _x(ipts, iX);
            }
         });
      };

      /**
       * Must call setupSolver before calling solve
       */
      void   setupSolver(int maxIter,
                         double tolerance,
                         TrDeltaControl * deltaControl,
                         int outputLevel=0) {
         SNLS_FORALL(i, 0, _npts, {
            _status[i] = SNLSStatus_t::unConverged ;
            _fevals[i] = 0;
         });

         _maxIter = maxIter ;
         _tolerance = tolerance ;

         _deltaControl = deltaControl ;

         this->setOutputlevel( outputLevel ) ;

      }

      void   setOutputlevel( int    outputLevel ) {
         _outputLevel = outputLevel ;
         _os          = nullptr ;
         //
         if ( _outputLevel > 0 ) {
            _os = &(std::cout) ;
         }
      }

      /// solve returns bool for whether or not all systems solved successfully
      ///
      /// on exit, _res is consistent with _x
      bool solve() {
         if ( _deltaControl == nullptr ) {
            SNLS_FAIL("solve", "_deltaControl not set") ;
         }
         
         SNLS_FORALL(i, 0, _npts, {
            _status[i] = SNLSStatus_t::unConverged;
            _fevals[i] = 0;
         });

         const int numblks = (_npts + _initial_batch_size - 1)/ _initial_batch_size;

         int offset = 0;

         // All of our temporary variables needed for the batch solve
         // We make use of the working arrays created initially to reuse
         // memory if multiple solves are called during the life of this object
         const auto es = snls::Device::GetCHAIES();
         rview1b use_nr(SNLS_RSETUP(wrkb_data, es, _initial_batch_size), _initial_batch_size);
         rview1b reject_prev(SNLS_RSETUP(wrkb_data, es, 2 * _initial_batch_size), _initial_batch_size);

         int woffset = _offset_work;
         const int off2d = _initial_batch_size * _nDim;
         const int off1d = _initial_batch_size;

         // 11 * batch_size + 5 * (batch_size * ndim) + 1 * batch_size * ndim*ndim
         // internal memory usage for solver...
         // if ndim = 8 and batch_size = 500k then this is roughly 460MBs
         // We're probably close to 500 MBs total then for the solver at size
         // if batch_size and npts are equal
         rview1d res_0(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d nr2norm(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d alpha(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d norm_s_sd_opt(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d norm_grad(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d norm_grad_inv(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d qa(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d qb(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d Jg2(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d res_cauchy(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d pred_resid(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;

         rview2d nrStep(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size, _nDim);
         woffset += off2d;
         rview2d grad(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size, _nDim);
         woffset += off2d;
         rview2d delx(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size, _nDim);
         woffset += off2d;
         rview2d solx(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size, _nDim);
         // offset += off2d;

         int m_fevals = 0;
         int m_nJFact = 0;
         int m_nIters = 0;

         for(int iblk = 0; iblk < numblks; iblk++) {

            _mfevals = 0 ;
            _nJFact = 0 ;
            _nIters = 0 ;
            // Checks to see if we're the last bulk and if not use the initial batch size
            // if we are we need to either use initial batch size or the npts%initial_batch_size
            const int fin_batch = (_npts % _initial_batch_size == 0) ? _initial_batch_size : _npts % _initial_batch_size;
            const int batch_size = (iblk != numblks - 1) ? _initial_batch_size : fin_batch;

            // Reinitialize all of our data back to these values
            SNLS_FORALL_T(i, 512, 0, batch_size, {
               _rjSuccess(i) = true;
               reject_prev(i) = false;
               use_nr(i) = false;
               nr2norm(i) = 0.0;
               alpha(i) = 0.0;
               norm_s_sd_opt(i) = 0.0;
               norm_grad(i) = 0.0;
               norm_grad_inv(i) = 0.0;
               qa(i) = 0.0;
               qb(i) = 0.0;

               _delta(i + offset) = _deltaControl->getDeltaInit();
            });

            this->computeRJ(_residual, _Jacobian, _rjSuccess, offset, batch_size); // at _x

            SNLS_FORALL_T(i, 512, 0, batch_size, {
               if ( !(_rjSuccess(i)) ) {
                  _status[i + offset] = SNLSStatus_t::initEvalFailure ;
               }
               // this breaks out of the internal lambda and is essentially a loop continue
               if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  _res(i + offset) = snls::linalg::norm<_nDim>(&_residual.data[i * _nDim]);
                  res_0(i) = _res(i + offset);
            });

            while ( _nIters < _maxIter ) {
               //
               _nIters += 1 ;
               // Start of batch compute kernel 1
               // This loop contains the LU solve/ the 
               // cauchy point calculations, update delta x, and 
               // update of the solution steps.
               // These could be broken down to 3 different 
               // compute kernels. However, it is more performant to
               // fuse all of them into one kernel.
               //
               // This compute kernel is the most sensitive to the number of threads
               // So, the number of threads might have to be varied to achieve
               // optimal performance for a given nonlinear system.
               SNLS_FORALL_T(i, 256, 0, batch_size,
               { // start of cauchy point calculations
                  // this breaks out of the internal lambda and is essentially a loop continue
                  if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  if ( !reject_prev(i) ) {
                     //
                     // have a newly accepted solution point
                     // compute information for step determination
                     // fix me: this won't work currently
                     snls::linalg::matTVecMult<_nDim, _nDim>(&_Jacobian.data[i * _nXnDim], &_residual.data[i * _nDim], &grad.data[i * _nDim]);
                     // used to keep :
                     //	ngrad[iX] = -grad[iX] ;
                     // 	nsd[iX]   = ngrad[iX] * norm_grad_inv ; 
                     
                     // find Cauchy point
                     //
                     {
                        double norm2_grad = snls::linalg::dotProd<_nDim>(&grad.data[i * _nDim], &grad.data[i * _nDim]);
                        norm_grad(i) = sqrt( norm2_grad ) ;
                        {
                           double ntemp[_nDim] ;
                           snls::linalg::matVecMult<_nDim, _nDim>(&_Jacobian.data[i * _nXnDim], &grad.data[i * _nDim], ntemp); // was -grad in previous implementation, but sign does not matter
                           Jg2(i) = snls::linalg::dotProd<_nDim>(ntemp, ntemp);
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

                     this->computeNewtonStep( &_Jacobian.data[i * _nXnDim], &_residual.data[i * _nDim], &nrStep.data[i * _nDim] ) ;
                     nr2norm(i) = snls::linalg::norm<_nDim>( &nrStep.data[i * _nDim] );
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
               // start of batch compute kernel 2
               // This calculates the new residual and jacobian given the updated x value
               this->computeRJ(_residual, _Jacobian, _rjSuccess, offset, batch_size) ; // at _x

               // The below set of fused kernels compute the updated delta for the step size,
               // reject the previous solution if the computeRJ up above failed,
               // and updates the res0 if the solution is still unconverged.
               // start of compute kernel 3
               SNLS_FORALL_T(i, 256, 0, batch_size,
               {
                  // Update the delta kernel
                  // this breaks out of the internal lambda and is essentially a loop continue
                  if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  if ( !(_rjSuccess(i)) ) {
                     // got an error doing the evaluation
                     // try to cut back step size and go again
                     bool deltaSuccess = _deltaControl->decrDelta(_os, _delta(i + offset), nr2norm(i), use_nr(i) ) ;
                     if ( ! deltaSuccess ) {
                        _status[i + offset] = deltaFailure ;
                        _fevals[i + offset] = _mfevals;
                        return; // equivalent to a continue in a while loop
                     }
                     reject_prev(i) = true ;
                  }
                  else {
                     _res(i + offset) = snls::linalg::norm<_nDim>(&_residual.data[i * _nDim]);
                     // allow to exit now, may have forced one iteration anyway, in which
                     // case the delta update can do funny things if the residual was
                     // already very small 
                     if ( _res(i + offset) < _tolerance ) {
                        _status[i + offset] = converged ;
                        _fevals[i + offset] = _mfevals;
                        return; // equivalent to a continue in a while loop
                     }
                     {
                        bool deltaSuccess = _deltaControl->updateDelta(_os,
                                                                     _delta(i + offset), _res(i + offset), res_0(i), pred_resid(i),
                                                                     reject_prev(i), use_nr(i), nr2norm(i)) ;
                        if ( ! deltaSuccess ) {
                           _status[i + offset] = deltaFailure ;
                           _fevals[i + offset] = _mfevals;
                           return; // equivalent to a continue in a while loop
                        }
                     }
                  }
                   // end of updating the delta
                  // rejects the previous solution if things failed earlier
                  if ( reject_prev(i) ) { 
                     _res(i + offset) = res_0(i);
                     this->reject( &delx.data[i * _nDim], i, offset ) ;
                  }
                  // update our res_0 for the next iteration
                  res_0(i) = _res(i + offset);
               }); // end of batch compute kernel 3
               // If this is true then that means all of the batched items
               // either failed or converged, and so we can exit and start the new batched data
               if(status_exit<true, SNLS_GPU_THREADS>(offset, batch_size)) {
                  break;
               }
            } // _nIters < _maxIter

            if(m_fevals < _mfevals) m_fevals = _mfevals;
            if(m_nJFact < _nJFact) m_nJFact = _nJFact;
            if(m_nIters < _nIters) m_nIters = _nIters;

            offset += batch_size;

         } // end of batch loop

         _mfevals = m_fevals;
         _nJFact = m_nJFact;
         _nIters = m_nIters;

         bool converged = status_exit<false, SNLS_GPU_THREADS>(0, _npts);
         return converged ;
      }

      /// Computes the residual, jacobian, and whether or not 
      /// the computation was a success using the current _x
      /// state of the solver. If there is an offset to the 
      /// _x variable that is supplied as well as the current
      /// batch size.
      inline void computeRJ(rview2d &r,
                            rview3d &J,
                            rview1b &rJSuccess,
                            const int offset,
                            const int batch_size) {
         
         _mfevals++ ;
         this->_crj.computeRJ(r, J, _x, rJSuccess, _status, offset, batch_size);
#ifdef SNLS_DEBUG
         // This is not efficient at all but it'll work for debugging cases
         if ( _outputLevel > 2 && _os != nullptr ) {
            // do finite differencing
            // assume system is scaled such that perturbation size can be standard
            auto mm = snls::memoryManager::getInstance();
            chai::ManagedArray<double> cJ_FD = mm.allocManagedArray<double>(_initial_batch_size * _nXnDim);
            // Need to copy the Jacobian data over to a local variable.
            // Since, I'm not aware of a way to just copy a portion of underlying
            // chai::managedArray over to host/device
            chai::ManagedArray<double> cJ = mm.allocManagedArray<double>(_initial_batch_size * _nXnDim);
            const int tot_npts = _initial_batch_size * _nXnDim;
            SNLS_FORALL(i, 0, tot_npts, {
               // If we want to completely avoid inner loops
               // we have to do the following to get the correct
               // indexing into J.
               // Although, we do incur the cost of a few divisions this
               // way. Nonetheless, this sort of way of doing things can
               // really boost performance on the GPU if our compute kernels
               // are friendly towards this sort of thing.
               // const int ipt = i / _nXnDim;
               // const int imod = (i % _nXnDim);
               // const int iX  = imod / _nDim;
               // const int jX  = imod % _nDim;
               // cJ[i] = J(ipt, iX, jX);
               // Alternatively, we can just use the underlying data here.
               cJ[i] = J.data[i]
            });
            // Dummy variable since we got rid of using raw pointers
            // The default initialization has a size of 0 which is what we want.
            rview3d J_dummy(nullptr, 0, 0, 0);
            chai::ManagedArray<double> cr_pert = mm.allocManagedArray<double>(_initial_batch_size * _nDim);
            chai::ManagedArray<double> cx_pert = mm.allocManagedArray<double>(_npts * _nDim);
            const auto es = snls::Device::GetCHAIES();
            rview2d r_pert(SNLS_RSETUP(cr_pert, es, 0), _initial_batch_size, _nDim);
            rview2d x_pert(SNLS_RSETUP(cx_pert, es, 0), _npts, _nDim);
            rview3d J_FD(SNLS_RSETUP(cJ_FD, es, 0), _initial_batch_size, _nDim, _nDim);

            const double pert_val     = 1.0e-7 ;
            const double pert_val_inv = 1.0/pert_val;

            for ( int iX = 0; iX < _nDim ; ++iX ) {
               SNLS_FORALL(iBatch, 0, batch_size, {
                  for ( int jX = 0; jX < _nDim ; ++jX ) {
                     x_pert(offset, jX) = _x(offset, jX);
                  }
                  x_pert(offset, iX) = x_pert(offset, iX) + pert_val;
               });
               this->_crj.computeRJ(r_pert, J_dummy, x_pert, rJSuccess, offset, batch_size);
               const bool success = reduced_rjSuccess<SNLS_GPU_THREADS>(rJSuccess, batch_size);
               if ( !success ) {
                  SNLS_FAIL(__func__, "Problem while finite-differencing");
               }
               SNLS_FORALL(iBatch, 0, batch_size, {
                  for ( int iR = 0; iR < _nDim ; iR++ ) {
                     J_FD(iBatch, iR, iX) = pert_val_inv * ( r_pert(iBatch, iR) - r(iBatch, iR) );
                  }
               });
            }

            // Terribly inefficient here if running on the GPU...
            const double* J_FD_data = &cJ_FD.data(chai::ExecutionSpace::CPU);
            const double* J_data = &cJ.data(chai::ExecutionSpace::CPU);
            for (int iBatch = 0; iBatch < batch_size; iBatch++) {
               const int moff = SNLS_MOFF(iBatch, _nXnDim);
               *_os << "Batch item # " << iBatch << std::endl;
               *_os << "J_an = " << std::endl ; printMatJ( &J_data[moff],    *_os );
               *_os << "J_fd = " << std::endl ; printMatJ( &J_FD_data[moff], *_os );
            }

            // Clean-up the memory these objects use.
            cx_pert.free();
            cr_pert.free();
            cJ_FD.free();
            cJ.free();

            // put things back the way they were ;
            this->_crj.computeRJ(r, J, _x, rJSuccess, offset, batch_size);
         } // _os != nullptr
#endif
      }
   private :
     __snls_hdev__ inline void  computeNewtonStep (double* const       J,
                                      const double* const r,
                                      double* const       newton  ) {
         _nJFact++ ;
#if HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
// This version of the Newton solver uses the LAPACK solver DGETRF() and DGETRS()
// 
// Note that we can replace this with a custom function if there are performance 
// specializations (say for a known fixed system size)
         // row-major storage
         const char trans = 'T';

         // LAPack is probably not the most efficient for the system sizes of interest ;
         // even simple linpack dgefa and dgesl would probably be better ;
         // but for now, just go with it

         int info=0;
         int ipiv[_nDim] ;
         DGETRF(&_nDim, &_nDim, J, &_nDim, ipiv, &info) ;

         if ( info != 0 ) { SNLS_FAIL(__func__, "info non-zero from dgetrf"); }

         for (int iX = 0; iX < _nDim; ++iX) {
            newton[iX] = - r[iX] ; 
         }

         int nRHS=1; info=0;
         DGETRS(&trans, &_nDim, &nRHS, J, &_nDim, ipiv, newton, &_nDim, &info);

         if ( info != 0 ) { SNLS_FAIL(__func__, "info non-zero from lapack::dgetrs()") ; }

#else
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)

         {
            const int n = _nDim;

            int   err = SNLS_LUP_Solve<n>(J, newton, r);
            if (err<0) {
               SNLS_FAIL(__func__," fail return from LUP_Solve()") ;
            }
            for (int i=0; (i<n); ++i) { newton[i] = -newton[i]; }
         }
#endif
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
      }
      // Update x using a provided delta x
      __snls_hdev__ inline void  update(const double* const delX, const int ielem, const int offset ) {
         const int toffset = ielem + offset;
         //delX is already offset
         for (int iX = 0; iX < _nDim; ++iX) {
            _x(toffset, iX) = _x(toffset, iX) + delX[iX] ;
         }
      }
      // Reject the updated x using a provided delta x
      __snls_hdev__ inline void  reject(const double* const delX, const int ielem, const int offset ) {
         const int toffset = ielem + offset;
         // delX is already offset
         for (int iX = 0; iX < _nDim; ++iX) {
            _x(toffset, iX) = _x(toffset, iX) - delX[iX] ;
         }
      }
      
   public:

      /// Performs a bitwise reduction and operation on _status to see if the
      /// current batch can exit.
      template <const bool batch_loop, const int NUMTHREADS>
      inline bool status_exit(const int offset,
                              const int batch_size)
      {
         bool red_add = false;
         const int end = offset + batch_size;
         SNLSStatus_t*  status = _status.data(Device::GetCHAIES());
         switch(Device::GetBackend()) {
   #ifdef RAJA_ENABLE_CUDA
            case(ExecutionStrategy::CUDA): {
               //RAJA::ReduceBitAnd<RAJA::cuda_reduce, bool> output(init_val);
               RAJA::ReduceSum<RAJA::cuda_reduce, int> output(0);
               RAJA::forall<RAJA::cuda_exec<512>>(RAJA::RangeSegment(offset, end), [=] __snls_device__ (int i) {
                  if(!batch_loop) { 
                     if(isConverged(status[i])) output += 1;
                  }
                  else {
                     if(status[i] != SNLSStatus_t::unConverged) output += 1;
                  }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
   #endif
   #if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
            case(ExecutionStrategy::OPENMP): {
               //RAJA::ReduceBitAnd<RAJA::omp_reduce_ordered, bool> output(init_val);
               RAJA::ReduceSum<RAJA::omp_reduce_ordered, int> output(0);
               RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(offset, end), [=] (int i) {
                  if(!batch_loop) { 
                     if(isConverged(status[i])) output += 1;
                  }
                  else {
                     if(status[i] != SNLSStatus_t::unConverged) output += 1;
                  }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
   #endif
            case(ExecutionStrategy::CPU):
            default: {
               //RAJA::ReduceBitAnd<RAJA::seq_reduce, bool> output(init_val);
               RAJA::ReduceSum<RAJA::seq_reduce, int> output(0);
               RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(offset, end), [=] (int i) {
                  if(!batch_loop) { 
                     if(isConverged(status[i])) output += 1;
                  }
                  else {
                     if(status[i] != SNLSStatus_t::unConverged) output += 1;
                  }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
         } // End of switch
         return red_add;
      } // end of status_exitable

      /// Performs a bitwise reduction and operation on _status to see if the
      /// current batch can exit.
      template <const int NUMTHREADS>
      inline bool reduced_rjSuccess(rview1b &rJSuccess,
                                    const int batch_size)
      {
         bool red_add = false;
         const int end = batch_size;
         switch(Device::GetBackend()) {
#ifdef RAJA_ENABLE_CUDA
            case(ExecutionStrategy::CUDA): {
               //RAJA::ReduceBitAnd<RAJA::cuda_reduce, bool> output(init_val);
               RAJA::ReduceSum<RAJA::cuda_reduce, int> output(0);
               RAJA::forall<RAJA::cuda_exec<512>>(RAJA::RangeSegment(0, end), [=] __snls_device__ (int i) {
                  if (rJSuccess(i)) { output += 1; }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
#endif
#if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
            case(ExecutionStrategy::OPENMP): {
               //RAJA::ReduceBitAnd<RAJA::omp_reduce_ordered, bool> output(init_val);
               RAJA::ReduceSum<RAJA::omp_reduce_ordered, int> output(0);
               RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, end), [=] (int i) {
                  if (rJSuccess(i)) { output += 1; }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
#endif
            case(ExecutionStrategy::CPU):
            default: {
               //RAJA::ReduceBitAnd<RAJA::seq_reduce, bool> output(init_val);
               RAJA::ReduceSum<RAJA::seq_reduce, int> output(0);
               RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, end), [=] (int i) {
                  if (rJSuccess(i)) { output += 1; }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
         } // End of switch
         return red_add;
      } // end of status_exitable

      // Returns a pointer to the status array now on the host.
      // The user is responsible for deleting this ptr after their done using it
      // through the use of the SNLS::memoryManager::dealloc<T>(T* ptr) function  
      SNLSStatus_t * getStatusHost() {
         return _status.data(chai::ExecutionSpace::CPU);
      }
   
#ifdef SNLS_DEBUG
#ifdef __cuda_host_only__
      void  printVecX (const double* const y, std::ostream & oss ) {
         oss << std::setprecision(14) ;
         for ( int iX=0; iX<_nDim; ++iX) {
            oss << y[iX] << " " ;
         }
         oss << std::endl ;
      }

      void  printMatJ (const double* const A, std::ostream & oss ) {
         oss << std::setprecision(14) ;
         for ( int iX=0; iX<_nDim; ++iX) {
            for ( int jX=0; jX<_nDim; ++jX) {
               oss << std::setw(21) << std::setprecision(11) << A[SNLS_NN_INDX(iX,jX,_nDim)] << " " ;
            }
            oss << std::endl ;
         } 
      }
#endif
#endif

   public:
      rview2d _x;

   protected:
      static const int _nXnDim = _nDim * _nDim ;
      
      int _mfevals, _nIters, _nJFact ;
      chai::ManagedArray<int> _fevals;
      rview1d _delta;
      rview1d _res;
      rview2d _residual;
      rview3d _Jacobian;
      rview1b _rjSuccess;
      chai::ManagedArray<bool>   wrkb_data;
      chai::ManagedArray<double> wrk_data;

   private:
      TrDeltaControl* _deltaControl ;

      int    _offset_work;
      int    _maxIter ;
      double _tolerance ;
      int    _outputLevel ;
      uint   _initial_batch_size ;
      uint   _npts ;

      std::ostream* _os ;

      chai::ManagedArray<SNLSStatus_t> _status;
};
  } // namespace batch
} // namespace snls

#endif //SNLS_RAJA_PERF_SUITE
#endif  // SNLS_TRDLDG_BATCH_H
