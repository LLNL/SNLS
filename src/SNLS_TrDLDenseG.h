// -*-c++-*-

#ifndef SNLS_TRDLDG_H
#define SNLS_TRDLDG_H

#include "SNLS_base.h"
#include "SNLS_linalg.h"
#include "SNLS_lup_solve.h"
#include "SNLS_TrDelta.h"
#include "SNLS_kernels.h"

#include <stdlib.h>
#include <iostream>
#ifdef __snls_host_only__
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

namespace snls {

// trust region type solver, dogleg approximation
// for dense general Jacobian matrix
//
// CRJ should :
// 	have member function
// 		     __snls_hdev__ bool computeRJ( double* const r, double* const J, const double* const x ) ;
// 		computeRJ function returns true for successful evaluation
// 		TODO ... J becomes a RAJA::View ?
//	have trait nDimSys
//
// TODO ...*** specialize to N=1 case, and N=2 also?
//
template< typename CRJ, int nDimSys = CRJ::nDimSys >
class SNLSTrDlDenseG 
{
   public:
      static_assert(has_valid_computeRJ<CRJ>::value || has_valid_computeRJ_lambda<CRJ>::value, "The CRJ implementation in SNLSTrDlDenseG needs to implement bool computeRJ( double* const r, double* const J, const double* const x ) or be a lambda function that takes in the same arguments");
      // static_assert(has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");

   public:
   // constructor
   __snls_hdev__ SNLSTrDlDenseG(CRJ &crj) :
               _crj(crj),
               _fevals(0), _nIters(0), _nJFact(0), _delta(1e8), _res(1e20),
               _deltaControl(nullptr),
               _outputLevel(0),
               _rhoLast(0.0),
               _os(nullptr),
               _status(unConverged)
               {
               }
   // destructor
   __snls_hdev__ ~SNLSTrDlDenseG() {
#ifdef __snls_host_only__
      if ( _outputLevel > 1 && _os != nullptr ) {
         *_os << "Function and Jacobian factorizations: " << _fevals << " " << _nJFact << std::endl;
      }
#endif
   }

   public:
      CRJ &_crj ;
      static constexpr int _nDim = nDimSys;

      __snls_hdev__ int     getNDim   () const { return(_nDim   ); }
      __snls_hdev__ int     getNFEvals() const { return(_fevals ); }
      __snls_hdev__ int     getNJEvals() const { return(_fevals ); }
      __snls_hdev__ double  getRhoLast() const { return(_rhoLast); }
      __snls_hdev__ double  getDelta  () const { return(_delta  ); }
      __snls_hdev__ double  getRes    () const { return(_res    ); }

      // setX can be used to set the initial guess
      __snls_hdev__ inline void setX( const double* const x ) {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x[iX] = x[iX] ;
         }
      }
   
      __snls_hdev__ inline void getX( double* const x ) const {
         for (int iX = 0; iX < _nDim; ++iX) {
            x[iX] = _x[iX] ;
         }
      } 

      /**
       * Must call setupSolver before calling solve
       */
      __snls_hdev__ void   setupSolver(int              maxIter,
                                       double           tolerance,
                                       TrDeltaControl * deltaControl,
                                       int              outputLevel=0 ) {
   
         _status = unConverged ;
         _fevals = 0 ;

         _maxIter = maxIter ;
         _tolerance = tolerance ;

         _deltaControl = deltaControl ;

         this->setOutputlevel( outputLevel ) ;

      }

      __snls_hdev__ void   setOutputlevel( int    outputLevel ) {
         _outputLevel = outputLevel ;
         _os          = nullptr ;
         //
#ifdef __snls_host_only__
         if ( _outputLevel > 0 ) {
            _os = &(std::cout) ;
         }
#endif
      }

      // solve returns status
      //
      // on exit, _res is consistent with _x
      __snls_hdev__ SNLSStatus_t solve() {
         
         if ( _deltaControl == nullptr ) {
            SNLS_FAIL("solve", "_deltaControl not set") ;
         }
         
         _status = unConverged ;
         _fevals = 0 ;
         _nJFact = 0 ;
         _nIters = 0 ;
         
         _delta = _deltaControl->getDeltaInit() ;
#ifdef __snls_host_only__
         if (_os) { *_os << "initial delta = " << _delta << std::endl ; }
#endif

         double residual[_nDim], Jacobian[_nXnDim] ;
         //
         {
            bool rjSuccess = this->computeRJ(residual, Jacobian) ; // at _x
            if ( !(rjSuccess) ) {
               _status = initEvalFailure ;
               return _status ;
            }
         }
         _res = snls::linalg::norm<_nDim>(residual);
         double res_0 = _res ;
#ifdef __snls_host_only__
         if (_os) { *_os << "res = " << _res << std::endl ; }
#endif
         
         bool reject_prev = false ;
         //
         // data for sorting out the step
         double nrStep[_nDim];
         double grad[_nDim];
         double delx[_nDim];
         double Jg_2;

         //
         while ( _nIters < _maxIter ) {
            //
            _nIters += 1 ;

            // This is done outside this step so that these operations can be done with varying solve
            // techniques such as LU/QR or etc...
            if(!reject_prev) {
               // So the LU solve does things in-place which causes issues when calculating the grad term...
               // So, we need to pull this out and perform this operation first
               snls::linalg::matTVecMult<_nDim, _nDim>(Jacobian, residual, grad);
               {
                  double ntemp[_nDim] ;
                  snls::linalg::matVecMult<_nDim, _nDim>(Jacobian, grad, ntemp);
                  Jg_2 = snls::linalg::dotProd<_nDim>(ntemp, ntemp);
               }
               const bool sol_stat = this->computeNewtonStep( Jacobian, residual, nrStep );
               if (!sol_stat) {
                  _status = SNLSStatus_t::linearSolveFailure;
                  break;
               }

            }
            //
            double pred_resid;
            bool use_nr = false;

            // If the step was rejected nrStep will be the same value as previously, and so we can just recalculate nr_norm here.
            const double nr_norm = snls::linalg::norm<_nDim>(nrStep);

            // computes the updated delta x, predicated residual error, and whether or not NR method was used.
            snls::dogleg<_nDim>(_delta, res_0, nr_norm, Jg_2, grad, nrStep,
                                delx, _x, pred_resid, use_nr, _os);
            reject_prev = false;

            //
            {
               bool rjSuccess = this->computeRJ(residual, Jacobian) ; // at _x
               snls::updateDelta<_nDim>(_deltaControl, residual, res_0, pred_resid, nr_norm, _tolerance, use_nr, rjSuccess,
                                        _delta, _res, _rhoLast, reject_prev, _status, _os);
               if(_status != SNLSStatus_t::unConverged) { break; }
            }

            if ( reject_prev ) {
#ifdef __snls_host_only__
               if ( _os != nullptr ) {
                  *_os << "rejecting solution" << std::endl ;
               }
#endif
               _res = res_0 ;
               this->reject( delx ) ;
            }
            //
            res_0 = _res;
            
         } // _nIters < _maxIter
         
         return _status ;
      
      }

      // convenience wrapper, for the current _x
      __snls_hdev__ inline bool computeRJ(double* const r,
                                          double* const J ) {
         
         _fevals++ ;
         bool retval;
         if constexpr(has_valid_computeRJ<CRJ>::value) {
            retval = this->_crj.computeRJ(r, J, _x);
         } else {
            retval = this->_crj(r, J, _x);
         }
         
#ifdef SNLS_DEBUG
#ifdef __snls_host_only__
         if ( _outputLevel > 2 && _os != nullptr ) {
            // do finite differencing
            // assume system is scaled such that perturbation size can be standard

            double r_base[_nDim]; 
            for ( int jX = 0; jX < _nDim ; ++jX ) {
               r_base[jX] = r[jX] ;
            }
            
            const double pert_val     = 1.0e-7 ;
            const double pert_val_inv = 1.0/pert_val ;
            
            double J_FD[_nXnDim] ;
            
            for ( int iX = 0; iX < _nDim ; ++iX ) {
               double r_pert[_nDim];
               double x_pert[_nDim];
               for ( int jX = 0; jX < _nDim ; ++jX ) {
                  x_pert[jX] = _x[jX] ;
               }
               x_pert[iX] = x_pert[iX] + pert_val ;
               bool retvalThis;
               if constexpr(has_valid_computeRJ<CRJ>::value) {
                  retvalThis = this->_crj.computeRJ(r_pert, nullptr, x_pert);
               } else {
                  retvalThis = this->_crj(r_pert, nullptr, x_pert);
               }
               if ( !retvalThis ) {
                  SNLS_FAIL(__func__, "Problem while finite-differencing");
               }
               for ( int iR = 0; iR < _nDim ; iR++ ) {
                  J_FD[SNLS_NN_INDX(iR,iX,_nDim)] = pert_val_inv * ( r_pert[iR] - r_base[iR] ) ;
               }
            }
            
            *_os << "J_an = " << std::endl ; snls::linalg::printMat<_nDim>( J,    *_os ) ;
            *_os << "J_fd = " << std::endl ; snls::linalg::printMat<_nDim>( J_FD, *_os ) ;

            // put things back the way they were ;
            if constexpr(has_valid_computeRJ<CRJ>::value) {
               retval = this->_crj.computeRJ(r, J, _x);
            } else {
               retval = this->_crj(r, J, _x);
            }
            
         } // _os != nullptr
#endif
#endif         
         return retval ;
         
      }
      
   private :
   
      __snls_hdev__ inline bool  computeNewtonStep (double* const       J,
                                                    const double* const r,
                                                    double* const       newton  ) {
         
         _nJFact++ ;
         
#if HAVE_LAPACK && SNLS_USE_LAPACK && defined(__snls_host_only__)
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

         if ( info != 0 ) {
            SNLS_WARN(__func__, "info non-zero from dgetrf");
            return false;
         }

         // std::copy( r, r + _nDim, newton );
         for (int iX = 0; iX < _nDim; ++iX) {
            newton[iX] = - r[iX] ; 
         }

         int nRHS=1; info=0;
         DGETRS(&trans, &_nDim, &nRHS, J, &_nDim, ipiv, newton, &_nDim, &info);

         if ( info != 0 ) {
            SNLS_WARN(__func__, "info non-zero from lapack::dgetrs()");
            return false;
         }

#else
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__snls_host_only__)

         {
            const int n = _nDim;

            int   err = SNLS_LUP_Solve<n>(J, newton, r);
            if (err<0) {
               SNLS_WARN(__func__," fail return from LUP_Solve()");
               return false;
            }
            //
            for (int i=0; (i<n); ++i) { newton[i] = -newton[i]; }

         }
#endif
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
         return true;
      }
      
      __snls_hdev__ inline void  reject(const double* const delX ) {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x[iX] = _x[iX] - delX[iX] ;
         }
      }

   public:
      double _x[_nDim] ;

   protected:
      static constexpr int _nXnDim = _nDim * _nDim ;
      
      int _fevals, _nIters, _nJFact ;
      double _delta, _res ;

   private:
      TrDeltaControl* _deltaControl ;

      int   _maxIter    ;
      double _tolerance  ;
      int   _outputLevel;

      // _rhoLast is not really needed -- but is kept for debug and testing purposes
      double _rhoLast ;

#ifdef __snls_host_only__
      std::ostream* _os ;
#else
      char* _os ; // do not use
#endif

      SNLSStatus_t  _status ;
};

} // namespace snls

#endif  // SNLS_TRDLDG_H
