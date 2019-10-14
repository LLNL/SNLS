// -*-c++-*-

#ifndef SNLS_TRDLDG_H
#define SNLS_TRDLDG_H

#include "SNLS_cuda_portability.h"
#include "SNLS_lup_solve.h"
#include "SNLS_port.h"

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
#if SNLS_HAVE_MSLIB
#include "MS_Matmodel.h"
// MS fortran wrappers for LAPack solvers
#include "MS_FortranWrappers.h"
#else

extern "C" {
   int DGETRF(const int* m, const int* n, double* A, const int* lda, int* ipiv, int* info);
   int DGETRS(const char* trans, const int* n, const int* nrhs, const double* const A, const int* lda,
              const int* const ipiv, double* b, const int* ldb, int* info);
}
#endif
#endif

//////////////////////////////////////////////////////////////////////

#if SNLS_HAVE_MSLIB
#include "MS_math.h"
#else
#if defined(_WIN32) && __INTEL_COMPILER
#include <mathimf.h>
#else
#include <math.h>
#endif
#endif

#if SNLS_HAVE_MSLIB
#include "MS_Log.h"
#ifdef __cuda_host_only__
#define SNLS_FAIL(loc,str) MS_Fail(loc,str);
#else
#define SNLS_FAIL(loc,str) MS_Fail(loc,str);
#endif
#else
#ifdef __cuda_host_only__
#include <stdio.h>
#include <exception>
#include <stdexcept>
#define SNLS_FAIL(loc,str) throw std::runtime_error(std::string("at ") + std::string(loc) + std::string(" failure : ") + std::string(str)) ;
#else
#define SNLS_FAIL(loc,str) printf("ERROR : SNLS failure in %s : %s\n",loc,str) ;
#endif
#endif

//////////////////////////////////////////////////////////////////////

#define SNLSTRDLDG_J_COLUMN_MAJOR 0

#if SNLSTRDLDG_J_COLUMN_MAJOR
// column-major storage
#define SNLSTRDLDG_J_INDX(p,q,nDim) (p)+(q)*(nDim)
#else
// row-major storage
#define SNLSTRDLDG_J_INDX(p,q,nDim) (p)*(nDim)+(q)
#endif

namespace snls {

typedef enum {
   converged          =  0,
   initEvalFailure    = -2,
   unConverged        = -10,
   deltaFailure       = -20,
   algFailure         = -100,
   unset              = -200,
   convFailure        =  1
} SNLSStatus_t ;

class TrDeltaControl
{
public:
   
   __snls_hdev__ TrDeltaControl() :
      _xiLG(0.75),
      _xiUG(1.4),
      _xiIncDelta(1.5),
      _xiLO(0.35),
      _xiUO(5.0),
      _xiDecDelta(0.25),
      _xiForcedIncDelta(1.2),
      _deltaInit(1.0),
      _deltaMin(1e-12),
      _deltaMax(1e4),
      _rejectResIncrease(true)
   {
      this->checkParams() ;
   }

   __snls_hdev__ double getDeltaInit() const { return _deltaInit;} ;

   __snls_hdev__   
   bool decrDelta(void *strm, double &delta, double normfull, bool took_full) const
   {
      bool success = true ;
      
      if ( took_full ) 
      {
         double tempa = delta    * _xiDecDelta ;
         double tempb = normfull * _xiDecDelta ;
         delta = sqrt( tempa*tempb ) ;
      }
      else 
         delta = delta * _xiDecDelta ;

      if ( delta < _deltaMin ) 
      {
         delta = _deltaMin ;

   #ifdef __cuda_host_only__
         if (strm) { *((std::ostream *) strm) << "delta now at min " << delta << std::endl ; }
   #endif

         success = false ;
      }

   #ifdef __cuda_host_only__
      else 
         if (strm) { *((std::ostream *) strm) << "decr delta to " << delta << std::endl ; }
   #endif

      return success;
   }

   __snls_hdev__
   void incrDelta(void* strm, double  &delta) const
   {
      delta = delta * _xiIncDelta;

      if ( delta > _deltaMax ) 
      {
         delta = _deltaMax ;

   #ifdef __cuda_host_only__
         if (strm) { *((std::ostream *) strm) << "delta now at max " << delta << std::endl ; }
   #endif
      }
   #ifdef __cuda_host_only__
      else 
         if (strm) { *((std::ostream *) strm) << "incr delta to "    << delta << std::endl ; }
   #endif
   }

   
   __snls_hdev__ bool updateDelta
   (
      void    * strm      ,
      double  & delta     ,
      double    res       ,
      double    res_0     ,
      double    pred_resid,
      bool    & reject    , // reject_prev
      bool      took_full , // use_nr
      double    normfull  , // nr2norm
      double  & rho       ) const
   {
      bool  success       = true;
      double actual_change = res - res_0;
      double pred_change   = pred_resid - res_0;

      if ( pred_change == 0e0 ) 
      {
         if ( delta >= _deltaMax ) {
            // things are going badly enough that the solver should probably stop
   #ifdef __cuda_host_only__
            if (strm) { *((std::ostream *) strm) << "predicted change is zero and delta at max" << std::endl; }
   #endif
            success = false ;
         }
         else {
   #ifdef __cuda_host_only__
            if (strm) { *((std::ostream *) strm) << "predicted change is zero, forcing delta larger" << std::endl; }
   #endif
            delta = fmin( delta * _xiForcedIncDelta, _deltaMax );
            // ierr = IERR_UPD_DELTA_PRZ_p; // do not report this as a failure
         }
      }
      else 
      {
         rho = actual_change / pred_change;
   #ifdef __cuda_host_only__
         if (strm) { *((std::ostream *) strm) << "rho = " << rho << std::endl; }
   #endif
         if ( ( rho > _xiLG ) &&
            ( actual_change < 0e0 ) &&
            ( rho < _xiUG ) ) {
            if ( ! took_full ) {
               // increase delta
               this->incrDelta(strm, delta);
            }
            // instead of truncating, just do not increase, as 
            // truncating can cause trouble
            // else { 
            // 
            //    double temp = delta;
            //    double temp_b = normfull * _xiIncDelta;
            //    if (temp_b < delta) {
            //       delta = temp_b;
   #ifdef __cuda_host_only__
            //       if ( _os != NULL ) {
            //          _os << "took full step, delta truncated from " << temp << " to " << delta << std::endl;
            //       }
   #endif
            //    }
            // }
         }
         else if ( ( rho < _xiLO ) ||
                  ( rho > _xiUO ) ) 
         { 
            // decrease delta
            success = this->decrDelta(strm, delta, normfull, took_full);
         }
      } // pred_change == 0
      
      reject = false;

      // do not make this >= , may have res and res_0 both zero and that is ok
      if ( ( actual_change > 0e0 ) && 
         ( _rejectResIncrease ) ) 
      { 
         // residual increasing, need to reject solution
         // have already decreased delta (rho < 0)
         reject = true;
      }
      
      return success;
   }


private:
   
   __snls_hdev__ void checkParams() const
   {
      if ( ( _deltaMin <= 0 ) ||
         ( _deltaMax <= _deltaMin ) ||
         ( _xiLG <= _xiLO ) ||
         ( _xiUG >= _xiUO ) ||
         ( _xiIncDelta <= 1 ) ||
         ( ( _xiDecDelta >= 1 ) || ( _xiDecDelta <= 0 ) ) ||
         ( _xiForcedIncDelta <= 1 ) 
         ) {
         SNLS_FAIL("TrDeltaControl::checkParams", "bad setting") ;
      }
   }

public:
   double _xiLG, _xiUG, _xiIncDelta, _xiLO, _xiUO, _xiDecDelta, _xiForcedIncDelta, _deltaInit, _deltaMin, _deltaMax;
   bool  _rejectResIncrease ;
};

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
// TODO ...*** specialize to N=1 case, nad N=2 also?
//
template< class CRJ >
class SNLSTrDlDenseG 
{
   public:
      static_assert(has_valid_computeRJ<CRJ>::value, "The CRJ implementation in SNLSTrDlDenseG needs to implement bool computeRJ( double* const r, double* const J, const double* const x )");
      static_assert(has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");
      static const int nxMult   = 10; // TODO ... may eventually be able to reduce this
      static const int nxXxMult =  3; // TODO ... may eventually be able to reduce this -- maybe do not need scratch space if factor Jacobian in-place, but may then need to guarantee that have doen and matrix-vector multiply first (computeSysMult for getting _ngrad -- maybe meaning that get rid of computeSysMult (or making it non-public) so that people do not think that we have stored a J that is good for multiplies)
      static const int niMult   =  1;

   public:
   // constructor
   __snls_hdev__ SNLSTrDlDenseG(CRJ &crj) :
               _crj(crj),
               _fevals(0), _nIters(0), _nJFact(0), _delta(1e8), _res(1e20),
               _r(nullptr), _x(nullptr), _J(nullptr), 
               _deltaControl(nullptr),
               _outputLevel(0),
               _rhoLast(0.0),
               _os(nullptr),
               _x0(nullptr), _nr(nullptr), _delx(nullptr), _ngrad(nullptr), _nsd(nullptr), _ntemp(nullptr), _p(nullptr), _rScratch(nullptr),
               _J0(nullptr), _JScratch(nullptr),
               _ipiv(nullptr),
               _status(unConverged)
               {
               };
   // destructor
   __snls_hdev__ ~SNLSTrDlDenseG() {
#ifdef __cuda_host_only__
      if ( _outputLevel > 1 && _os != nullptr ) {
         *_os << "Function and Jacobian factorizations: " << _fevals << " " << _nJFact << std::endl;
      }
#endif
   };

   public:
      CRJ &_crj ;
      static const int _nDim = CRJ::nDimSys ;
               
      __snls_hdev__ int     getNDim   () const { return(_nDim  ); };
      __snls_hdev__ int     getNFEvals() const { return(_fevals); };
      __snls_hdev__ double* getXPntr  () const { return(_x     ); };
      __snls_hdev__ double* getRPntr  () const { return(_r     ); };
      __snls_hdev__ double* getJPntr  () const { return(_J     ); };
      __snls_hdev__ double  getRhoLast() const { return(_rhoLast); };
      __snls_hdev__ double  getDelta  () const { return(_delta ); };
      __snls_hdev__ double  getRes    () const { return(_res   ); };

      // setX can be used to set the initial guess
      __snls_hdev__ void setX( const double* const x ) {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x[iX] = x[iX] ;
         }
      };
   
      __snls_hdev__ void getX( double* const x ) const {
         for (int iX = 0; iX < _nDim; ++iX) {
            x[iX] = _x[iX] ;
         }
      };   

   
      __snls_hdev__ void   setupSolver(int              maxIter,
                                       double           tolerance,
                                       TrDeltaControl * deltaControl,
                                       int              outputLevel=0 ) {
   
         _x           = &(_nxStorage[0]) ;
         _x0          = &(_nxStorage[_nDim]) ;
         _r           = &(_nxStorage[_nDim*2]) ;
         _nr          = &(_nxStorage[_nDim*3]) ;
         _delx        = &(_nxStorage[_nDim*4]) ;
         _ngrad       = &(_nxStorage[_nDim*5]) ;
         _nsd         = &(_nxStorage[_nDim*6]) ;
         _ntemp       = &(_nxStorage[_nDim*7]) ;
         _p           = &(_nxStorage[_nDim*8]) ;
         _rScratch    = &(_nxStorage[_nDim*9]) ;
         // nxMult
   
         _J           = &(_nxXxStorage[0]) ;
         _J0          = &(_nxXxStorage[_nXnDim]) ;
         _JScratch    = &(_nxXxStorage[_nXnDim*2]) ;
         // nxXxMult
   
         _ipiv        = &(_niStorage[0]) ;
         // niMult
   
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
#ifdef __cuda_host_only__
         if ( _outputLevel > 0 ) {
            _os = &(std::cout) ;
         }
#endif
      }
      
      // solve returns status
      //
      // on exit, _res is consistent with _x
      __snls_hdev__ SNLSStatus_t solve() {
   
         _status = unConverged ;
         _fevals = 0 ;
         _nJFact = 0 ;
         _nIters = 0 ;

         bool use_nr = false ;
         bool reject_prev = false ;
         bool have_ngrad = false, have_nr = false, have_trdl_dirs = false, have_p_a_b = false ;

         _delta = _deltaControl->getDeltaInit() ;
#ifdef __cuda_host_only__
         if (_os) { *_os << "initial delta = " << _delta << std::endl ; }
#endif

         {
            bool rjSuccess = this->computeRJ() ; // _r, _J, _x
            if ( !(rjSuccess) ) {
               _status = initEvalFailure ;
               return _status ;
            }
         }
         this->set0() ;
         _res = this->normvec(_r) ;
         double res_0 = _res ;
#ifdef __cuda_host_only__
         if (_os) { *_os << "res = " << _res << std::endl ; }
#endif

         reject_prev = false ;

         double alpha, Jg2 ;
         double nr2norm ;
         double norm_grad ;
         double norm_grad_inv ;
         double qa, qb ;
         //
         while ( _nIters < _maxIter ) {
            //
            _nIters += 1 ;

            use_nr = false ;

            if ( ! have_nr ) {
               // get newton step
               this->computeNewtonStep( _nr ) ;
               nr2norm = this->normvec( _nr ) ;
               have_nr = true ;
            }
      
            if ( nr2norm <= _delta ) {
               // okay to use whole newton step
               // 2-norm(NR step) <= delta
               use_nr = true ;
            }

            if ( (! use_nr) && (! have_trdl_dirs) ) {

               // need to get other directions
               //
               // always have current residual and Jacobian

               if ( ! have_ngrad ) {
                  // the (negative) gradient is J'*r
                  // steapest descent direction is -J'*r
                  this->computeSysMult( _r, _ngrad, true );
                  for (int iX = 0; iX < _nDim; ++iX) {
                     _ngrad[iX] = -_ngrad[iX] ;
                  }
                  have_ngrad = true ;
               }
          
               // find Cauchy point

               double norm2_grad = this->normvecSq( _ngrad ) ;
               norm_grad = sqrt( norm2_grad ) ;
               this->computeSysMult( _ngrad, _ntemp, false ) ;
               Jg2 = this->normvecSq( _ntemp ) ;
               if ( Jg2 > 0e0 ) { 
                  alpha = norm2_grad / Jg2 ;
               }
               else {
                  alpha = 1e0 ;
               }

               // find n_sd
               norm_grad_inv = 1e0 ;
               if ( norm_grad > 0e0 ) {
                  norm_grad_inv = 1e0 / norm_grad ;
               }
               for (int iX = 0; iX < _nDim; ++iX) {
                  _nsd[iX] = _ngrad[iX] * norm_grad_inv ; 
               }

               have_trdl_dirs = true ;

            }

            double pred_resid ;
            //
            if ( use_nr ) {

               for (int iX = 0; iX < _nDim; ++iX) {
                  _delx[iX] = _nr[iX] ;
               }
               pred_resid = 0e0 ;

#ifdef __cuda_host_only__
               if ( _os != nullptr ) {
                  *_os << "trying newton step" << std::endl ;
               }
#endif

            }
            else {

               // step along dogleg path

               // optimal step along steapest descent is alpha*ngrad
               double norm_s_sd_opt = alpha*norm_grad ;

               if ( norm_s_sd_opt >= _delta ) {

                  // use step along steapest descent direction

                  for (int iX = 0; iX < _nDim; ++iX) {
                     _delx[iX] = _nsd[iX] * _delta ;
                  }
            
                  double val = -(_delta*norm_grad) + 0.5*_delta*_delta*Jg2 * (norm_grad_inv*norm_grad_inv) ;
                  pred_resid = 2.0*val + res_0*res_0 ;
                  double signFact = ( pred_resid < 0 ? -1e0 : 1e0 ) ;
                  pred_resid = signFact * sqrt(fabs(pred_resid)) ;
            
#ifdef __cuda_host_only__
                  if ( _os != nullptr ) {
                     *_os << "trying step along first leg" << std::endl ;
                  }
#endif
            
               }
               else{

                  if ( ! have_p_a_b ) {
               
                     for (int iX = 0; iX < _nDim; ++iX) {
                        _p[iX] = _nr[iX] - alpha*_ngrad[iX] ;
                     }
   
                     double gam0 = 0e0 ;
                     for (int iX = 0; iX < _nDim; ++iX) {
                        gam0 += _p[iX] * _nsd[iX] ;
                     }
   
                     // get n_perp to find gam1
                     for (int iX = 0; iX < _nDim; ++iX) {
                        _ntemp[iX] = _p[iX] - gam0 * _nsd[iX] ;
                     }
                     //
                     double normTemp = this->normvec( _ntemp ) ;
                     if ( normTemp > 0e0 ) {
                        double normTempInv = 1.0 / normTemp ;
                        for (int iX = 0; iX < _nDim; ++iX) {
                           _ntemp[iX] = _ntemp[iX] * normTempInv ;
                        }
                     }
   
                     double gam1 = 0e0 ;
                     for (int iX = 0; iX < _nDim; ++iX) {
                        gam1 += _p[iX] * _ntemp[iX] ;
                     }
   
                     qa = gam0*gam0 + gam1*gam1 ;
                     qb = 2.0 * gam0 * norm_s_sd_opt ;
               
                     have_p_a_b = true ;

                  } // not have_p_a_b

                  // qc and beta depend on delta
                  //
                  double qc = norm_s_sd_opt*norm_s_sd_opt - _delta*_delta ;
                  //
                  double beta = (-qb+sqrt(qb*qb-4.0*qa*qc))/(2.0*qa) ;

                  for (int iX = 0; iX < _nDim; ++iX) {
                     _delx[iX] = alpha*_ngrad[iX] + beta*_p[iX] ;
                  }
                  //
                  this->computeSysMult( _delx, _ntemp, false );
                  //
                  for (int iX = 0; iX < _nDim; ++iX) {
                     _ntemp[iX] = _r[iX] + _ntemp[iX] ;
                  }
                  pred_resid = this->normvec( _ntemp ) ;

#ifdef __cuda_host_only__
                  if ( _os != nullptr ) {
                     *_os << "trying step along second leg" << std::endl ;
                  }
#endif
               } // if norm_s_sd_opt >= delta

            } // if use_nr

            this->update( _delx ) ;
            reject_prev = false ;

            for (int iX = 0; iX < _nDim; ++iX) {
               _rScratch[iX] = _r[iX] ;
            }
            //
            {
               bool rjSuccess = this->computeRJ(); // _r, _J, _x
               if ( !(rjSuccess) ) {
                  // got an error doing the evaluation
                  // try to cut back step size and go again
                  bool deltaSuccess = _deltaControl->decrDelta(_os, _delta, nr2norm, use_nr ) ;
                  if ( ! deltaSuccess ) {
                     _status = deltaFailure ;
                     break ; // while ( _nIters < _maxIter ) 
                  }
                  reject_prev = true ;
               }
               else {
                  _res = this->normvec(_r) ;
#ifdef __cuda_host_only__
                  if ( _os != nullptr ) {
                     *_os << "res = " << _res << std::endl ;
                  }
#endif

                  // allow to exit now, may have forced one iteration anyway, in which
                  // case the delta update can do funny things if the residual was
                  // already very small 

                  if ( _res < _tolerance ) {
#ifdef __cuda_host_only__
                     if ( _os != nullptr ) {
                        *_os << "converged" << std::endl ;
                     }
#endif
                     _status = converged ;
                     break ; // while ( _nIters < _maxIter ) 
                  }

                  {
                     bool deltaSuccess = _deltaControl->updateDelta(_os,
                                                                    _delta, _res, res_0, pred_resid,
                                                                    reject_prev, use_nr, nr2norm, _rhoLast) ;
                     if ( ! deltaSuccess ) {
                        _status = deltaFailure ;
                        break ; // while ( _nIters < _maxIter ) 
                     }
                  }

               }
            }

            if ( reject_prev ) { 

#ifdef __cuda_host_only__
               if ( _os != nullptr ) {
                  *_os << "rejecting solution" << std::endl ;
               }
#endif
               _res = res_0 ;
               this->reject() ;
               for (int iX = 0; iX < _nDim; ++iX) {
                  _r[iX] = _rScratch[iX] ;
               }
            }
            else {
               this->set0() ;
               have_ngrad = false; have_nr = false; have_trdl_dirs = false; have_p_a_b = false ;
               // always have current residual and Jacobian
            }

            res_0 = _res ;
    
         } // _nIters < _maxIter

         return _status ;
      
      }

      // convenience wrapper
      __snls_hdev__ bool  computeRJ() {
         
         _fevals++ ;
         bool retval = this->_crj.computeRJ(_r, _J, _x);
         
#ifdef DEBUG
#ifdef __cuda_host_only__
         if ( _outputLevel > 2 && _os != nullptr ) {
            // do finite differencing
            // assume system is scaled such that perturbation size can be standard

            double r_base[_nDim]; 
            for ( int jX = 0; jX < _nDim ; ++jX ) {
               r_base[jX] = _r[jX] ;
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
               bool retvalThis = this->_crj.computeRJ( r_pert, nullptr, x_pert ) ;
               if ( !retvalThis ) {
                  SNLS_FAIL(__func__, "Problem while finite-differencing");
               }
               for ( int iR = 0; iR < _nDim ; iR++ ) {
                  J_FD[SNLSTRDLDG_J_INDX(iR,iX,_nDim)] = pert_val_inv * ( r_pert[iR] - r_base[iR] ) ;
               }
            }
            
            *_os << "J_an = " << std::endl ; printMatJ( _J,   *_os ) ;
            *_os << "J_fd = " << std::endl ; printMatJ( J_FD, *_os ) ;

            // put things back the way they were ;
            retval = this->_crj.computeRJ(_r, _J, _x);
            
         } // _os != nullptr
#endif
#endif         
         return retval ;
         
      }
      
      __snls_hdev__ void  computeNewtonStep (double* const newton  ) {
         
         _nJFact++ ;
         
#if HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
// This version of the Newton solver uses the LAPACK solver DGETRF() and DGETRS()
// 
// Note that we can replace this with a custom function if there are performance 
// specializations (say for a known fixed system size)
   
#if SNLSTRDLDG_J_COLUMN_MAJOR
         const char trans = 'N';
#else
         const char trans = 'T';
#endif

         // might be able to avoid this copy and the use of _JScratch eventually,
         // but for now do the factorization in scratch space
         //
         for (int iiJ = 0; iiJ < _nXnDim; ++iiJ) {
            _JScratch[iiJ] = _J[iiJ] ;
         }

         // LAPack is probably not the most efficient for the system sizes of interest ;
         // even simple linpack dgefa and dgesl would probably be better ;
         // but for now, just go with it

         int info=0;
         DGETRF(&_nDim, &_nDim, _JScratch, &_nDim, _ipiv, &info) ;

         if ( info != 0 ) { SNLS_FAIL(__func__, "info non-zero from dgetrf"); }

         // std::copy( _r, _r + _nDim, newton );

         for (int iX = 0; iX < _nDim; ++iX) {
            newton[iX] = - _r[iX] ; 
         }

         int nRHS=1; info=0;
         DGETRS(&trans, &_nDim, &nRHS, _JScratch, &_nDim, _ipiv, newton, &_nDim, &info);

         if ( info != 0 ) { SNLS_FAIL(__func__, "info non-zero from lapack::dgetrs()") ; }

#else
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)

// This is a alternate version that will work when LAPACK is not available.
// Solver was originally intended for testing the GPU-enabled code base..
//
// Note - this method has been updated to deal with both row and column-major matrices.

         {
            const int n = _nDim;

            // create a local, row-major copy that can be modified by the LUP solver...

#if SNLSTRDLDG_J_COLUMN_MAJOR
            for (int i=0,k=0; (i<n); ++i )
               for (int j=0,m=i; (j<n); ++j, ++k, m+=n) { _JScratch[m] = _J[k]; }
#else
            for (int i=0; (i<(n*n)); ++i ) { _JScratch[i] = _J[i]; }
#endif

            int   err = SNLS_LUP_Solve<n>(_JScratch,newton,_r);      // J = LUP_Solve(J,newton,r)
            //
            for (int i=0; (i<n); ++i) { newton[i] = -newton[i]; }     // newton = -newton

            if (err<0) {
               SNLS_FAIL(__func__," fail return from LUP_Solve()") ;
            }
         }
#endif
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
      }
      
      __snls_hdev__ void  computeSysMult    (const double* const v, double* const p, bool transpose ) {
#if SNLSTRDLDG_J_COLUMN_MAJOR
         bool sysByV =     transpose ;
#else
         bool sysByV = not transpose ;
#endif
   
         if ( sysByV ) {
            int ipX = 0 ;
            for (int pX = 0; pX < _nDim; ++pX) {
               p[pX] = 0e0 ;
               for (int kX = 0; kX < _nDim; ++kX) {
                  p[pX] += _J[kX+ipX] * v[kX] ;
               }
               ipX += _nDim ; // ipX = pX*_nDim ;
            }
         }
         else {
            for (int pX = 0; pX < _nDim; ++pX) {
               p[pX] = 0e0 ;
            }
            int ikX = 0 ;
            for (int kX = 0; kX < _nDim; ++kX) {
               for (int pX = 0; pX < _nDim; ++pX) {
                  p[pX] += _J[ikX+pX] * v[kX] ;
               }
               ikX += _nDim ; // ikW = kX*_nDim ;
            }
         }
      }
      
      __snls_hdev__ void  update            (const double* const delX ) {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x[iX] = _x0[iX] + delX[iX] ;
         }
      }
      
      __snls_hdev__ void  reject            () {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x[iX] = _x0[iX] ;
         }
         for (int iiJ = 0; iiJ < _nXnDim; ++iiJ) {
            _J[iiJ] = _J0[iiJ] ;
         }
      }
      
      __snls_hdev__ void  set0              () {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x0[iX] = _x[iX] ;
         }
         for (int iiJ = 0; iiJ < _nXnDim; ++iiJ) {
            _J0[iiJ] = _J[iiJ] ;
         }
      }
      
      __snls_hdev__ double normvec           (const double* const v) {
         return sqrt( this->normvecSq(v) ) ;
      }
      
      __snls_hdev__ double normvecSq         (const double* const v) {
         double a = 0e0;
         for (int iX = 0; iX < _nDim; ++iX) {
            a += v[iX]*v[iX] ;
         }
         return a ;
      }
      
#ifdef DEBUG
#ifdef __cuda_host_only__
      __snls_hdev__ void  printVecX         (const double* const y, std::ostream & oss ) {
         oss << std::setprecision(14) ;
         for ( int iX=0; iX<_nDim; ++iX) {
            oss << y[iX] << " " ;
         }
         oss << std::endl ;
      }
      
      __snls_hdev__ void  printMatJ         (const double* const A, std::ostream & oss ) {
         oss << std::setprecision(14) ;
         for ( int iX=0; iX<_nDim; ++iX) {
            for ( int jX=0; jX<_nDim; ++jX) {
               oss << std::setw(21) << std::setprecision(11) << A[SNLSTRDLDG_J_INDX(iX,jX,_nDim)] << " " ;
            }
            oss << std::endl ;
         } 
      }
#endif
#endif

   protected:
      static const int _nXnDim = _nDim * _nDim ;
      
      int _fevals, _nIters, _nJFact ;
      double _delta, _res ;
      double *_r, *_x, *_J ;

   private:
      TrDeltaControl* _deltaControl ;

      int   _maxIter    ;
      double _tolerance  ;
      int   _outputLevel;

      // _rhoLast is not really needed -- but is kept for debug and testing purposes
      double _rhoLast ;

#ifdef __cuda_host_only__
      std::ostream* _os ;
#else
      char* _os ; // do not use
#endif

      double _nxStorage[nxMult*_nDim];
      double _nxXxStorage[nxXxMult*_nDim*_nDim];
      int   _niStorage[niMult*_nDim];
      //
      double *_x0, *_nr, *_delx, *_ngrad, *_nsd, *_ntemp, *_p, *_rScratch ;
      double *_J0, *_JScratch ;
      int   *_ipiv ;

      SNLSStatus_t  _status ;
};

} // namespace snls

#endif  // SNLS_TRDLDG_H
