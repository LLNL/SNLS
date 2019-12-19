// -*-c++-*-

#ifndef SNLS_TRDLDG_H
#define SNLS_TRDLDG_H

#include "SNLS_base.h"
#include "SNLS_lup_solve.h"

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

// row-major storage
#define SNLSTRDLDG_J_INDX(p,q,nDim) (p)*(nDim)+(q)

namespace snls {

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
               
      __snls_hdev__ int     getNDim   () const { return(_nDim   ); };
      __snls_hdev__ int     getNFEvals() const { return(_fevals ); };
      __snls_hdev__ double  getRhoLast() const { return(_rhoLast); };
      __snls_hdev__ double  getDelta  () const { return(_delta  ); };
      __snls_hdev__ double  getRes    () const { return(_res    ); };

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
         
         if ( _deltaControl == nullptr ) {
            SNLS_FAIL("solve", "_deltaControl not set") ;
         }
         
         _status = unConverged ;
         _fevals = 0 ;
         _nJFact = 0 ;
         _nIters = 0 ;
         
         _delta = _deltaControl->getDeltaInit() ;
#ifdef __cuda_host_only__
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
         _res = this->normvec(residual) ;
         double res_0 = _res ;
#ifdef __cuda_host_only__
         if (_os) { *_os << "res = " << _res << std::endl ; }
#endif
         
         bool reject_prev = false ;
         //
         // data for sorting out the step
         double nr2norm = 0.0 ;
         double alpha = 0.0 ;
         double norm_s_sd_opt = 0.0 ;
         double norm_grad = 0.0 ;
         double norm_grad_inv = 0.0 ;
         double qa = 0.0, qb = 0.0 ;
         double Jg2 = 0.0 ;
         double res_cauchy = 0.0 ;
         double nrStep[_nDim] ;
         double grad[_nDim] ;
         //
         while ( _nIters < _maxIter ) {
            //
            _nIters += 1 ;

            if ( !reject_prev ) {
               //
               // have a newly accepted solution point
               // compute information for step determination

               this->computeSysMult( Jacobian, residual, grad, true );
               // used to keep :
               //	ngrad[iX] = -grad[iX] ;
               // 	nsd[iX]   = ngrad[iX] * norm_grad_inv ; 
               
               // find Cauchy point
               //
               {
                  double norm2_grad = this->normvecSq( grad ) ;
                  norm_grad = sqrt( norm2_grad ) ;
                  {
                     double ntemp[_nDim] ; 
                     this->computeSysMult( Jacobian, grad, ntemp, false ) ; // was -grad in previous implementation, but sign does not matter
                     Jg2 = this->normvecSq( ntemp ) ;
                  }
                  
                  alpha = 1e0 ;
                  res_cauchy = res_0 ;
                  if ( Jg2 > 0e0 ) { 
                     alpha = norm2_grad / Jg2 ;
                     res_cauchy = sqrt(fmax(res_0*res_0 - alpha*norm2_grad,0.0)) ;
                  }

                  // optimal step along steapest descent is alpha*ngrad
                  norm_s_sd_opt = alpha*norm_grad ;
                  
                  norm_grad_inv = 1e0 ;
                  if ( norm_grad > 0e0 ) {
                     norm_grad_inv = 1e0 / norm_grad ;
                  }
                  
               }

               this->computeNewtonStep( Jacobian, residual, nrStep ) ;
               nr2norm = this->normvec( nrStep ) ;
               //
               // as currently implemented, J is not factored in-place and computeSysMult no loner works ;
               // or would have to be reworked for PLU=J

               {

                  // inline the use of p instead of computing it
                  //
                  // double p[_nDim] ;
                  // for (int iX = 0; iX < _nDim; ++iX) {
                  //    p[iX] = nrStep[iX] + alpha*grad[iX] ; // nrStep - alpha*ngrad
                  // }

                  {
                     double gam0 = 0e0 ;
                     double norm2_p = 0.0 ;
                     for (int iX = 0; iX < _nDim; ++iX) {
                        double p_iX = nrStep[iX] + alpha*grad[iX] ;
                        norm2_p += p_iX*p_iX ;
                        gam0 += p_iX * grad[iX] ;
                     }
                     gam0 = -gam0 * norm_grad_inv ;

                     qb = 2.0 * gam0 * norm_s_sd_opt ;

                     qa = norm2_p ;
                     
                  }
                     
               }

            } // !reject_prev

            // compute the step given _delta
            //
            {
               double delx[_nDim] ;
               double pred_resid ;
               bool use_nr = false ;
               //
               if ( nr2norm <= _delta ) {
                  // use Newton step
                  use_nr = true ;

                  for (int iX = 0; iX < _nDim; ++iX) {
                     delx[iX] = nrStep[iX] ;
                  }
                  pred_resid = 0e0 ;

#ifdef __cuda_host_only__
                  if ( _os != nullptr ) {
                     *_os << "trying newton step" << std::endl ;
                  }
#endif
               }
               else { // use_nr

                  // step along dogleg path
               
                  if ( norm_s_sd_opt >= _delta ) {

                     // use step along steapest descent direction

                     {
                        double factor = -_delta * norm_grad_inv ;
                        for (int iX = 0; iX < _nDim; ++iX) {
                           delx[iX] = factor * grad[iX] ;
                        }
                     }

                     {
                        double val = -(_delta*norm_grad) + 0.5*_delta*_delta*Jg2 * (norm_grad_inv*norm_grad_inv) ;
                        pred_resid = sqrt(fmax(2.0*val + res_0*res_0,0.0)) ;
                     }
            
#ifdef __cuda_host_only__
                     if ( _os != nullptr ) {
                        *_os << "trying step along first leg" << std::endl ;
                     }
#endif
            
                  }
                  else{

                     // qc and beta depend on delta
                     //
                     double qc = norm_s_sd_opt*norm_s_sd_opt - _delta*_delta ;
                     //
                     double beta = (-qb+sqrt(qb*qb-4.0*qa*qc))/(2.0*qa) ;
#ifdef DEBUG
                     if ( beta > 1.0 || beta < 0.0 ) {
                        SNLS_FAIL(__func__, "beta not in [0,1]") ;
                     }
#endif
                     beta = fmax(0.0,fmin(1.0,beta)) ; // to deal with and roundoff

                     // delx[iX] = alpha*ngrad[iX] + beta*p[iX] = beta*nrStep[iX] - (1.0-beta)*alpha*grad[iX]
                     //
                     {
                        double omb  = 1.0-beta ;
                        double omba = omb*alpha ;
                        for (int iX = 0; iX < _nDim; ++iX) {
                           delx[iX] = beta * nrStep[iX] - omba*grad[iX] ;
                        }
                        pred_resid = omb * res_cauchy ;
                     }

#ifdef __cuda_host_only__
                     if ( _os != nullptr ) {
                        *_os << "trying step along second leg" << std::endl ;
                     }
#endif
                  } // if norm_s_sd_opt >= delta

               } // use_nr
               
               this->update( delx ) ;
               reject_prev = false ;
               //
               {
                  bool rjSuccess = this->computeRJ(residual, Jacobian) ; // at _x
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
                     _res = this->normvec(residual) ;
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
                  this->reject( delx ) ;
               }
            }
            //
            res_0 = _res ;
            
         } // _nIters < _maxIter
         
         return _status ;
      
      }

      // convenience wrapper, for the current _x
      __snls_hdev__ inline bool computeRJ(double* const r,
                                          double* const J ) {
         
         _fevals++ ;
         bool retval = this->_crj.computeRJ(r, J, _x);
         
#ifdef DEBUG
#ifdef __cuda_host_only__
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
               bool retvalThis = this->_crj.computeRJ( r_pert, nullptr, x_pert ) ;
               if ( !retvalThis ) {
                  SNLS_FAIL(__func__, "Problem while finite-differencing");
               }
               for ( int iR = 0; iR < _nDim ; iR++ ) {
                  J_FD[SNLSTRDLDG_J_INDX(iR,iX,_nDim)] = pert_val_inv * ( r_pert[iR] - r_base[iR] ) ;
               }
            }
            
            *_os << "J_an = " << std::endl ; printMatJ( J,    *_os ) ;
            *_os << "J_fd = " << std::endl ; printMatJ( J_FD, *_os ) ;

            // put things back the way they were ;
            retval = this->_crj.computeRJ(r, J, _x);
            
         } // _os != nullptr
#endif
#endif         
         return retval ;
         
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

         // std::copy( r, r + _nDim, newton );
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
            //
            for (int i=0; (i<n); ++i) { newton[i] = -newton[i]; }

         }
#endif
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
      }
      
      __snls_hdev__ inline void  computeSysMult(const double* const J, 
                                                const double* const v,
                                                double* const       p,
                                                bool                transpose ) {
         
         // row-major storage
         bool sysByV = not transpose ;
   
         if ( sysByV ) {
            int ipX = 0 ;
            for (int pX = 0; pX < _nDim; ++pX) {
               p[pX] = 0e0 ;
               for (int kX = 0; kX < _nDim; ++kX) {
                  p[pX] += J[kX+ipX] * v[kX] ;
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
                  p[pX] += J[ikX+pX] * v[kX] ;
               }
               ikX += _nDim ; // ikW = kX*_nDim ;
            }
         }
      }
      
      __snls_hdev__ inline void  update(const double* const delX ) {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x[iX] = _x[iX] + delX[iX] ;
         }
      }
      
      __snls_hdev__ inline void  reject(const double* const delX ) {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x[iX] = _x[iX] - delX[iX] ;
         }
      }
      
      __snls_hdev__ inline double normvec(const double* const v) {
         return sqrt( this->normvecSq(v) ) ;
      }
      
      __snls_hdev__ inline double normvecSq(const double* const v) {
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

   public:
      double _x[_nDim] ;

   protected:
      static const int _nXnDim = _nDim * _nDim ;
      
      int _fevals, _nIters, _nJFact ;
      double _delta, _res ;

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

      SNLSStatus_t  _status ;
};

} // namespace snls

#endif  // SNLS_TRDLDG_H
