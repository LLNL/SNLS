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

#if HAVE_LAPACK && SNLS_USE_LAPACK
#if SNLS_HAVE_MSLIB
#include "MS_Matmodel.h"
// MS fortran wrappers for LAPack solvers
#include "MS_FortranWrappers.h"
#else

#define SNLS_USE_LAPACK 0

extern "C" {
   int DGETRF(const int* m, const int* n, double* A, const int* lda, int* ipiv, int* info);
   int DGETRS(const char* trans, const int* n, const int* nrhs, const double* const A, const int* lda,
              const int* const ipiv, double* b, const int* ldb, int* info);
}
#endif

#ifdef _WIN32
#include <mathimf.h>
#else
#include <math.h>
#endif

#ifndef M_PI
#define M_PI       3.14159265358979323846264338327950288
#endif

#ifndef M_SQRT2    
#define M_SQRT2    1.41421356237309504880168872420969808
#endif

typedef double real8 ;

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
   convFailure        =  1
} SNLSStatus_t ;

class TrDeltaControl
{
public:
   __snls_hdev__  TrDeltaControl();

   __snls_hdev__ real8 getDeltaInit() const { return _deltaInit;} ;


   __snls_hdev__ void  incrDelta   (void *strm, real8 &delta) const;
   __snls_hdev__ bool  decrDelta   (void *strm, real8 &delta, real8 normfull, bool took_full) const;

   __snls_hdev__ bool  updateDelta (void   *strm       ,
                                    real8  &delta      ,
                                    real8   res        ,
                                    real8   res_0      ,
                                    real8   pred_resid ,
                                    bool   &reject_prev,
                                    bool    took_full  ,
                                    real8   normfull   ,
                                    real8  &rho         ) const ;

private:
   __snls_hdev__ void checkParams() const ;
public:
   real8 _xiLG, _xiUG, _xiIncDelta, _xiLO, _xiUO, _xiDecDelta, _xiForcedIncDelta, _deltaInit, _deltaMin, _deltaMax;
   bool  _rejectResIncrease ;
};

/** Helper templates to ensure compliant CRJ implementations */
template<typename CRJ, typename = void>
struct has_valid_computeRJ : std::false_type { static constexpr bool value = false;};

template<typename CRJ>
struct has_valid_computeRJ <
   CRJ,typename std::enable_if<
       std::is_same<
           decltype(std::declval<CRJ>().computeRJ(std::declval<real8* const>(), std::declval<real8* const>(),std::declval<const real8 *>())),
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
// 		     __snls_hdev__ bool computeRJ( real8* const r, real8* const J, const real8* const x ) ;
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
      static_assert(has_valid_computeRJ<CRJ>::value, "The CRJ implementation in SNLSTrDlDenseG needs to implement bool computeRJ( real8* const r, real8* const J, const real8* const x )");
      static_assert(has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");
      static const int nxMult   = 10; // TODO ... may eventually be able to reduce this
      static const int nxXxMult =  3; // TODO ... may eventually be able to reduce this -- maybe do not need scratch space if factor Jacobian in-place, but may then need to guarantee that have doen and matrix-vector multiply first (computeSysMult for getting _ngrad -- maybe meaning that get rid of computeSysMult (or making it non-public) so that people do not think that we have stored a J that is good for multiplies)
      static const int niMult   =  1;

   public:
   // constructor
   __snls_hdev__ SNLSTrDlDenseG(CRJ &crj) :
               _crj(crj),
               _fevals(0), _nIters(0), _nJFact(0),
               _r(nullptr), _x(nullptr), _J(nullptr), 
               _deltaControl(nullptr),
               _outputLevel(0),
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
               
      __snls_hdev__ int    getNDim   () const { return(_nDim  ); };
      __snls_hdev__ int    getNFEvals() const { return(_fevals); };
      __snls_hdev__ real8* getXPntr  () const { return(_x     ); };
      __snls_hdev__ real8* getRPntr  () const { return(_r     ); };
      __snls_hdev__ real8* getJPntr  () const { return(_J     ); };

      __snls_hdev__ void   setupSolver(int    maxIter,
                                       real8  tolerance,
                                       TrDeltaControl* deltaControl,
                                       int    outputLevel=0 ) {
   
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
         if ( _outputLevel > 0 ) {
#ifdef __cuda_host_only__
            _os = &(std::cout) ;
#endif
         }
      }
      
      // solve returns status
      __snls_hdev__ SNLSStatus_t solve() {
   
         _status = unConverged ;
         _fevals = 0 ;
         _nJFact = 0 ; // ... cnt%fj
         _nIters = 0 ; // ... cnt%ni

         bool use_nr = false ;
         bool reject_prev = false ;
         // bool converged = false ; ...
         bool have_ngrad = false, have_nr = false, have_trdl_dirs = false, have_p_a_b = false ; // ...have = .false.

         real8 delta = _deltaControl->getDeltaInit() ;

         {
            bool rjSuccess = this->computeRJ() ; // _r, _J, _x
            if ( !(rjSuccess) ) {
               _status = initEvalFailure ;
               return _status ;
            }
         }
         this->set0() ;
         real8 res = this->normvec(_r) ;
         real8 res_0 = res ;
#ifdef __cuda_host_only__
         if (_os) { *_os << "res = " << res << std::endl ; }
#endif

         reject_prev = false ;

         real8 alpha, Jg2 ;
         real8 nr2norm ;
         real8 norm_grad ;
         real8 norm_grad_inv ;
         real8 qa, qb ;
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
      
            if ( nr2norm <= delta ) {
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

               real8 norm2_grad = this->normvecSq( _ngrad ) ;
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

            real8 pred_resid ;
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
               real8 norm_s_sd_opt = alpha*norm_grad ;

               if ( norm_s_sd_opt >= delta ) {

                  // use step along steapest descent direction

                  for (int iX = 0; iX < _nDim; ++iX) {
                     _delx[iX] = _nsd[iX] * delta ;
                  }
            
                  real8 val = -(delta*norm_grad) + 0.5*delta*delta*Jg2 * (norm_grad_inv*norm_grad_inv) ;
                  pred_resid = 2.0*val + res_0*res_0 ;
                  real8 signFact = ( pred_resid < 0 ? -1e0 : 1e0 ) ;
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
   
                     real8 gam0 = 0e0 ;
                     for (int iX = 0; iX < _nDim; ++iX) {
                        gam0 += _p[iX] * _nsd[iX] ;
                     }
   
                     // get n_perp to find gam1
                     for (int iX = 0; iX < _nDim; ++iX) {
                        _ntemp[iX] = _p[iX] - gam0 * _nsd[iX] ;
                     }
                     //
                     real8 normTemp = this->normvec( _ntemp ) ;
                     if ( normTemp > 0e0 ) {
                        real8 normTempInv = 1.0 / normTemp ;
                        for (int iX = 0; iX < _nDim; ++iX) {
                           _ntemp[iX] = _ntemp[iX] * normTempInv ;
                        }
                     }
   
                     real8 gam1 = 0e0 ;
                     for (int iX = 0; iX < _nDim; ++iX) {
                        gam1 += _p[iX] * _ntemp[iX] ;
                     }
   
                     qa = gam0*gam0 + gam1*gam1 ;
                     qb = 2.0 * gam0 * norm_s_sd_opt ;
               
                     have_p_a_b = true ;

                  } // not have_p_a_b

                  // qc and beta depend on delta
                  //
                  real8 qc = norm_s_sd_opt*norm_s_sd_opt - delta*delta ;
                  //
                  real8 beta = (-qb+sqrt(qb*qb-4.0*qa*qc))/(2.0*qa) ;

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
                  bool deltaSuccess = _deltaControl->decrDelta(_os, delta, nr2norm, use_nr ) ;
                  if ( ! deltaSuccess ) {
                     _status = deltaFailure ;
                     break ; // while ( _nIters < _maxIter ) 
                  }
                  reject_prev = true ;
               }
               else {
                  res = this->normvec(_r) ;
#ifdef __cuda_host_only__
                  if ( _os != nullptr ) {
                     *_os << "res = " << res << std::endl ;
                  }
#endif

                  // allow to exit now, may have forced one iteration anyway, in which
                  // case the delta update can do funny things if the residual was
                  // already very small 

                  if ( res < _tolerance ) {
#ifdef __cuda_host_only__
                     if ( _os != nullptr ) {
                        *_os << "converged" << std::endl ;
                     }
#endif
                     _status = converged ;
                     break ; // while ( _nIters < _maxIter ) 
                  }

                  {
                     real8 rho ;
                     bool deltaSuccess = _deltaControl->updateDelta(_os,
                                                                    delta, res, res_0, pred_resid,
                                                                    reject_prev, use_nr, nr2norm, rho) ;
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
               res = res_0 ;
               this->reject() ;
               for (int iX = 0; iX < _nDim; ++iX) {
                  _r[iX] = _rScratch[iX] ;
               }
            }
            else {
               this->set0() ;
               have_ngrad = false; have_nr = false; have_trdl_dirs = false; have_p_a_b = false ; // ...have = .false.
               // always have current residual and Jacobian
            }

            res_0 = res ;
    
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

            real8 r_base[_nDim]; 
            for ( int jX = 0; jX < _nDim ; ++jX ) {
               r_base[jX] = _r[jX] ;
            }
            
            const real8 pert_val     = 1.0e-7 ;
            const real8 pert_val_inv = 1.0/pert_val ;
            
            real8 J_FD[_nXnDim] ;
            
            for ( int iX = 0; iX < _nDim ; ++iX ) {
               real8 r_pert[_nDim];
               real8 x_pert[_nDim];
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
      
      __snls_hdev__ void  computeNewtonStep (real8* const newton  ) {
         
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
      
      __snls_hdev__ void  computeSysMult    (const real8* const v, real8* const p, bool transpose ) {
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
      
      __snls_hdev__ void  update            (const real8* const delX ) {
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
      
      __snls_hdev__ real8 normvec           (const real8* const v) {
         return sqrt( this->normvecSq(v) ) ;
      }
      
      __snls_hdev__ real8 normvecSq         (const real8* const v) {
         real8 a = 0e0;
         for (int iX = 0; iX < _nDim; ++iX) {
            a += v[iX]*v[iX] ;
         }
         return a ;
      }
      
#ifdef DEBUG
#ifdef __cuda_host_only__
      __snls_hdev__ void  printVecX         (const real8* const y, std::ostream & oss ) {
         oss << std::setprecision(14) ;
         for ( int iX=0; iX<_nDim; ++iX) {
            oss << y[iX] << " " ;
         }
         oss << std::endl ;
      }
      
      __snls_hdev__ void  printMatJ         (const real8* const A, std::ostream & oss ) {
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
      real8 *_r, *_x, *_J ;

   private:
      TrDeltaControl* _deltaControl ;

      int   _maxIter    ;
      real8 _tolerance  ;
      int   _outputLevel;

#ifdef __cuda_host_only__
      std::ostream* _os ;
#else
      char* _os ; // do not use
#endif

      real8 _nxStorage[nxMult*_nDim];
      real8 _nxXxStorage[nxXxMult*_nDim*_nDim];
      int   _niStorage[niMult*_nDim];
      //
      real8 *_x0, *_nr, *_delx, *_ngrad, *_nsd, *_ntemp, *_p, *_rScratch ;
      real8 *_J0, *_JScratch ;
      int   *_ipiv ;

      SNLSStatus_t  _status ;
};

} // namespace snls

#endif  // SNLS_TRDLDG_H
