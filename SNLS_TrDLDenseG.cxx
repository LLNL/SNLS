#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

#include "SNLS_cuda_portability.h"

#include "SNLS_lup_solve.h"
#include "SNLS_TrDLDenseG.h"

#define SNLS_USE_LAPACK 0

namespace snls {

__snls_hdev__
bool SNLSTrDlDenseG::computeRJ()
{
   bool retval = this->computeRJ(_r, _J, _x);
   return retval ;
}

__snls_hdev__
SNLSTrDlDenseG::SNLSStatus_t SNLSTrDlDenseG::solve()
{
//#ifdef __cuda_host_only__
//   const char* me = "SNLSTrDlDenseG::solve" ;
//#endif   
   
   _status = SNLSTrDlDenseG::unConverged ;
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
         _status = SNLSTrDlDenseG::initEvalFailure ;
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
        if ( _os != NULL ) {
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
            if ( _os != NULL ) {
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
            if ( _os != NULL ) {
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
               _status = SNLSTrDlDenseG::deltaFailure ;
               break ; // while ( _nIters < _maxIter ) 
            }
            reject_prev = true ;
         }
         else {
            res = this->normvec(_r) ;
#ifdef __cuda_host_only__
            if ( _os != NULL ) {
               *_os << "res = " << res << std::endl ;
            }
#endif

            // allow to exit now, may have forced one iteration anyway, in which
            // case the delta update can do funny things if the residual was
            // already very small 

            if ( res < _tolerance ) {
#ifdef __cuda_host_only__
               if ( _os != NULL ) {
                  *_os << "converged" << std::endl ;
               }
#endif
               _status = SNLSTrDlDenseG::converged ;
               break ; // while ( _nIters < _maxIter ) 
            }

            {
               real8 rho ;
               bool deltaSuccess = _deltaControl->updateDelta(_os,
                                                              delta, res, res_0, pred_resid,
                                                              reject_prev, use_nr, nr2norm, rho) ;
               if ( ! deltaSuccess ) {
                  _status = SNLSTrDlDenseG::deltaFailure ;
                  break ; // while ( _nIters < _maxIter ) 
               }
            }

         }
      }

      if ( reject_prev ) { 

#ifdef __cuda_host_only__
         if ( _os != NULL ) {
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
      
   // ...***
   // ~/Projects/nonlinear_solvers/Trilinos/trilinos-11.2.3-Source/packages/nox/src/NOX_Solver_TrustRegionBased.C
   // ~/mdef/build/matlib/matEvpc/tr_dogleg_nJ.F90
   // ~/mdef/build/matlib/matEvpc/trust_util_module.f90
   // ~/mdef/build/matlib/matEvpc/evptn.F90
   // ~/mdef/build/matlib/matEvpc/evp_solve_l.f90

}

// constructor
__snls_hdev__
TrDeltaControl::TrDeltaControl() :
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

__snls_hdev__
void TrDeltaControl::checkParams() const
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

__snls_hdev__   
bool TrDeltaControl::decrDelta(void *strm, real8 &delta, real8 normfull, bool took_full) const
{
   bool success = true ;
   
   if ( took_full ) 
   {
      real8 tempa = delta    * _xiDecDelta ;
      real8 tempb = normfull * _xiDecDelta ;
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
void TrDeltaControl::incrDelta(void* strm, real8  &delta) const
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

__snls_hdev__
bool TrDeltaControl::updateDelta
(
   void   *strm      ,
   real8  &delta     ,
   real8   res       ,
   real8   res_0     ,
   real8   pred_resid,
   bool   &reject    , // reject_prev
   bool    took_full , // use_nr
   real8   normfull  , // nr2norm
   real8  &rho       ) const
{
   bool  success       = true;
   real8 actual_change = res - res_0;
   real8 pred_change   = pred_resid - res_0;

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
         //    real8 temp = delta;
         //    real8 temp_b = normfull * _xiIncDelta;
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

__snls_hdev__
void SNLSTrDlDenseG::setupSolver
(
   int             nDim        ,
   real8* const    nxStorage   ,
   real8* const    nxXxStorage ,
   int*   const    niStorage   , 
   int             maxIter     ,
   real8           tolerance   ,
   TrDeltaControl* deltaControl,
   int             outputLevel 
)
{
   
   _nDim   =  nDim ;
   _nXnDim = _nDim*_nDim ;
   
   _nxStorage   = nxStorage ;
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
   
   _nxXxStorage = nxXxStorage ;
   _J           = &(_nxXxStorage[0]) ;
   _J0          = &(_nxXxStorage[_nXnDim]) ;
   _JScratch    = &(_nxXxStorage[_nXnDim*2]) ;
   
   _niStorage   = niStorage ;
   _ipiv        = &(_niStorage[0]) ;
   
   _status = SNLSTrDlDenseG::unConverged ;
   _fevals = 0 ;

   _maxIter = maxIter ;
   _tolerance = tolerance ;

   _deltaControl = deltaControl ;

   this->setOutputlevel( outputLevel ) ;

}

__snls_hdev__
void SNLSTrDlDenseG::setOutputlevel( int    outputLevel )
{

   _outputLevel = outputLevel ;
   _os          = NULL ;
   //
   if ( _outputLevel > 0 ) 
   {
#ifdef __cuda_host_only__
      _os = &(std::cout) ;
#endif
   }
   
}


#if HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)

// computeNewtonStep()
//
// This version of the Newton solver uses the LAPACK solver DGETRF() and DGETRS()
// 
// Note that we can replace this with a custom function if there are performance 
// specializations (say for a known fixed system size)
//-------------------------------------------------------------------------------------------

__snls_hdev__
void SNLSTrDlDenseG::computeNewtonStep( real8* const newton ) // ... sys_solve
{
   _nJFact++ ;
   
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
}

#else
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
   
// computeNewtonStep()
//
// This is a alternate version that will work when LAPACK is not available.
// It uses an crappy Numerical Recipes solver and was originally intended for testing 
// the GPU-enabled code base..
//
// Note - this method has been updated to deal with both row and column-major matrices.
//-------------------------------------------------------------------------------------------

__snls_hdev__
void SNLSTrDlDenseG::computeNewtonStep( real8* const newton ) // ... sys_solve
{
   _nJFact++ ;
   
   int n = _nDim;

   // create a local, row-major copy that can be modified by the LUP solver...

#if SNLSTRDLDG_J_COLUMN_MAJOR
   for (int i=0,k=0; (i<n); ++i )
   for (int j=0,m=i; (j<n); ++j, ++k, m+=n) { _JScratch[m] = _J[k]; }
#else
   for (int i=0; (i<(n*n)); ++i ) { _JScratch[i] = _J[i]; }
#endif

   real8 tol = 1.0e-50;                                      // tol = (really small value for LUP pivoting)
   int   err = SNLS_LUP_Solve(_JScratch,newton,_r,n,tol);      // J = LUP_Solve(J,newton,r)
                                                             //
   for (int i=0; (i<n); ++i) { newton[i] = -newton[i]; }     // newton = -newton

   if (err<0) {
      SNLS_FAIL(__func__," fail return from LUP_Solve()") ;
   }
}
#endif
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)

// can replace this with a custom function if there are performance specializations (say for a known fixed system size)
//
// compute 
// 	p = J . v,
// or (if transpose)
// 	p = J^T . v
//
__snls_hdev__
void SNLSTrDlDenseG::computeSysMult(const real8* const v, // ... sys_mult
                                           real8* const p,
                                           bool transpose )
{
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

__snls_hdev__
void SNLSTrDlDenseG::update(const real8* const delX)
{
   for (int iX = 0; iX < _nDim; ++iX) {
      _x[iX] = _x0[iX] + delX[iX] ;
   }
}

__snls_hdev__
void SNLSTrDlDenseG::reject() 
{
   for (int iX = 0; iX < _nDim; ++iX) {
      _x[iX] = _x0[iX] ;
   }
   for (int iiJ = 0; iiJ < _nXnDim; ++iiJ) {
      _J[iiJ] = _J0[iiJ] ;
   }
}

__snls_hdev__
void SNLSTrDlDenseG::set0() 
{
   for (int iX = 0; iX < _nDim; ++iX) {
      _x0[iX] = _x[iX] ;
   }
   for (int iiJ = 0; iiJ < _nXnDim; ++iiJ) {
      _J0[iiJ] = _J[iiJ] ;
   }
}

__snls_hdev__
real8 SNLSTrDlDenseG::normvecSq(const real8* const v)
{
   real8 a = 0e0;
   for (int iX = 0; iX < _nDim; ++iX) {
       a += v[iX]*v[iX] ;
   }
   return a ;
}

__snls_hdev__
real8 SNLSTrDlDenseG::normvec(const real8* const v)
{
   return sqrt( this->normvecSq(v) ) ;
}

__snls_hdev__
void
SNLSTrDlDenseG::printVecX( const real8* const y )
{
#ifdef __cuda_host_only__
   std::cout << std::setprecision(14) ;
   for ( int iX=0; iX<_nDim; ++iX) std::cout << y[iX] << " " ; std::cout << std::endl ;
#endif
}

__snls_hdev__
void
SNLSTrDlDenseG::printMatJ( const real8* const A )
{
#ifdef __cuda_host_only__
   std::cout << std::setprecision(14) ;
   for ( int iX=0; iX<_nDim; ++iX) {
      for ( int jX=0; jX<_nDim; ++jX) {
         std::cout << A[SNLSTRDLDG_J_INDX(iX,jX)] << " " ;
      }
      std::cout << std::endl ;
   } 
#endif
}

} // namespace snls
