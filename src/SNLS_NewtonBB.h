// -*-c++-*-

#ifndef SNLS_NEWTONBB_H
#define SNLS_NEWTONBB_H

#include "SNLS_base.h"
#include "SNLS_kernels.h"

#include <stdlib.h>
#include <iostream>
#ifdef __snls_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

//////////////////////////////////////////////////////////////////////

namespace snls {

const double mulTolXDefault = 1e-4 ;

// 1D Newton solver, with bounding/bisection checks
//
template< typename CFJ >
class NewtonBB
{
public:
   // Note this is not perfect and might fail as seen in the notes for these helper functions
   // However, the compiler can help to identify some of these issues
   static_assert(has_valid_computeFJ<CFJ>::value || has_valid_computeFJ_lamb<CFJ>::value, "The CFJ implementation in SNLSNewtonBB needs to implement bool computeFJ( double &f, double &J, double x ) or a lambda function with same types");
   
   // constructor
   __snls_hdev__ NewtonBB( CFJ &cfj, bool unbounded) :
      _cfj(cfj),
      _unbounded(unbounded),
      _boundStepGrowthFactor(1.2),
      _boundOvershootFactor(1.2),
      _tol(1e-8),
      _maxIter(40),
      _fevals(0),
      _outputLevel(0),
      _os(nullptr)
      {
         _tolx = _tol * mulTolXDefault ;
      }
   
   // destructor
   __snls_hdev__ ~NewtonBB() {
#ifdef __snls_host_only__
      if ( _outputLevel > 1 && _os != nullptr ) {
         *_os << "NewtonBB Function evaluations : " << _fevals << std::endl;
      }
#endif
   }

   /**
    * This call is optional 
    */
   __snls_hdev__
   void
   setupSolver(int       maxIter,
               double    tolerance,
               double    mulTolX=mulTolXDefault,
               int       outputLevel=0 ) {
      
      _maxIter = maxIter ;
      _tol     = tolerance ;
      _tolx    = _tol * mulTolX;
      this->setOutputlevel( outputLevel ) ;
      
   }
   
   __snls_hdev__
   inline
   int
   getNFEvals() const { return(_fevals); }

   __snls_hdev__
   inline
   bool
   computeFJ(double& fh, double& Jh, double xh) {
      if constexpr(has_valid_computeFJ<CFJ>::value) {
         return _cfj.computeFJ(fh, Jh, xh);
      }
      else {
         return _cfj(fh, Jh, xh);
      }
   }
   
   /** find bounds for zero of a function for which x does not have limits ;
    *
    * xl and xh need to be set as inputs, fl and fh do not
    *
    * on exit value of true, fl and fh are consistent with xl and xh
    */
   __snls_hdev__
   inline
   bool
   doBoundA( double &xl,
             double &xh,
             double &fl,
             double &fh ) {
      
      if ( xl > xh ) {
         snls_swap(xl, xh);
      }
      
      double delH = 0.2*fmax(xh-xl,fmax(fabs(xl), 1.0)) ;
      double delL = delH ;

      bool success ;

      double dxhi, dxli ;
      {
         double Jh ;
         success = computeFJ(fh, Jh, xh) ; _fevals++ ; 
         if ( !success ) {
            return false ;
         }
         double Jl ;
         success = computeFJ(fl, Jl, xl) ; _fevals++ ; 
         if ( !success ) {
            return false ;
         }
         if ( fabs(fh) < _tol || fabs(fl) < _tol || fl*fh < 0.0 ) {
            return true ;
         }
         dxhi = -Jh / fh;
         dxli = -Jl / fl;
      }

      bool newH = false ;
      double xhPrev = xh ;
      double xlPrev = xl ;
      //
      for ( int iBracket = 0; iBracket < _maxIter; ++iBracket ){

         // the ordering here biases the search toward exploring smaller x values
         //
         if ( iBracket < 10 && dxli < 0.0 ) {
            xlPrev = xl ; xl += fmax(-delL, _boundOvershootFactor / dxli) ; newH = false ;
         }
         else if ( iBracket < 10 && dxhi > 0.0 ) {
            xhPrev = xh ; xh += fmin( delH, _boundOvershootFactor / dxhi) ; newH = true ;
         }
         else {
            // take turns
            if ( newH ) { 
               xlPrev = xl ; xl -= delL ; newH = false ;
            }
            else {
               xhPrev = xh ; xh += delH ; newH = true ;
            }
         }
         // Create a lambda function to deal with the bound updates portion of things as
         // updating the lower bounds was exactly the same as doing it for the upper bounds code
         // wise...
         auto bnd_fnc = [this, &success]
         (double &func, double &x_val, double &x_prev, double &delta, double &delta_x, double &x_other, double &func_other) {

            double J;
            double func_prev = func ;
            success = computeFJ(func, J, x_val) ; this->_fevals++ ;
#ifdef __snls_host_only__
            if (this->_os) { *this->_os << "NewtonBB in bounding, have x, f, J : "
                            << x_val << " " <<  func << " "  << J << std::endl ; }
#endif
            if ( !success ) {
               // try a smaller step
               x_val = x_prev ; func = func_prev ; // delta_x is as was before
               delta *= 0.1;
#ifdef __snls_host_only__
               if (this->_os) { *this->_os << "NewtonBB trouble in bounding, cut delta back to = " << delta << std::endl ; }
#endif
               return false;
            }
            if ( fabs(func) < this->_tol ) {
               return true ;
            }
            if ( func_prev * func < 0.0 ) {
               // bracketed
               x_other = x_prev ;
               func_other = func_prev ;
               return true ;
            }
            delta_x = -J / func;
            delta *= this->_boundStepGrowthFactor;
            return false;
         };

         if ( newH ) {
            const bool ret_val = bnd_fnc(fh, xh, xhPrev, delH, dxhi, xl, fl);
            if (ret_val) return true;
            if (!success) continue;
         }
         else {
            const bool ret_val = bnd_fnc(fl, xl, xlPrev, delL, dxli, xh, fh);
            if (ret_val) return true;
            if (!success) continue;
         }
      
      } // iBracket

      return false ;
      
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
      
   // x is an initial guess, must be in [xl,xh]
   //
   __snls_hdev__
   inline
   SNLSStatus_t 
   solve( double      &x,
          double       xl,
          double       xh
          ) {
      
      SNLSStatus_t status = unset ;
   
      double fun, J ;
      bool success = computeFJ(fun, J, x) ; _fevals++ ;
      if ( !success ) {
         status = initEvalFailure ;
         return status ;
      }
      //
      if ( fabs(fun) < _tol ) {
         status = converged ;
         return status ;
      }
      
      double fl, fh ;
      {
         double tmpJ;
         success = computeFJ(fl, tmpJ, xl) ; _fevals++ ;
         if ( !success ) {
            status = initEvalFailure ;
            return status ;
         }
         if ( fabs(fl) < _tol ) {
            x = xl ;
            status = converged ;
            return status ;
         }
         success = computeFJ(fh, tmpJ, xh) ; _fevals++ ;
         if ( !success ) {
            status = initEvalFailure ;
            return status ;
         }
         if ( fabs(fh) < _tol ) {
            x = xh ; 
            status = converged ;
            return status ;
         }
      }
   
      if ( fl * fh > 0 ) {
         // root is not bracketed

         if ( _unbounded ) {
            bool success = doBoundA(xl, xh, fl, fh) ;
            if ( !success ) {
               status = bracketFailure ;
               return status ;
            }
         }
         else {
            status = bracketFailure ;
            return status ;
         }
      }

      // make xl point at which fl=f(xl) < 0
      //
      if ( fl > 0.0 ) {
         snls_swap(fl, fh);
         snls_swap(xl, xh);
      }
   
      double dxold = fabs( xh - xl ) ;
      double dx = dxold ;

      if ( fun < 0.0 && fun > fl) {
         xl = x ;
      }
      else if ( fun > 0.0 && fun < fh ) {
         xh = x ;
      }

      //     main loop over a given number of iterations. checks whether
      //     extrapolated value using the gradient for newton iteration 
      //     is beyond the bounds and either uses the Newton estimate of
      //     bisection depending on outcome. convergence is checked on 
      //     the value of the function, closeness of variable to either
      //     limit and change of variable over an iteration.
      //
      for ( int j = 0; j < _maxIter; ++j ) {
      
         if ( ((x - xh) * J - fun) * ((x - xl) * J - fun) >= 0.0
              || fabs(2.0 * fun) > fabs(dxold * J) ) {
            dxold = dx;
            dx = ( xh - xl) / 2.0;
            x = xl + dx;
#ifdef __snls_host_only__
            if (_os) { *_os << "NewtonBB doing bisection with dx : " << fabs(dx) << std::endl ; }
#endif
         }
         else {
            dxold = dx;
            dx = -fun / J;
#ifdef __snls_host_only__
            if (_os) { *_os << "NewtonBB trying newton step with dx : " << dx << std::endl ; }
#endif
            x += dx;
         }
  
         if ( fabs(dx) < _tolx  &&  j>10 ) {
            //
            // could additionally check (fabs(x) > _tolx) && (fabs(dx) / fabs(x) < _tol) for convergence
            // but that may in some cases be too sloppy
            //
#ifdef __snls_host_only__
            if ( _os != nullptr ) {
               *_os << "converged by bracketing" << std::endl ;
            }
#endif
            status = convByBracket ;
            return status ;
         }

         success = computeFJ(fun, J, x) ; _fevals++ ;
#ifdef __snls_host_only__
         if (_os) { *_os << "NewtonBB evaluation with f, J, x : " 
                         << fun << " " << J << " " << x << std::endl ; }
#endif
         if ( !success ) {
            status = evalFailure ;
            return status ;
         }
  
         if ( fabs(fun) < _tol ) {
#ifdef __snls_host_only__
            if ( _os != nullptr ) {
               *_os << "converged" << std::endl ;
            }
#endif
            status = converged ;
            return status ;
         }
  
         if ( fun < 0.0 ) {
            xl = x;
         }
         else {
            xh = x;
         }

      } // iteration, j

      status = unConverged ;
      return status ;
   }

public:
   CFJ     &_cfj ;
   bool    _unbounded;
   double  _boundStepGrowthFactor ;
   double  _boundOvershootFactor ;
   
private:
   double  _tol, _tolx ;
   int     _maxIter ;
   int     _fevals ;
   
   int   _outputLevel;
#ifdef __snls_host_only__
   std::ostream* _os ;
#else
   char* _os ; // do not use
#endif

}; // class NewtonBB

} // namespace snls

#endif  // SNLS_NEWTONBB_H
