// -*-c++-*-

#ifndef SNLS_NEWTONBB_H
#define SNLS_NEWTONBB_H

#include "SNLS_base.h"
#include "SNLS_cuda_portability.h"
#include "SNLS_port.h"

#include <stdlib.h>
#include <iostream>
#ifdef __cuda_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

//////////////////////////////////////////////////////////////////////

namespace snls {

/** Helper templates to ensure compliant CFJ implementations */
template<typename CFJ, typename = void>
struct has_valid_computeFJ : std::false_type { static constexpr bool value = false;};

template<typename CFJ>
struct has_valid_computeFJ <
   CFJ,typename std::enable_if<
       std::is_same<
           decltype(std::declval<CFJ>().computeFJ(std::declval<double&>(), std::declval<double&>(),std::declval<double>())),
           bool  
       >::value
       ,
       void
   >::type
>: std::true_type { static constexpr bool value = true;};

// 1D Newton solver, with bounding/bisection checks
//
template< class CFJ, bool unbounded >
class NewtonBB
{
public:
   static_assert(has_valid_computeFJ<CFJ>::value, "The CFJ implementation in SNLSNewtonBB needs to implement bool computeFJ( double &f, double &J, double x )");
   
   // constructor
   __snls_hdev__ NewtonBB( CFJ       &cfj,
                           double    tol=1e-8,
                           int       maxIter=40,
                           double    mulTolX=1e-4
                           ) :
      _cfj(cfj),
      _tol(tol),
      _maxIter(maxIter),
      _fevals(0)
      {
         _tolx = _tol * mulTolX ;
      } ;
   
   // destructor
   __snls_hdev__ ~NewtonBB() {} ;

   __snls_hdev__
   inline
   int
   getNFEvals() const { return(_fevals); };
   
   // bound for function for which x does not have limits ;
   //
   // xl and xh need to be set as inputs, fl and fh do not
   //
   // on exit of true, fl and fh are consistent with xl and xh
   //
   __snls_hdev__
   inline
   bool
   doBoundA( double &xl,
             double &xh,
             double &fl,
             double &fh ) {
      
      if ( xl > xh ) {
         double tempvar = xl ;
         xl = xh ;
         xh = tempvar ;
      }
      
      double del = 0.2*fmax(xh-xl,fmax(fabs(xl), 1.0)) ;

      bool success ;

      double dxhi, dxli ;
      {
         double Jh ;
         success = this->_cfj.computeFJ(fh, Jh, xh) ; _fevals++ ; 
         if ( !success ) {
            return false ;
         }
         double Jl ;
         success = this->_cfj.computeFJ(fl, Jl, xl) ; _fevals++ ; 
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

         if ( iBracket < 10 && dxhi > 0.0 ) {
            xhPrev = xh ; xh = xh + fmin( del, 1.2 / dxhi) ; newH = true ;
         }
         else if ( iBracket < 10 && dxli < 0.0 ) {
            xlPrev = xl ; xl = xl + fmax(-del, 1.2 / dxli) ; newH = false ;
         }
         else {
            // take turns
            if ( newH ) { 
               xlPrev = xl ; xl = xl - del ; newH = false ;
            }
            else {
               xhPrev = xh ; xh = xh + del ; newH = true ;
            }
         }
         //
         del = del * 1.8;
         
         if ( newH ) {
            double J ;
            double fhPrev = fh ;
            success = this->_cfj.computeFJ(fh, J, xh) ; _fevals++ ; 
            if ( !success ) {
               // could concieve of a fallback
               return false ;
            }
            if ( fabs(fh) < _tol ) {
               return true ;
            }
            if ( fhPrev * fh < 0.0 ) { 
               // bracketed
               xl = xhPrev ;
               fl = fhPrev ; 
               return true ;
            }
            dxhi = -J / fh;
         }
         else {
            double J ;
            double flPrev = fl ;
            success = this->_cfj.computeFJ(fl, J, xl) ; _fevals++ ; 
            if ( !success ) {
               // could concieve of a fallback
               return false ;
            }
            if ( fabs(fl) < _tol ) {
               return true ;
            }
            if ( flPrev * fl < 0.0 ) {
               // bracketed
               xh = xlPrev ;
               fh = flPrev ;
               return true ;
            }
            dxli = -J / fl;
         }

      }

      return false ;
      
   } ;

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
      bool success = this->_cfj.computeFJ(fun, J, x) ; _fevals++ ;
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
         success = this->_cfj.computeFJ(fl, tmpJ, xl) ; _fevals++ ;
         if ( !success ) {
            status = initEvalFailure ;
            return status ;
         }
         if ( fabs(fl) < _tol ) {
            x = xl ;
            status = converged ;
            return status ;
         }
         success = this->_cfj.computeFJ(fh, tmpJ, xh) ; _fevals++ ;
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
   
      if ( fl*fh > 0 ) {
         // root is not bracketed

         if ( unbounded ) {
            bool success = doBoundA(xl, xh, fl, fh) ;
            if ( !success ) {
               status = algFailure ;
               return status ;
            }
         }
         else {
            status = algFailure ;
            return status ;
         }
      }

      // make xl point at which fl=f(xl) < 0
      //
      if ( fl > 0.0 ) {
         double tempvar ;
         
         tempvar = fl ;
         fl = fh ;
         fh = tempvar ;

         tempvar = xl ;
         xl = xh ;
         xh = tempvar ;
      }
   
      double dxold = fabs( xh - xl ) ;
      double dx = dxold ;

      if ( fun < 0.0 && fun > fl ) {
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
         }
         else {
            dxold = dx;
            dx = fun / J;
            x = x - dx;
         }
  
         if ( fabs(dx) < _tolx  &&  j>10 ) {
            status = converged ;
            return status ;
         }
         if ( fabs(x) > _tolx ) {
            if ( fabs(dx) / fabs(x) < _tol ) {
               status = converged ;
               return status ;
            }
         }

         success = this->_cfj.computeFJ(fun, J, x) ; _fevals++ ;
         if ( !success ) {
            status = evalFailure ;
            return status ;
         }
  
         if ( fabs(fun) < _tol ) {
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
private:
   double  _tol, _tolx ;
   int     _maxIter ;
   int     _fevals ;
   
}; // class NewtonBB

} // namespace snls

#endif  // SNLS_NEWTONBB_H
