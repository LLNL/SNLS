#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

using namespace std;

#include "../src/SNLS_TrDLDenseG_Batch.h"
#include "../src/SNLS_device_forall.h"
#include "../src/SNLS_memory_manager.h"

#ifndef LAMBDA_BROYDEN 
#define LAMBDA_BROYDEN 0.9999
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#define NL_MAXITER 200
#define NL_TOLER 1e-12

#define FUNASOLN 2.345

/*!
  Comment as in the Trilinos NOX package:
  
  This test problem is a modified extension of the "Broyden
  Tridiagonal Problem" from Jorge J. More', Burton S. Garbow, and
  Kenneth E. Hillstrom, Testing Unconstrained Optimization Software,
  ACM TOMS, Vol. 7, No. 1, March 1981, pp. 14-41.  The modification
  involves squaring the last equation fn(x) and using it in a
  homotopy-type equation.

  The parameter "lambda" is a homotopy-type parameter that may be
  varied from 0 to 1 to adjust the ill-conditioning of the problem.
  A value of 0 is the original, unmodified problem, while a value of
  1 is that problem with the last equation squared.  Typical values
  for increasingly ill-conditioned problems might be 0.9, 0.99,
  0.999, etc.

  The standard starting point is x(i) = -1, but setting x(i) = 0 tests
  the selected global strategy.
*/
class Broyden
{
public:
   static const int nDimSys = 8 ;

   // constructor
   __snls_hdev__  Broyden(double lambda )
      : _lambda(lambda)
      {
#ifdef __cuda_host_only__
         std::cout << "Broyden ill-conditioning: lambda = "
                   << std::setw(21) << std::setprecision(11) << _lambda << std::endl;
#endif
      } ;

   __snls_hdev__ void computeRJ(double* const r,
                                double* const J,
                                const double* x,
                                bool* const rJSuccess,
                                const int offset,
                                const int batch_size)
      {
	SNLS_FORALL(ib, offset, offset + batch_size, { 
         double fn ;
         const int nDim = nDimSys ; // convenience -- less code change below
         const int voff = ib * nDim;
         const int moff = ib * nDim * nDim;
#ifdef __cuda_host_only__         
#if DEBUG > 1
         std::cout << "Evaluating at x = " ;
         for (int i=1; i<nDim; ++i) {
            std::cout << std::setw(21) << std::setprecision(11) << x[voff + i] << " ";
         }
         std::cout << std::endl ;
#endif
#endif
         bool doComputeJ = (J != nullptr) ;
         if ( doComputeJ ) {
            for ( int ijJ=0; ijJ<_nXn; ++ijJ ) {
               J[ijJ + moff] = 0.0 ;
            }
         }
         
         r[voff + 0] = (3-2*x[voff + 0])*x[voff + 0] - 2*x[voff + 1] + 1;
         for (int i=1; i<nDim-1; i++)
            r[voff + i] = (3-2*x[voff + i])*x[voff + i] - x[voff + i-1] - 2*x[voff + i+1] + 1;

         fn = (3-2*x[voff + nDim-1])*x[voff + nDim-1] - x[voff + nDim-2] + 1;
         r[voff + nDim-1] = (1-_lambda)*fn + _lambda*(fn*fn);

         if ( doComputeJ ) {
            // F(0) = (3-2*x[0])*x[0] - 2*x[1] + 1;
            J[moff + SNLSTRDLDG_J_INDX(0,0,nDim)] = 3 - 4*x[voff + 0];
            J[moff + SNLSTRDLDG_J_INDX(0,1,nDim)] = -2;

            // F(i) = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;
            for (int i=1; i<nDim-1; i++) {
               J[moff + SNLSTRDLDG_J_INDX(i,i-1,nDim)] = -1;
               J[moff + SNLSTRDLDG_J_INDX(i,i,nDim)]   = 3 - 4*x[voff + i];
               J[moff + SNLSTRDLDG_J_INDX(i,i+1,nDim)] = -2;
            }

            // F(n-1) = ((3-2*x[n-1])*x[n-1] - x[n-2] + 1)^2;
            fn = (3-2*x[voff + nDim-1])*x[voff + nDim-1] - x[voff + nDim-2] + 1;
            double dfndxn = 3-4*x[voff + nDim-1];
            J[moff + SNLSTRDLDG_J_INDX(nDim-1,nDim-1,nDim)] = (1-_lambda)*(dfndxn) + _lambda*(2*dfndxn*fn);
            J[moff + SNLSTRDLDG_J_INDX(nDim-1,nDim-2,nDim)] = (1-_lambda)*(-1) + _lambda*(-2*fn);
         }

         rJSuccess[ib] = true ;
       });
         
      } ;
   
   private:
      double _lambda ;
      static const int _nXn = nDimSys*nDimSys ;
};

void setX(snls::SNLSTrDlDenseG_Batch<Broyden> &solver, int nDim) {
   auto mm = snls::memoryManager::getInstance();
   double *x = mm.alloc<double>(nDim);
   SNLS_FORALL(iX, 0, nDim, {
      x[iX] = 0.0;  
   });
   solver.setX(x);
   mm.dealloc<double>(x);
}

TEST(snls,broyden_a) // int main(int , char ** )
{
   const int nDim = Broyden::nDimSys ;
   const int nBatch = 4000;

   Broyden broyden( 0.9999 ) ; // LAMBDA_BROYDEN 
   snls::SNLSTrDlDenseG_Batch<Broyden> solver(broyden, nBatch) ;
   snls::TrDeltaControl_Batch deltaControlBroyden ;
   deltaControlBroyden._deltaInit = 1e0 ;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControlBroyden, 0, nBatch);
   setX(solver, nDim);
   //
   auto mm = snls::memoryManager::getInstance();
   double *r = mm.alloc<double>(nDim * nBatch);
   double *J = mm.alloc<double>(nDim*nDim * nBatch);
   bool *rjSuccess = mm.alloc<bool>(nBatch);
   //
   // any of these should be equivalent:
   // broyden.computeRJ(r, J, solver._x);
   // solver._crj.computeRJ(r, J, solver._x); 
   solver.computeRJ(&r[0], &J[0], &rjSuccess[0], 0, nBatch);

   bool status = solver.solve( ) ;
   EXPECT_TRUE( status ) << "Expected solver to converge" ;
   if ( !status ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   EXPECT_EQ( solver.getNFEvals(), 19 ) << "Expected 19 function evaluations for this case" ;
   mm.dealloc<double>(r);
   mm.dealloc<double>(J);
   mm.dealloc<bool>(rjSuccess);
}
