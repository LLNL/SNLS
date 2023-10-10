#include <cstdlib>
#include <iostream>

#include <benchmark/benchmark.h>

using namespace std;

#include "SNLS_config.h"
#include "SNLS_TrDLDenseG.h"
#include "SNLS_NewtonBB.h"

#ifndef LAMBDA_BROYDEN 
#define LAMBDA_BROYDEN 0.9999
#endif

#define NL_MAXITER 200
#define NL_TOLER 1e-12

#define FUNASOLN 2.345

#if defined(__GNUC__)
#define BENCHMARK_NOINLINE __attribute__((noinline))
#else
#define BENCHMARK_NOINLINE
#endif

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
        //  std::cout << "Broyden ill-conditioning: lambda = "
        //            << std::setw(21) << std::setprecision(11) << _lambda << std::endl;          
      } ;

   __snls_hdev__ bool computeRJ(double* const r,
                                double* const J,
                                const double* const x )
      {
         double fn ;
         const int nDim = nDimSys ; // convenience -- less code change below
         
#if DEBUG > 1
         std::cout << "Evaluating at x = " ;
         for (int i=1; i<nDim; ++i) {
            std::cout << std::setw(21) << std::setprecision(11) << x[i] << " ";
         }
         std::cout << std::endl ;
#endif

         bool doComputeJ = (J != nullptr) ;
         if ( doComputeJ ) {
            for ( int ijJ=0; ijJ<_nXn; ++ijJ ) {
               J[ijJ] = 0.0 ;
            }
         }
         
         r[0] = (3-2*x[0])*x[0] - 2*x[1] + 1;
         for (int i=1; i<nDim-1; i++)
            r[i] = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;

         fn = (3-2*x[nDim-1])*x[nDim-1] - x[nDim-2] + 1;
         r[nDim-1] = (1-_lambda)*fn + _lambda*(fn*fn);

         if ( doComputeJ ) {
            // F(0) = (3-2*x[0])*x[0] - 2*x[1] + 1;
            J[SNLS_NN_INDX(0,0,nDim)] = 3 - 4*x[0];
            J[SNLS_NN_INDX(0,1,nDim)] = -2;

            // F(i) = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;
            for (int i=1; i<nDim-1; i++) {
               J[SNLS_NN_INDX(i,i-1,nDim)] = -1;
               J[SNLS_NN_INDX(i,i,nDim)]   = 3 - 4*x[i];
               J[SNLS_NN_INDX(i,i+1,nDim)] = -2;
            }

            // F(n-1) = ((3-2*x[n-1])*x[n-1] - x[n-2] + 1)^2;
            fn = (3-2*x[nDim-1])*x[nDim-1] - x[nDim-2] + 1;
            double dfndxn = 3-4*x[nDim-1];
            J[SNLS_NN_INDX(nDim-1,nDim-1,nDim)] = (1-_lambda)*(dfndxn) + _lambda*(2*dfndxn*fn);
            J[SNLS_NN_INDX(nDim-1,nDim-2,nDim)] = (1-_lambda)*(-1) + _lambda*(-2*fn);
         }

         return true ;
         
      } ;
   
   private:
      double _lambda ;
      static const int _nXn = nDimSys*nDimSys ;
};

class FunA
{
public:

   // constructor
   __snls_hdev__  FunA(double alpha ) : _alpha(alpha) { _xSoln = FUNASOLN; } ;

   __snls_hdev__ bool computeFJ(double &f,
                                double &J,
                                double  x )
      {
         double arg = _alpha * (x-_xSoln) ;
         f = tanh( arg ) ;
         double temp = 1.0 / cosh( arg ) ; // = sech( arg ) 
         J = _alpha * temp * temp ; 
         return true ;
      } ;

   void operator()(double &f,
                   double &J,
                   double  x ) { this->computeFJ(f,J,x); } ;
   
   private:
      double _alpha ;
      double _xSoln ;
   
};

double BENCHMARK_NOINLINE broyden_a()
{
   const int nDim = Broyden::nDimSys ;

   Broyden broyden( 0.9999 ) ; // LAMBDA_BROYDEN 
   snls::SNLSTrDlDenseG<Broyden> solver(broyden) ;
   snls::TrDeltaControl deltaControlBroyden ;
   deltaControlBroyden._deltaInit = 1e0 ;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControlBroyden, 0);

   for (int iX = 0; iX < nDim; ++iX) {
      solver._x[iX] = 0e0 ;
   }
   //
   double r[nDim], J[nDim*nDim] ;
   //
   // any of these should be equivalent:
   // broyden.computeRJ(r, J, solver._x);
   // solver._crj.computeRJ(r, J, solver._x); 
   solver.computeRJ(r, J); 

   snls::SNLSStatus_t status = solver.solve( ) ;
//    EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }
//    std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";

   return solver.getRes();
}

static void BM_Broyden(benchmark::State& state) {
  double residual = 0.0;
  for (auto _ : state) residual = broyden_a();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << residual << std::endl;
  state.SetLabel(ss.str());
}

double BENCHMARK_NOINLINE newtonbb_a()
{
   FunA fun(5.0) ;
   snls::NewtonBB<FunA,true> solver(&fun) ;
   const double bounds = -10000.0;
   double x = bounds ;
   snls::SNLSStatus_t status = solver.solve(x, bounds, bounds) ;
   // EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;   
   // std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";    
   // EXPECT_LT( fabs(x-FUNASOLN), 1e-7 ) << "Expected the correct solution" ;
   // EXPECT_EQ( solver.getNFEvals(), 16 ) << "Expected 16 function evaluations for this case" ;
   return x;
}

static void BM_NewtonBB(benchmark::State& state) {
  double x = 0.0;
  for (auto _ : state) x = newtonbb_a();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << x << std::endl;
  state.SetLabel(ss.str());
}

BENCHMARK(BM_Broyden);
BENCHMARK(BM_NewtonBB);
BENCHMARK_MAIN();
