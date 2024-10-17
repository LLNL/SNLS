#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

using namespace std;

#include "SNLS_config.h"
#include "SNLS_base.h"
#include "SNLS_TrDLDenseG.h"
#include "SNLS_NewtonBB.h"
#include "SNLS_testmodels.h"

template<typename T>
inline
snls::SNLSStatus_t
broyden_solver(T& solver, const double delta_init = 1.0) {
   snls::TrDeltaControl deltaControlBroyden ;
   deltaControlBroyden._deltaInit = delta_init;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControlBroyden, 0);

   for (int iX = 0; iX < T::_nDim; ++iX) {
      solver._x[iX] = 0e0 ;
   }
   //
   double r[T::_nDim], J[T::_nDim * T::_nDim] ;
   solver.computeRJ(r, J);

   snls::SNLSStatus_t status = solver.solve( ) ;
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }

   return status;
}

TEST(snls,broyden_a) // int main(int , char ** )
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.9999;
   Broyden broyden(lambda, broyden_lambda(lambda, nDimSys)); // LAMBDA_BROYDEN
   snls::SNLSTrDlDenseG<decltype(broyden)> solver(broyden);
   snls::SNLSStatus_t status = broyden_solver(solver);

   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   EXPECT_EQ( solver.getNFEvals(), 19 ) << "Expected 19 function evaluations for this case";
}

TEST(snls,broyden_a_lambda) // int main(int , char ** )
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.9999;
   auto broyden = broyden_lambda(lambda, nDimSys);
   snls::SNLSTrDlDenseG<decltype(broyden), nDimSys> solver(broyden);
   snls::SNLSStatus_t status = broyden_solver(solver);

   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   EXPECT_EQ( solver.getNFEvals(), 19 ) << "Expected 19 function evaluations for this case";
}

TEST(snls,broyden_b)
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.99999999;
   Broyden broyden(lambda, broyden_lambda(lambda, nDimSys)); // LAMBDA_BROYDEN
   snls::SNLSTrDlDenseG<decltype(broyden)> solver(broyden);
   snls::SNLSStatus_t status = broyden_solver(solver, 100e0);

   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";    
   EXPECT_EQ( solver.getNFEvals(), 23 ) << "Expected 23 function evaluations for this case" ;
}

TEST(snls,broyden_c) // int main(int , char ** )
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.99;
   Broyden broyden(lambda, broyden_lambda(lambda, nDimSys)); // LAMBDA_BROYDEN
   snls::SNLSTrDlDenseG<decltype(broyden)> solver(broyden);
   snls::SNLSStatus_t status = broyden_solver(solver, 1e-4);

   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   EXPECT_EQ( solver.getNFEvals(), 35 ) << "Expected 35 function evaluations for this case" ;
}

TEST(snls,newtonbb_a)
{
   const double alpha = 5.0;
   const double xsoln = FUNASOLN;
   FunA fun(5.0, fun_a_lambda(alpha, xsoln));
   snls::NewtonBB solver(fun, true);

   double x = 0.0;
   snls::SNLSStatus_t status = solver.solve(x, 0.0, 0.0) ;
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;   
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";    
   EXPECT_LT( fabs(x-FUNASOLN), 1e-7 ) << "Expected the correct solution" ;
   EXPECT_EQ( solver.getNFEvals(), 16 ) << "Expected 16 function evaluations for this case" ;
}


TEST(snls,newtonbb_lambda_a)
{

   const double alpha = 5.0;
   const double xsoln = FUNASOLN;
   auto fun = fun_a_lambda(alpha, xsoln);
   snls::NewtonBB solver(fun, true);

   double x = 0.0;
   snls::SNLSStatus_t status = solver.solve(x, 0.0, 0.0);
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;   
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";    
   EXPECT_LT( fabs(x-FUNASOLN), 1e-7 ) << "Expected the correct solution" ;
   EXPECT_EQ( solver.getNFEvals(), 16 ) << "Expected 16 function evaluations for this case" ;
}

TEST(snls,newtonbb_func_a)
{
   const double alpha = 5.0;
   const double xsoln = FUNASOLN;
   std::function<auto (double &f, double &J, double x) -> bool> fun = fun_a_lambda(alpha, xsoln);
   snls::NewtonBB solver(fun, true);

   double x = 0.0;
   snls::SNLSStatus_t status = solver.solve(x, 0.0, 0.0);
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;   
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";    
   EXPECT_LT( fabs(x-FUNASOLN), 1e-7 ) << "Expected the correct solution" ;
   EXPECT_EQ( solver.getNFEvals(), 16 ) << "Expected 16 function evaluations for this case" ;
}

TEST(snls,newtonbb_b)
{
   const double alpha = 5.0;
   const double xsoln = FUNASOLN;
   FunA fun(5.0, fun_a_lambda(alpha, xsoln));
   snls::NewtonBB solver(fun, true);

   double x = 0.0 ;
   snls::SNLSStatus_t status = solver.solve(x, -10.0, 10.0) ;
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;   
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";    
   EXPECT_LT( fabs(x-FUNASOLN), 1e-7 ) << "Expected the correct solution" ;
   EXPECT_EQ( solver.getNFEvals(), 9 ) << "Expected 9 function evaluations for this case" ;
}

TEST(snls,newtonbb_c)
{
   const double alpha = 5.0;
   const double xsoln = FUNASOLN;
   FunA fun(5.0, fun_a_lambda(alpha, xsoln));
   snls::NewtonBB solver(fun, true);

   double x = FUNASOLN+2.0 ;
   snls::SNLSStatus_t status = solver.solve(x, FUNASOLN+1.0, FUNASOLN+10.0) ;
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;   
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";    
   EXPECT_LT( fabs(x-FUNASOLN), 1e-7 ) << "Expected the correct solution" ;
   EXPECT_EQ( solver.getNFEvals(), 10 ) << "Expected 10 function evaluations for this case" ;
}
