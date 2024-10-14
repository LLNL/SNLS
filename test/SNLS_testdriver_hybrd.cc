#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>

#include "SNLS_config.h"
#include "SNLS_HybrdTrDLDenseG.h"
#include "SNLS_testmodels.h"

template<typename T>
inline
snls::SNLSStatus_t
hybrid_broyden_solver(T& solver) {

   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   deltaControl._xiDecDelta = 0.6;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 0);

   for (int iX = 0; iX < T::_nDim; ++iX) {
      solver.m_x[iX] = 0e0 ;
   }
   double r[T::_nDim], J[T::_nDim * T::_nDim] ;
   solver.computeRJ(r, J);

   snls::SNLSStatus_t status = solver.solve();
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_WARN(__func__,errmsg);
   }
   return status;
}

template<typename T>
inline 
snls::SNLSStatus_t
hybrid_chebyq_solver(T& solver) {

   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   deltaControl._xiDecDelta = 0.75;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 0);

   double h = 1.0 / (((double) T::_nDim) + 1.0);
   for (int j = 0; j < T::_nDim; j++){
      solver.m_x[j] = (((double) j) + 1.0) * h;
   }

   double r[T::_nDim], J[T::_nDim * T::_nDim] ;
   solver.computeRJ(r, J);

   snls::SNLSStatus_t status = solver.solve();
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_WARN(__func__,errmsg);
   }
   return status;
}

// This test will fail due to how our trust region controls our step size
// It's possible that this could be solved if we tuned the trust region parameters just right.
TEST(snls, chebyquad_a) // int main(int , char ** )
{
   constexpr int nDimSys = 3;
   auto chebyquad = make_cheby_quad3(cheby_quad_lambda(nDimSys));
   snls::SNLSHybrdTrDLDenseG<decltype(chebyquad), nDimSys> solver(chebyquad);
   snls::SNLSStatus_t status = hybrid_chebyq_solver(solver);

   std::cout << "Status " << status << std::endl;
   EXPECT_FALSE( snls::isConverged(status) ) << "Expected solver to fail" ;
}

TEST(snls, chebyquad_b) // int main(int , char ** )
{
   constexpr int nDimSys = 5;
   auto chebyquad = make_cheby_quad5(cheby_quad_lambda(nDimSys));
   snls::SNLSHybrdTrDLDenseG<decltype(chebyquad), nDimSys> solver(chebyquad);
   snls::SNLSStatus_t status = hybrid_chebyq_solver(solver);

   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   EXPECT_EQ( solver.getNFEvals(), 71) << "Expected 71 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 4) << "Expected 4 jacobian evaluations for this case";
}

TEST(snls, chebyquad_b_lambda) // int main(int , char ** )
{
   constexpr int nDimSys = 5;
   auto chebyquad = cheby_quad_lambda(nDimSys);
   snls::SNLSHybrdTrDLDenseG<decltype(chebyquad), nDimSys> solver(chebyquad);
   snls::SNLSStatus_t status = hybrid_chebyq_solver(solver);
 

   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   EXPECT_EQ( solver.getNFEvals(), 71) << "Expected 71 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 4) << "Expected 4 jacobian evaluations for this case";
}

TEST(snls, chebyquad_b_func) // int main(int , char ** )
{
   constexpr int nDimSys = 5;
   std::function<auto (double* const, double* const, const double* const) -> bool> chebyquad = cheby_quad_lambda(nDimSys);
   snls::SNLSHybrdTrDLDenseG<decltype(chebyquad), nDimSys> solver(chebyquad);
   snls::SNLSStatus_t status = hybrid_chebyq_solver(solver);
 
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   EXPECT_EQ( solver.getNFEvals(), 71) << "Expected 71 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 4) << "Expected 4 jacobian evaluations for this case";
}

#if defined(__snls_host_only__)

TEST(snls,broyden_a) // int main(int , char ** )
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.9999;
   Broyden broyden(lambda, broyden_lambda(lambda, nDimSys)); // LAMBDA_BROYDEN
   snls::SNLSHybrdTrDLDenseG<decltype(broyden)> solver(broyden);
   snls::SNLSStatus_t status = hybrid_broyden_solver(solver);

   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   std::cout << "Residual: " << solver.getRes() << "\n";
   snls::linalg::printVec<nDimSys>(solver.m_x);
   EXPECT_EQ( solver.getNFEvals(), 30) << "Expected 30 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 2) << "Expected 2 jacobian evaluations for this case";
}

TEST(snls,broyden_b) // int main(int , char ** )
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.99999999;
   Broyden broyden(lambda, broyden_lambda(lambda, nDimSys)); // LAMBDA_BROYDEN
   snls::SNLSHybrdTrDLDenseG<decltype(broyden)> solver(broyden);
   snls::SNLSStatus_t status = hybrid_broyden_solver(solver);

   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   std::cout << "Residual: " << solver.getRes() << "\n";
   snls::linalg::printVec<nDimSys>(solver.m_x);
   EXPECT_EQ( solver.getNFEvals(), 35) << "Expected 35 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 2) << "Expected 2 jacobian evaluations for this case";
}

TEST(snls,broyden_c) // int main(int , char ** )
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.99;
   Broyden broyden(lambda, broyden_lambda(lambda, nDimSys)); // LAMBDA_BROYDEN
   snls::SNLSHybrdTrDLDenseG<decltype(broyden)> solver(broyden);
   snls::SNLSStatus_t status = hybrid_broyden_solver(solver);

   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   std::cout << "Residual: " << solver.getRes() << "\n";
   snls::linalg::printVec<nDimSys>(solver.m_x);
   EXPECT_EQ( solver.getNFEvals(), 23) << "Expected 23 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 2) << "Expected 2 jacobian evaluations for this case";
}

#endif
