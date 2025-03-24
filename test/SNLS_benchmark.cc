#include "SNLS_base.h"

#include "SNLS_TrDLDenseG.h"
#include "SNLS_NewtonBB.h"
#include "SNLS_HybrdTrDLDenseG.h"
#include "SNLS_testmodels.h"

#include <cstdlib>
#include <iostream>
#include <sstream>

#include <benchmark/benchmark.h>

#if defined(__GNUC__)
#define BENCHMARK_NOINLINE __attribute__((noinline))
#else
#define BENCHMARK_NOINLINE
#endif

template<typename T>
inline double
broyden_solver(T& solver) {
   snls::TrDeltaControl deltaControlBroyden ;
   deltaControlBroyden._deltaInit = 1e0 ;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControlBroyden, 0);

   for (int iX = 0; iX < solver.getNDim(); ++iX) {
      solver._x[iX] = 0e0 ;
   }
   //
   double r[solver.getNDim()], J[solver.getNDim() * solver.getNDim()] ;
   solver.computeRJ(r, J); 

   snls::SNLSStatus_t status = solver.solve( ) ;
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }

   return solver.getRes();
}

template<typename T>
inline double
hybrid_broyden_solver(T& solver) {

   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   deltaControl._xiDecDelta = 0.6;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 0);

   for (int iX = 0; iX < solver.getNDim(); ++iX) {
      solver.m_x[iX] = 0e0 ;
   }
   double r[solver.getNDim()], J[solver.getNDim() * solver.getNDim()] ; 
   solver.computeRJ(r, J); 

   snls::SNLSStatus_t status = solver.solve();
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_WARN(__func__,errmsg);
   }
   return solver.getRes();
}

template<typename T>
inline double
newtonbb_solver(T& solver) {
   const double bounds = -10000.0;
   double x = bounds ;
   solver.solve(x, bounds, bounds) ;
   return x;
}

double BENCHMARK_NOINLINE broyden_a_class()
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.9999;

   Broyden broyden(lambda, broyden_lambda(lambda, nDimSys)); // LAMBDA_BROYDEN 
   snls::SNLSTrDlDenseG<decltype(broyden)> solver(broyden);
   return broyden_solver(solver);
}

double BENCHMARK_NOINLINE broyden_a_lambda()
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.9999;
   auto broyden = broyden_lambda(lambda, nDimSys);
   snls::SNLSTrDlDenseG<decltype(broyden), nDimSys> solver(broyden) ;
   return broyden_solver(solver);
}

double BENCHMARK_NOINLINE broyden_a_func()
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.9999;
   std::function<auto (double* const r, 
                      double* const J, 
                      const double* const x ) -> bool> broyden = broyden_lambda(lambda, nDimSys);

   snls::SNLSTrDlDenseG<decltype(broyden), nDimSys> solver(broyden) ;
   return broyden_solver(solver);
}

double BENCHMARK_NOINLINE hybrid_broyden_a_class()
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.9999;
   Broyden broyden(lambda, broyden_lambda(lambda, nDimSys)); // LAMBDA_BROYDEN
   snls::SNLSHybrdTrDLDenseG<decltype(broyden)> solver(broyden);
   return hybrid_broyden_solver(solver);
}

double BENCHMARK_NOINLINE hybrid_broyden_a_lambda()
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.9999;
   auto broyden = broyden_lambda(lambda, nDimSys);
   snls::SNLSHybrdTrDLDenseG<decltype(broyden), nDimSys> solver(broyden) ;
   return hybrid_broyden_solver(solver);
}

double BENCHMARK_NOINLINE hybrid_broyden_a_func()
{
   constexpr int nDimSys = BROYDEN_NDIM_SYS;
   constexpr double lambda = 0.9999;
   std::function<auto (double* const r, 
                       double* const J, 
                       const double* const x ) -> bool> broyden = broyden_lambda(lambda, nDimSys);

   snls::SNLSHybrdTrDLDenseG<decltype(broyden), nDimSys> solver(broyden) ;
   return hybrid_broyden_solver(solver);
}

double BENCHMARK_NOINLINE newtonbb_a_class()
{
   const double alpha = 5.0;
   const double xsoln = FUNASOLN;
   FunA fun(5.0, fun_a_lambda(alpha, xsoln));
   snls::NewtonBB<decltype(fun)> solver(fun, true);
   return newtonbb_solver(solver);
}

double BENCHMARK_NOINLINE newtonbb_a_lambda()
{
   const double alpha = 5.0;
   const double xsoln = FUNASOLN;
   auto fun = fun_a_lambda(alpha, xsoln);
   snls::NewtonBB<decltype(fun)> solver(fun, true);
   return newtonbb_solver(solver);

}

double BENCHMARK_NOINLINE newtonbb_a_func()
{
   const double alpha = 5.0;
   const double xsoln = FUNASOLN;
   std::function<auto (double &f, double &J, double x) -> bool> fun = fun_a_lambda(alpha, xsoln);
   snls::NewtonBB<decltype(fun)> solver(fun, true) ;
   return newtonbb_solver(solver);

}

static void BM_Broyden_Class(benchmark::State& state) {
  double residual = 0.0;
  for (auto _ : state) residual = broyden_a_class();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << residual << std::endl;
  state.SetLabel(ss.str());
}

static void BM_Broyden_Lambda(benchmark::State& state) {
  double residual = 0.0;
  for (auto _ : state) residual = broyden_a_lambda();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << residual << std::endl;
  state.SetLabel(ss.str());
}

static void BM_Broyden_Func(benchmark::State& state) {
  double residual = 0.0;
  for (auto _ : state) residual = broyden_a_func();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << residual << std::endl;
  state.SetLabel(ss.str());
}

static void BM_Hybrid_Broyden_Class(benchmark::State& state) {
  double residual = 0.0;
  for (auto _ : state) residual = hybrid_broyden_a_class();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << residual << std::endl;
  state.SetLabel(ss.str());
}

static void BM_Hybrid_Broyden_Lambda(benchmark::State& state) {
  double residual = 0.0;
  for (auto _ : state) residual = hybrid_broyden_a_lambda();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << residual << std::endl;
  state.SetLabel(ss.str());
}

static void BM_Hybrid_Broyden_Func(benchmark::State& state) {
  double residual = 0.0;
  for (auto _ : state) residual = hybrid_broyden_a_func();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << residual << std::endl;
  state.SetLabel(ss.str());
}

static void BM_NewtonBB_Class(benchmark::State& state) {
  double x = 0.0;
  for (auto _ : state) x = newtonbb_a_class();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << x << std::endl;
  state.SetLabel(ss.str());
}

static void BM_NewtonBB_Lambda(benchmark::State& state) {
  double x = 0.0;
  for (auto _ : state) x = newtonbb_a_lambda();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << x << std::endl;
  state.SetLabel(ss.str());
}

static void BM_NewtonBB_Func(benchmark::State& state) {
  double x = 0.0;
  for (auto _ : state) x = newtonbb_a_func();
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << x << std::endl;
  state.SetLabel(ss.str());
}

// Test all of our different test cases to check our performance for different cases
BENCHMARK(BM_Broyden_Class);
BENCHMARK(BM_Broyden_Lambda);
BENCHMARK(BM_Broyden_Func);
BENCHMARK(BM_Hybrid_Broyden_Class);
BENCHMARK(BM_Hybrid_Broyden_Lambda);
BENCHMARK(BM_Hybrid_Broyden_Func);
BENCHMARK(BM_NewtonBB_Class);
BENCHMARK(BM_NewtonBB_Lambda);
BENCHMARK(BM_NewtonBB_Func);
BENCHMARK_MAIN();

