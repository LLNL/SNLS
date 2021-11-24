#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

using namespace std;

#include "SNLS_config.h"
#include "SNLS_NLSPowell.h"
#include "SNLS_TrDelta.h"
#include "SNLS_kernels.h"

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
         std::cout << "Broyden ill-conditioning: lambda = "
                   << std::setw(21) << std::setprecision(11) << _lambda << std::endl;          
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

template<int nd>
class ChebyQuad
{
   public:
   static const int nDimSys = nd;
   ChebyQuad(){}
   ~ChebyQuad(){}

   bool computeRJ(double* const r,
                  double* const J,
                  const double* const x)
   {
      const int n = nDimSys;
      double temp1, temp2, temp, ti, d1, tk;
      double temp3, temp4;
      int iev;
    
      for (int i = 0; i < n; i++) {
         r[i] = 0.0;
      }
      
      for (int j = 0; j < n; j++) {
         temp1 = 1.0;
         temp2 = 2.0 * x[j] - 1.0;
         temp = 2.0 * temp2;
         for (int i = 0; i < n; i++) {
               r[i] += temp2;
               ti = temp * temp2 - temp1;
               temp1 = temp2;
               temp2 = ti;
         }
      }

      tk = 1.0 / ((double) n);
      iev = -1;
      for (int k = 0; k < n; k++) {
         r[k] = tk * r[k];
         if (iev > 0) {
               d1 = ((double) k) + 1.0;
               r[k] += 1.0 / (d1 * d1 - 1.0);
         }
         iev = -iev;
      }
      if (J != nullptr) {
         tk = 1.0 / ((double) n);
         for (int j = 0; j < n; j++) {
            temp1 = 1.0;
            temp2 = 2.0 * x[j] - 1.0;
            temp = 2.0 * temp2;
            temp3 = 0.0;
            temp4 = 2.0;
            for (int k = 0; k < n; k++) {
                  J[SNLS_NN_INDX(j, k, nDimSys)] = tk * temp4;
                  ti = 4. * temp2 + temp * temp4 - temp3;
                  temp3 = temp4;
                  temp4 = ti;
                  ti = temp * temp2 - temp1;
                  temp1 = temp2;
                  temp2 = ti;
            }
         }
      }
            
    return true;
   }

};

TEST(snls, chebyquad_a) // int main(int , char ** )
{
   const int nDim = ChebyQuad<3>::nDimSys;

   ChebyQuad<3> chebyquad; // LAMBDA_BROYDEN 
   snls::SNLSHybrdTrDLDenseG<ChebyQuad<3>> solver(chebyquad);
   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 1);
  
   double h = 1.0 / (((double) nDim) + 1.0);
   for (int j = 0; j < nDim; j++){
      solver.m_x[j] = (((double) j) + 1.0) * h;
   }
   //
   double r[nDim], J[nDim*nDim] ;
   //
   // any of these should be equivalent:
   // broyden.computeRJ(r, J, solver._x);
   // solver._crj.computeRJ(r, J, solver._x);
   solver.computeRJ(r, J); 

   snls::SNLSStatus_t status = solver.solve( );
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   EXPECT_EQ( solver.getNFEvals(), 30) << "Expected 29 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 3) << "Expected 3 jacobian evaluations for this case";
}

TEST(snls, chebyquad_b) // int main(int , char ** )
{
   const int nDim = ChebyQuad<5>::nDimSys;

   ChebyQuad<5> chebyquad; // LAMBDA_BROYDEN 
   snls::SNLSHybrdTrDLDenseG<ChebyQuad<5>> solver(chebyquad);
   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 1);
  
   double h = 1.0 / (((double) nDim) + 1.0);
   for (int j = 0; j < nDim; j++){
      solver.m_x[j] = (((double) j) + 1.0) * h;
   }
   //
   double r[nDim], J[nDim*nDim] ;
   //
   // any of these should be equivalent:
   // broyden.computeRJ(r, J, solver._x);
   // solver._crj.computeRJ(r, J, solver._x);
   solver.computeRJ(r, J); 

   snls::SNLSStatus_t status = solver.solve( );
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   EXPECT_EQ( solver.getNFEvals(), 75) << "Expected 75 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 8) << "Expected 8 jacobian evaluations for this case";
}

TEST(snls,broyden_a) // int main(int , char ** )
{
   const int nDim = Broyden::nDimSys ;

   Broyden broyden(0.9999) ; // LAMBDA_BROYDEN 
   snls::SNLSHybrdTrDLDenseG<Broyden> solver(broyden);
   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 1);

   for (int iX = 0; iX < nDim; ++iX) {
      solver.m_x[iX] = 0e0 ;
   }
   //
   double r[nDim], J[nDim*nDim] ;
   //
   // any of these should be equivalent:
   // broyden.computeRJ(r, J, solver._x);
   // solver._crj.computeRJ(r, J, solver._x); 
   solver.computeRJ(r, J); 

   snls::SNLSStatus_t status = solver.solve( );
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   std::cout << "Residual: " << solver.getRes() << "\n";
   snls::linalg::printVec<nDim>(solver.m_x);
   EXPECT_EQ( solver.getNFEvals(), 36) << "Expected 36 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 3) << "Expected 3 jacobian evaluations for this case";
}

TEST(snls,broyden_b) // int main(int , char ** )
{
   const int nDim = Broyden::nDimSys ;

   Broyden broyden(0.99999999) ; // LAMBDA_BROYDEN 
   snls::SNLSHybrdTrDLDenseG<Broyden> solver(broyden);
   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 1);

   for (int iX = 0; iX < nDim; ++iX) {
      solver.m_x[iX] = 0e0 ;
   }
   //
   double r[nDim], J[nDim*nDim] ;
   //
   // any of these should be equivalent:
   // broyden.computeRJ(r, J, solver._x);
   // solver._crj.computeRJ(r, J, solver._x); 
   solver.computeRJ(r, J); 

   snls::SNLSStatus_t status = solver.solve( );
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   std::cout << "Residual: " << solver.getRes() << "\n";
   snls::linalg::printVec<nDim>(solver.m_x);
   EXPECT_EQ( solver.getNFEvals(), 41) << "Expected 41 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 3) << "Expected 3 jacobian evaluations for this case";
}

TEST(snls,broyden_c) // int main(int , char ** )
{
   const int nDim = Broyden::nDimSys ;

   Broyden broyden(0.99) ; // LAMBDA_BROYDEN 
   snls::SNLSHybrdTrDLDenseG<Broyden> solver(broyden);
   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 1);

   for (int iX = 0; iX < nDim; ++iX) {
      solver.m_x[iX] = 0e0 ;
   }
   //
   double r[nDim], J[nDim*nDim] ;
   //
   // any of these should be equivalent:
   // broyden.computeRJ(r, J, solver._x);
   // solver._crj.computeRJ(r, J, solver._x); 
   solver.computeRJ(r, J); 

   snls::SNLSStatus_t status = solver.solve( );
   EXPECT_TRUE( snls::isConverged(status) ) << "Expected solver to converge" ;
   if ( status != snls::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }
   solver.computeRJ(r, J); 
   std::cout << "Function evaluations: " << solver.getNFEvals() << "\n";
   std::cout << "Jacobian evaluations: " << solver.getNJEvals() << "\n";
   // This returns a slightly different solution from the MINPACK HYBRDJ and SNLS_TrDLDenseG solvers
   // However, it also returns a residual that's around the same value. So, it result
   // in a similar minimum to the PDE...
   EXPECT_EQ( solver.getNFEvals(), 28) << "Expected 28 function evaluations for this case";
   EXPECT_EQ( solver.getNJEvals(), 3) << "Expected 3 jacobian evaluations for this case";
}

