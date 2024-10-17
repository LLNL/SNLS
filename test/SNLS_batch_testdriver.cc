#include <cstdlib>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

using namespace std;

#include "SNLS_config.h"

#if defined(SNLS_RAJA_PORT_SUITE)

#include "SNLS_TrDLDenseG_Batch.h"
#include "SNLS_device_forall.h"
#include "SNLS_view_types.h"
#include "SNLS_memory_manager.h"

#include "chai/ManagedArray.hpp"

#ifndef LAMBDA_BROYDEN 
#define LAMBDA_BROYDEN 0.9999
#endif

#define NL_MAXITER 200
#define NL_TOLER 1e-12

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
   __snls_host__  Broyden(double lambda )
      : _lambda(lambda)
      {
#ifdef __snls_host_only__
         std::cout << "Broyden ill-conditioning: lambda = "
                   << std::setw(21) << std::setprecision(11) << _lambda << std::endl;
#endif
      } ;

   // The residual (r), Jacobian (J), and rJSuccess all have dimensions
   // of the batch size provided to the solver earlier.
   // x is of the same size as the number of points of systems your solving for
   // status is also of the same size as x and it tells us which points are still unconverged
   // Residual has dimensions (batch_size, nDimSys)
   // Jacobian has dimentions (batch_size, nDimSys, nDimSys) -> (npts, nrows, ncols) aka we follow c array ordering
   // x has the dimensions of (total_num_pts, nDimSys)
   // rJSuccess has dimensions of (batch_size)
   // success has dimensions of (total_num_pts)
   __snls_host__ void computeRJ(snls::rview2d &r,
                                snls::rview3d &J,
                                const snls::rview2d &x,
                                snls::rview1b &rJSuccess,
                                const chai::ManagedArray<snls::SNLSStatus_t>& status,
                                const int offset,
                                const int batch_size)
      {
         auto lambda = _lambda;
         snls::forall<SNLS_GPU_BLOCKS>(0, batch_size, [=] __snls_hdev__ (int ib) {
         // First check to see the current point is unconverged, and if it
         // isn't then we skip the point all together.
         if (status[ib + offset] != snls::SNLSStatus_t::unConverged) { 
            return;
         }
         double fn ;
         const int nDim = nDimSys ; // convenience -- less code change below
         // If your batch size is different from your total number of points
         // than your x array will have be offset by some amount >= 0
         const int xoff = offset + ib;
#ifdef __snls_host_only__   
#if SNLS_DEBUG > 1
         std::cout << "Evaluating at x = " ;
         for (int i=1; i<nDim; ++i) {
            std::cout << std::setw(21) << std::setprecision(11) << x(xoff, i) << " ";
         }
         std::cout << std::endl ;
#endif
#endif
         // Here we're just checking to see if the Jacobian has any size
         // at all. If it does then we'll compute the Jacobian.
         // One could have additional checks to ensure its size and
         // what you expect it to be match up.
         bool doComputeJ = (J.get_layout().size() > 0) ;
         if ( doComputeJ ) {
            for ( int ii=0; ii< nDim; ++ii ) {
               for (int jj = 0; jj < nDim; ++jj)
               {
                  J(ib, ii, jj) = 0.0;
               }
            }
         }
         
         r(ib, 0) = (3-2*x(xoff, 0))*x(xoff, 0) - 2*x(xoff, 1) + 1;
         for (int i=1; i<nDim-1; i++)
            r(ib, i) = (3-2*x(xoff, i))*x(xoff, i) - x(xoff, i-1) - 2*x(xoff, i+1) + 1;

         fn = (3-2*x(xoff, nDim-1))*x(xoff, nDim-1) - x(xoff, nDim-2) + 1;
         r(ib, nDim-1) = (1-lambda)*fn + lambda*(fn*fn);

         if ( doComputeJ ) {
            // F(0) = (3-2*x[0])*x[0] - 2*x[1] + 1;
            J(ib, 0, 0) = 3 - 4*x(xoff, 0);
            J(ib, 0, 1) = -2;

            // F(i) = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;
            for (int i=1; i<nDim-1; i++) {
               J(ib, i, i-1) = -1;
               J(ib, i, i)   = 3 - 4*x(xoff, i);
               J(ib, i, i+1) = -2;
            }

            // F(n-1) = ((3-2*x[n-1])*x[n-1] - x[n-2] + 1)^2;
            fn = (3-2*x(xoff, nDim-1))*x(xoff, nDim-1) - x(xoff, nDim-2) + 1;
            double dfndxn = 3-4*x(xoff, nDim-1);
            J(ib, nDim-1, nDim-1) = (1-lambda)*(dfndxn) + lambda*(2*dfndxn*fn);
            J(ib, nDim-1, nDim-2) = (1-lambda)*(-1) + lambda*(-2*fn);
         }

         rJSuccess(ib) = true ;
       });
         
      };
   
   private:
      double _lambda ;
      static const int _nXn = nDimSys*nDimSys ;
};

// Here we're setting out initial x values. Normally, one might do this
// in the same code block as the solver. However, the solver is in a function
// that is considered a private class function and so sadly lambda captures don't
// work. Therefore, we had to move this to a public function that the lambda captures
// would work in.
void setX(snls::batch::SNLSTrDlDenseG_Batch<Broyden> &solver, int npts) {
   // We could do this a number of differnt ways.
   // However, we're just going to use chai arrays as their easy to use
   auto mm = snls::memoryManager::getInstance();

   auto x = mm.allocManagedArray<double>(npts);
   // provide a seed so things are reproducible
   std::default_random_engine gen(42);
   std::uniform_real_distribution<double> udistrib(-1.0, 1.0);
   // Since we are using a host function to allocate things we bring our 
   // chai array memory to the host and then just initiate our initial array
   // here.
   // If we'd set the execution strategy for our Device class to CPU or OpenMP
   // then we  could have just used the forall loop here and just been fine.
   // Let's assume we've compiled with CUDA then we do something like:
   // auto device = snls::Device(ExecutionStrategy::CPU);
   // SNLS_FORALL(i, 0, npts, { /* do init stuff */ });
   // // Now set this back to the original execution strategy
   // device.SetBackend(ExecutionStrategy::CUDA);
   // If we didn't use a host function to init things
   // Then we could set the solver._x directly within a SNLS_FORALL
   double* xH = x.data(chai::ExecutionSpace::CPU);
   for (int i = 0; i < npts; i++) {
     // No idea how ill-conditioned this system is so don't want to perturb things
     // too much from our initial guess
     xH[i] = 0.001 * udistrib(gen);
   }
   // This function copies our local x array data to the solver._x array data
   // If we pass in a raw pointer like down below we need to make sure the data
   // exists in the same memory location. This is why we're getting the 
   // chai memory location that corresponds to what our Device backend is set to
   solver.setX(x.data(snls::Device::GetInstance().GetCHAIES()));
   // Alternatively, since we're using a chai managed array we could have just done the below.
   // Here the chai managed array will automatically transfer the data if need be to the correct
   // location based on our use of the snls_forall macros in setX.
   // solver.setX(x);
   // Make sure to call free() for your managedArray data type in order to properly
   // free the data when you're done using it.
   x.free();
}

TEST(snls,broyden_a) // int main(int , char ** )
{
   const int nDim = Broyden::nDimSys ;
   const int nBatch = 1000;

   Broyden broyden( 0.9999 ) ; // LAMBDA_BROYDEN 

   // Here we're setting our batch size to be the same size as the number systems
   // we want to solve for.
   snls::batch::SNLSTrDlDenseG_Batch<Broyden> solver(broyden, nBatch, nBatch);
   snls::TrDeltaInput deltaControlBroyden;
   deltaControlBroyden.deltaInit = 1.0;
   solver.setupSolver(NL_MAXITER, NL_TOLER, deltaControlBroyden, 0);
   setX(solver, nDim * nBatch);
   //
   {
      // Here we're making use of the internal solver working arrays
      // for the below computeRJ call, so we can avoid the need to
      // allocate additional data.
      snls::rview2d& rv = solver.getResidualVec();
      snls::rview3d& Jv = solver.getJacobianMat();
      snls::rview1b& rjSuccess = solver.getRJSuccessVec();
      //
      solver.computeRJ(rv, Jv, rjSuccess, 0, nBatch);
   }
   // status here tells us whether or not all the systems solved correctly.
   // If even just one system failed then this would return a value of false.
   // If one wanted to check the status for each point then one could
   // call solver.getStatusHost() which would return a pointer to the status
   // array on the host.
   bool status = solver.solve( ) ;
   EXPECT_TRUE( status ) << "Expected solver to converge" ;
   if ( !status ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }
   std::cout << "Function evaluations: " << solver.getMaxNFEvals() << "\n";
   EXPECT_EQ( solver.getMaxNFEvals(), 19 ) << "Expected 19 function evaluations for this case" ;
}

#endif //HAVE_RAJA_PERF_SUITE
