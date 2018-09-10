#include <cstdlib>
#include <iostream>

using namespace std;

#include "SNLS_TrDLDenseG.h"

#define NDIM_BROYDEN 8
#define LAMBDA_BROYDEN 0.9999
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
class SNLSBroyden : public snls::SNLSTrDlDenseG
{
public:

   // constructor
   __snls_hdev__  SNLSBroyden(int nDim, double lambda,
                         real8* nxStorage, real8 *nxXxStorage, int *niStorage,
                         int outputLevel=0 )
      : SNLSTrDlDenseG(), _lambda(-1)
      {
         _deltaControlBroyden._deltaInit = 1e0 ;
         this->setupSolver(nDim, 
                           nxStorage, nxXxStorage, niStorage,
                           NL_MAXITER, NL_TOLER, &_deltaControlBroyden,
                           outputLevel );
         _lambda = lambda ;
         std::cout << "Broyden ill-conditioning: lambda = " << _lambda << "\n";          
      } ;

   // dectructor
   __snls_hdev__ ~SNLSBroyden()
      {
         std::cout << "Function evaluations: " << this->getNFEvals() << "\n"; 
      } ;

   __snls_hdev__ bool computeRJ(real8* const r,
                                real8* const J,
                                const real8* const x )
      {
         real8 fn ;
         real8 dfndxn;
         
         r[0] = (3-2*x[0])*x[0] - 2*x[1] + 1;
         for (int i=1; i<_nDim-1; i++)
            r[i] = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;

         fn = (3-2*x[_nDim-1])*x[_nDim-1] - x[_nDim-2] + 1;
         r[_nDim-1] = (1-_lambda)*fn + _lambda*(fn*fn);

         _fevals++ ;

         // F(0) = (3-2*x[0])*x[0] - 2*x[1] + 1;
         J[SNLSTRDLDG_J_INDX(0,0)] = 3 - 4*x[0];
         J[SNLSTRDLDG_J_INDX(0,1)] = -2;

         // F(i) = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;
         for (int i=1; i<_nDim-1; i++) {
            J[SNLSTRDLDG_J_INDX(i,i-1)] = -1;
            J[SNLSTRDLDG_J_INDX(i,i)]   = 3 - 4*x[i];
            J[SNLSTRDLDG_J_INDX(i,i+1)] = -2;
         }

         // F(n-1) = ((3-2*x[n-1])*x[n-1] - x[n-2] + 1)^2;
         fn = (3-2*x[_nDim-1])*x[_nDim-1] - x[_nDim-2] + 1;
         dfndxn = 3-4*x[_nDim-1];
         J[SNLSTRDLDG_J_INDX(_nDim-1,_nDim-1)] = (1-_lambda)*(dfndxn) + _lambda*(2*dfndxn*fn);
         J[SNLSTRDLDG_J_INDX(_nDim-1,_nDim-2)] = (1-_lambda)*(-1) + _lambda*(-2*fn);

         return true ;
         
      } ;
   
   private:
      real8 _lambda ;
   snls::TrDeltaControl _deltaControlBroyden ;
};

#ifndef __cuda_host_only__

__snls_device__
void Test_SNLSBroyden_D (SNLSBroyden *snls)
{
   const int nDim=NDIM_BROYDEN;
   
   real8 nxStorage[nDim*snls::SNLSTrDlDenseG::nxMultTrDlDenseG], nxXxStorage[nDim*nDim*snls::SNLSTrDlDenseG::nxXxMultTrDlDenseG] ;
   int   niStorage[nDim*snls::SNLSTrDlDenseG::niMultTrDlDenseG] ;

   *snls = SNLSBroyden( nDim, LAMBDA_BROYDEN, 
                        nxStorage, nxXxStorage, niStorage );

   int    i   = (blockIdx.x * blockDim.x) + threadIdx.x; 

   printf("(%s::%s(ln=%d) cuda bx=%d bw=%2d thrd=%2d i=%2d snls=%p)\n", __FILE__, __func__, __LINE__, blockIdx .x, blockDim .x, threadIdx.x, i, snls);

   // real8 r[nDim], J[nDim*nDim] ;
   real8* x = snls->getXPntr() ;
   //
   for (int iX = 0; iX < nDim; ++iX) {
      x[iX] = 0e0 ;
   }

   // snls->computeRJ(r, J, x); // do not bother testing computeRJ here -- it will get executed under solve
   snls->solve( ) ;
   
}

__snls_global__
void Test_SNLSBroyden_K
(
   const int   n         ///< max number of active threads
)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x; 

   SNLSBroyden   snls[40];

   if (i<n) { Test_SNLSBroyden_D(snls+i); }

}

void snls::Test_SNLSBroyden_GPU(const int npoints)
{
   int nthrds  = 32;
   int nblks   = (npoints+(nthrds-1))/nthrds;

   Test_SNLSBroyden_K<<<nblks,nthrds>>>(npoints);
   cudaDeviceSynchronize();

   cudaError_t  cu_err     = cudaGetLastError();
   char        *cu_err_str = (char *) ( cu_err ? cudaGetErrorString(cu_err) : 0 );

   if (cu_err_str) { printf("%s::%s(ln=%d) : CUDA error=\"%s\"\n", __FILE__, __func__, __LINE__, cu_err_str ); exit(0); }
}

#endif
// ifndef __cuda_host_only__

int main(int , char ** )
{
   const int nDim=NDIM_BROYDEN;

   real8 nxStorage[nDim*snls::SNLSTrDlDenseG::nxMultTrDlDenseG], nxXxStorage[nDim*nDim*snls::SNLSTrDlDenseG::nxXxMultTrDlDenseG] ;
   int   niStorage[nDim*snls::SNLSTrDlDenseG::niMultTrDlDenseG] ;

   SNLSBroyden              thing     = SNLSBroyden( nDim, LAMBDA_BROYDEN, nxStorage, nxXxStorage, niStorage, 10 ) ;
   SNLSBroyden             *thingPntr = &(thing);
   snls::SNLSTrDlDenseG   *thingBase = dynamic_cast<snls::SNLSTrDlDenseG *>(thingPntr);

   real8* x = thingBase->getXPntr() ;
   for (int iX = 0; iX < nDim; ++iX) {
      x[iX] = 0e0 ;
   }
   //
   real8 r[nDim], J[nDim*nDim] ;
   //
   thingBase->computeRJ(&(r[0]), &(J[0]), x);

#ifdef __cuda_host_only__
   snls::SNLSTrDlDenseG::SNLSStatus_t status = thingBase->solve( ) ;
   if ( status != snls::SNLSTrDlDenseG::converged ){
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Solver failed to converge! Using tol=%g and maxIter=%i",NL_TOLER,NL_MAXITER);
      SNLS_FAIL(__func__,errmsg);
   }
#else
   int npoints=40;
   Test_SNLSBroyden_GPU(npoints);
#endif

   exit(0);
}
