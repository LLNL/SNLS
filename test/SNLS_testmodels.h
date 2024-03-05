#pragma once

#include "SNLS_base.h"

#ifndef LAMBDA_BROYDEN 
#define LAMBDA_BROYDEN 0.9999
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#define NL_MAXITER 200
#define NL_TOLER 1e-12

#define FUNASOLN 2.345

// Note with CUDA compilations we run into issues similar to those described here:
// https://gitlab.lrz.de/IP/quickvec/-/issues/1 for our lambda functions if we use __snls_hdev__ or __snls_device__.
// This is a limitation of NVCC since it appears to return 2 different types when a lambda function has the
// __host__ __device__ decoration on it. One of the types is for the host compiler and the other is for the
// device compilation. So, if we use auto in the return position this causes issues for the nvcc compiler or
// at least that's my understanding of things after reading the documentation for extended lambdas in CUDA:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda .
// If we really wanted to we could probably use a functor instead or some macro usages but that's not worth the trouble.
// In the mean time, it's easier to just revert all the __snls_hdev__ calls to __snls_host__ calls so things compile. 
// However, we only see this pop-up when compiling the SNLS_benchmark.cc case
// No easy solution here as the alternative to the lambda functions is to define
// a struct that implements the code and just return that rather than having the lambda funcs...

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

inline
auto broyden_lambda(const double lambda, const int nDimSys) {
   return [nDimSys, lambda] __snls_host__(double* const r,
                            double* const J,
                            const double* const x ) -> bool 
    {
        double fn;
        const int nDim = nDimSys; // convenience -- less code change below
        const int nXn = nDim * nDim;
        
#if DEBUG > 1
        std::cout << "Evaluating at x = " ;
        for (int i=1; i<nDim; ++i) {
            std::cout << std::setw(21) << std::setprecision(11) << x[i] << " ";
        }
        std::cout << std::endl ;
#endif

        bool doComputeJ = (J != nullptr) ;
        if ( doComputeJ ) {
            for ( int ijJ=0; ijJ<nXn; ++ijJ ) {
                J[ijJ] = 0.0 ;
            }
        }
        
        r[0] = (3-2*x[0])*x[0] - 2*x[1] + 1;
        for (int i=1; i<nDim-1; i++)
            r[i] = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;

        fn = (3-2*x[nDim-1])*x[nDim-1] - x[nDim-2] + 1;
        r[nDim-1] = (1-lambda)*fn + lambda*(fn*fn);

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
            J[SNLS_NN_INDX(nDim-1,nDim-1,nDim)] = (1-lambda)*(dfndxn) + lambda*(2*dfndxn*fn);
            J[SNLS_NN_INDX(nDim-1,nDim-2,nDim)] = (1-lambda)*(-1) + lambda*(-2*fn);
        }
        return true;
    };
}

constexpr int BROYDEN_NDIM_SYS = 8;

template<class T>
class Broyden
{
public:
   static const int nDimSys = BROYDEN_NDIM_SYS;
   // constructor
   __snls_hdev__  
   Broyden(double lambda, T &&lambda_func)
    : _lambda(lambda), func(std::forward<T>(lambda_func)) {};

   __snls_host__
   bool computeRJ(double* const r,
                  double* const J,
                  const double* const x)
    {
        // Make use of a perfectly forwarded lambda function just as across all the test cases
        // we have one that needs to run on the GPU. So, we can't make use of std::function in that case
        // which is a shame as performance wise it was good enough for our use case...
        // Performance wise on benchmarking it was found this has negligible effects on things.
        return func(r, J, x);
    } ;
   
   private:
      double _lambda ;
      const T func;
      static const int _nXn = nDimSys * nDimSys ;
};

// This problem is described originally in 
// Fletcher, R. "Function minimization without evaluating derivatives - a review." The Computer Journal 8.1 (1965): 33-41.
// doi: https://doi.org/10.1093/comjnl/8.1.33
// It's original description in the Fletcher paper is a function that "represents those found in practice".
inline
auto cheby_quad_lambda(const int nDimSys) {
    return [nDimSys] __snls_host__(double* const r,
                      double* const J,
                      const double* const x) -> bool
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
    };
}

template<int nd, class T>
class ChebyQuad
{
public:
    static const int nDimSys = nd;
    ChebyQuad(T&& lambda) : func(std::forward<T>(lambda)) {}
    ~ChebyQuad(){}

    __snls_host__
    bool computeRJ(double* const r,
                    double* const J,
                    const double* const x)
    {
        return func(r, J, x);
    }
private:
    const T func;
};

template<class T>
auto make_cheby_quad3(T&& lambda) {
    return ChebyQuad<3, T>(std::forward<T>(lambda));
}

template<class T>
auto make_cheby_quad5(T&& lambda) {
    return ChebyQuad<5, T>(std::forward<T>(lambda));
}

inline
auto fun_a_lambda(const double alpha, const double xsoln) {
    return [alpha, xsoln] __snls_host__(double &f, double &J, double x) -> bool 
    {
        double arg = alpha * (x - xsoln) ;
        f = tanh( arg ) ;
        const double temp = 1.0 / cosh( arg ); // = sech( arg ) 
        J = alpha * temp * temp; 
        return true;
    };
}

template<class T>
class FunA
{
public:
    // constructor
    // Take in a lambda function that was built from fun_a_lambda
    // this way we can share implementation details across all our different
    // cases. We avoid using a std::function to hold things here
    // as it results in a 20% hit in performance.
    FunA(double alpha, T &&lambda) 
    : _alpha(alpha), _xSoln(FUNASOLN), func(std::forward<T>(lambda)) {};

    __snls_host__
    bool computeFJ(double &f,
                  double &J,
                  double  x)
    {
        return func(f, J, x);
    } ;

    __snls_host__
    void operator()(double &f,
                    double &J,
                    double  x ) { 
        this->computeFJ(f,J,x); 
    };
   
private:
    double _alpha;
    double _xSoln;
    const T func;
};
