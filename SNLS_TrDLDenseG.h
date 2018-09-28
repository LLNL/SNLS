#ifndef SNLS_TRDLDG_H
#define SNLS_TRDLDG_H

#include <iostream>

#include "SNLS_cuda_portability.h"

//////////////////////////////////////////////////////////////////////

#include "SNLS_port.h"
#if HAVE_MSLIB
#include "MS_Matmodel.h"
#else

extern "C" {
   int DGETRF(const int* m, const int* n, double* A, const int* lda, int* ipiv, int* info);
   int DGETRS(const char* trans, const int* n, const int* nrhs, const double* const A, const int* lda,
              const int* const ipiv, double* b, const int* ldb, int* info);
}


#ifdef _WIN32
#include <mathimf.h>
#else
#include <math.h>
#endif

#ifndef M_PI
#define M_PI       3.14159265358979323846264338327950288
#endif

#ifndef M_SQRT2    
#define M_SQRT2    1.41421356237309504880168872420969808
#endif

typedef double real8 ;

#endif

//////////////////////////////////////////////////////////////////////

#if HAVE_MSLIB
#include "MS_math.h"
#include "MS_Log.h"
// MS fortran wrappers for LAPack solvers
#include "MS_FortranWrappers.h"
#ifdef __cuda_host_only__
#define SNLS_FAIL(loc,str) MS_Fail(loc,str);
#else
#define SNLS_FAIL(loc,str) MS_Fail(loc,str);
#endif
#else
#ifdef __cuda_host_only__
#include <stdio.h>
#define SNLS_FAIL(loc,str) fprintf(stderr, "Failure in %s\n\t%s\n",loc,str) ; fflush(stderr) ; exit(EXIT_FAILURE) ;
#else
#define SNLS_FAIL(loc,str) printf(stderr, "Failure in %s : %s\n",loc,str) ;
#endif
#endif

//////////////////////////////////////////////////////////////////////

#define SNLSTRDLDG_J_COLUMN_MAJOR 1

#if SNLSTRDLDG_J_COLUMN_MAJOR
// column-major storage
#define SNLSTRDLDG_J_INDX(p,q) (p)+(q)*_nDim
#else
// row-major storage
#define SNLSTRDLDG_J_INDX(p,q) (p)*_nDim+(q)
#endif

namespace snls {

class TrDeltaControl
{
public:
   __snls_hdev__          TrDeltaControl();
   __snls_hdev__ virtual ~TrDeltaControl() {};

   __snls_hdev__ real8 getDeltaInit() const { return _deltaInit;} ;


   __snls_hdev__ void  incrDelta   (void *strm, real8 &delta) const;
   __snls_hdev__ bool  decrDelta   (void *strm, real8 &delta, real8 normfull, bool took_full) const;

   __snls_hdev__ bool  updateDelta (void   *strm       ,
                                    real8  &delta      ,
                                    real8   res        ,
                                    real8   res_0      ,
                                    real8   pred_resid ,
                                    bool   &reject_prev,
                                    bool    took_full  ,
                                    real8   normfull   ,
                                    real8  &rho         ) const ;

private:
   __snls_hdev__ void checkParams() const ;
public:
   real8 _xiLG, _xiUG, _xiIncDelta, _xiLO, _xiUO, _xiDecDelta, _xiForcedIncDelta, _deltaInit, _deltaMin, _deltaMax;
   bool  _rejectResIncrease ;

};

// trust region type solver, dogleg approximation
// for dense general Jacobian matrix
//
class SNLSTrDlDenseG
{
   public:
      static const int nxMultTrDlDenseG   = 10; // may eventually be able to reduce this
      static const int nxXxMultTrDlDenseG =  3;
      static const int niMultTrDlDenseG   =  1;

      typedef enum {
         converged          =  0,
         initEvalFailure    = -2,
         unConverged        = -10,
         deltaFailure       = -20,
         algFailure         = -100,
         convFailure        =  1
      } SNLSStatus_t ;

   public:

   // constructor
   __snls_hdev__ SNLSTrDlDenseG() :
               _nDim(-1), _nXnDim(-1),
               _fevals(0), _nIters(0), _nJFact(0),
               _r(NULL), _x(NULL), _J(NULL), 
               _deltaControl(NULL),
               _outputLevel(0),
               _os(NULL),
               _nxStorage(NULL), _x0(NULL), _nr(NULL), _delx(NULL), _ngrad(NULL), _nsd(NULL), _ntemp(NULL), _p(NULL), _rScratch(NULL),
               _nxXxStorage(NULL), _J0(NULL), _JScratch(NULL),
               _niStorage(NULL), _ipiv(NULL),
               _status(unConverged)
               {};

   public:
               
      // destructor
      __snls_hdev__ virtual ~SNLSTrDlDenseG() {};

      __snls_hdev__ int    getNDim   () const { return(_nDim  ); };
      __snls_hdev__ int    getNFEvals() const { return(_fevals); };
      __snls_hdev__ real8* getXPntr  () const { return(_x     ); };
      __snls_hdev__ real8* getRPntr  () const { return(_r     ); };
      __snls_hdev__ real8* getJPntr  () const { return(_J     ); };

      __snls_hdev__ void   setupSolver(int nDim,
                                  real8* const nxStorage, real8* const nxXxStorage, int* const niStorage,
                                  int    maxIter, real8  tolerance ,
                                  TrDeltaControl* deltaControl,
                                  int    outputLevel=0 );

      __snls_hdev__ void   setOutputlevel( int    outputLevel ) ;
      
      // solve returns status
      __snls_hdev__ SNLSStatus_t solve();

      // computeRJ functions return true for successful evaluation
      __snls_hdev__ bool  computeRJ                 () ;
      __snls_hdev__ virtual bool  computeRJ         (real8* const r, real8* const J, const real8* const x )=0 ;
      
      __snls_hdev__ virtual void  computeNewtonStep (real8* const newton  ) ;
      __snls_hdev__ virtual void  computeSysMult    (const real8* const v, real8* const p, bool transpose ) ; // p = J . v, or p = J^T . v
      __snls_hdev__ virtual void  update            (const real8* const delX ) ;
      __snls_hdev__ virtual void  reject            ();
      __snls_hdev__ virtual void  set0              ();
      __snls_hdev__ virtual real8 normvec           (const real8* const v);
      __snls_hdev__ virtual real8 normvecSq         (const real8* const v);
      __snls_hdev__ virtual void  printVecX         (const real8* const y ) ;
      __snls_hdev__ virtual void  printMatJ         (const real8* const A ) ;


   protected:
      int _nDim, _nXnDim;
      int _fevals, _nIters, _nJFact ;
      real8 *_r, *_x, *_J ;

   private:
      TrDeltaControl* _deltaControl ;

      int   _maxIter    ;
      real8 _tolerance  ;
      int   _outputLevel;

#ifdef __cuda_host_only__
      std::ostream* _os ;
#else
      char* _os ; // do not use
#endif

      real8 *_nxStorage, *_x0, *_nr, *_delx, *_ngrad, *_nsd, *_ntemp, *_p, *_rScratch ;
      real8 *_nxXxStorage, *_J0, *_JScratch ;
      int   *_niStorage, *_ipiv ;

      SNLSStatus_t  _status ;
};

} // namespace snls

#endif  // MS_SNLS_include
