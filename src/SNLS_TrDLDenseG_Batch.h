// -*-c++-*-

#ifndef SNLS_TRDLDG_BATCH_H
#define SNLS_TRDLDG_BATCH_H

#include "SNLS_base.h"
#include "SNLS_lup_solve.h"
#include "SNLS_device_forall.h"
#include "SNLS_memory_manager.h"
#include "RAJA/RAJA.hpp"
#include "chai/ManagedArray.hpp"

#include <stdlib.h>
#include <iostream>
#ifdef __cuda_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

#ifndef SNLS_USE_LAPACK
#define SNLS_USE_LAPACK 0
#endif

#if HAVE_LAPACK && SNLS_USE_LAPACK

extern "C" {
   int DGETRF(const int* m, const int* n, double* A, const int* lda, int* ipiv, int* info);
   int DGETRS(const char* trans, const int* n, const int* nrhs, const double* const A, const int* lda,
              const int* const ipiv, double* b, const int* ldb, int* info);
}
#endif

//////////////////////////////////////////////////////////////////////

// row-major storage
#define SNLSTRDLDG_J_INDX(p,q,nDim) (p)*(nDim)+(q)

// useful RAJA views for our needs
typedef RAJA::View<bool, RAJA::Layout<1> > rview1b;
typedef RAJA::View<double, RAJA::Layout<1> > rview1d;
typedef RAJA::View<double, RAJA::Layout<2> > rview2d;
typedef RAJA::View<double, RAJA::Layout<3> > rview3d;

#define RSETUP(wdata, es, offset) &(wdata.data(es))[offset]

namespace snls {
  namespace batch{

class TrDeltaControl_Batch
{
public:
   
   __snls_hdev__ TrDeltaControl_Batch() :
      _xiLG(0.75),
      _xiUG(1.4),
      _xiIncDelta(1.5),
      _xiLO(0.35),
      _xiUO(5.0),
      _xiDecDelta(0.25),
      _xiForcedIncDelta(1.2),
      _deltaInit(1.0),
      _deltaMin(1e-12),
      _deltaMax(1e4),
      _rejectResIncrease(true)
   {
      this->checkParams() ;
   }

   __snls_hdev__ double getDeltaInit() const { return _deltaInit;} ;

   __snls_hdev__   
   bool decrDelta(void *strm, double &delta, double normfull, bool took_full) const
   {
      bool success = true ;
      
      if ( took_full ) 
      {
         double tempa = delta    * _xiDecDelta ;
         double tempb = normfull * _xiDecDelta ;
         delta = sqrt( tempa*tempb ) ;
      }
      else 
         delta = delta * _xiDecDelta ;

      if ( delta < _deltaMin ) 
      {
         delta = _deltaMin ;

#ifdef __cuda_host_only__
         if (strm) { *((std::ostream *) strm) << "delta now at min " << delta << std::endl ; }
#endif

         success = false ;
      }

#ifdef __cuda_host_only__
      else 
         if (strm) { *((std::ostream *) strm) << "decr delta to " << delta << std::endl ; }
#endif

      return success;
   }

   __snls_hdev__
   void incrDelta(void* strm, double  &delta) const
   {
      delta = delta * _xiIncDelta;

      if ( delta > _deltaMax ) 
      {
         delta = _deltaMax ;

#ifdef __cuda_host_only__
         if (strm) { *((std::ostream *) strm) << "delta now at max " << delta << std::endl ; }
#endif
      }
#ifdef __cuda_host_only__
      else 
         if (strm) { *((std::ostream *) strm) << "incr delta to "    << delta << std::endl ; }
#endif
   }

   
   __snls_hdev__ bool updateDelta
   (
      void    * strm      ,
      double  & delta     ,
      double    res       ,
      double    res_0     ,
      double    pred_resid,
      bool    & reject    , // reject_prev
      bool      took_full , // use_nr
      double    normfull  , // nr2norm
      double  & rho       ) const
   {
      bool  success       = true;
      double actual_change = res - res_0;
      double pred_change   = pred_resid - res_0;

      if ( pred_change == 0e0 ) 
      {
         if ( delta >= _deltaMax ) {
            // things are going badly enough that the solver should probably stop
#ifdef __cuda_host_only__
            if (strm) { *((std::ostream *) strm) << "predicted change is zero and delta at max" << std::endl; }
#endif
            success = false ;
         }
         else {
#ifdef __cuda_host_only__
            if (strm) { *((std::ostream *) strm) << "predicted change is zero, forcing delta larger" << std::endl; }
#endif
            delta = fmin( delta * _xiForcedIncDelta, _deltaMax );
            // ierr = IERR_UPD_DELTA_PRZ_p; // do not report this as a failure
         }
      }
      else 
      {
         rho = actual_change / pred_change;
#ifdef __cuda_host_only__
         if (strm) { *((std::ostream *) strm) << "rho = " << rho << std::endl; }
#endif
         if ( ( rho > _xiLG ) &&
            ( actual_change < 0e0 ) &&
            ( rho < _xiUG ) ) {
            if ( ! took_full ) {
               // increase delta
               this->incrDelta(strm, delta);
            }
            // instead of truncating, just do not increase, as 
            // truncating can cause trouble
            // else { 
            // 
            //    double temp = delta;
            //    double temp_b = normfull * _xiIncDelta;
            //    if (temp_b < delta) {
            //       delta = temp_b;
#ifdef __cuda_host_only__
            //       if ( _os != NULL ) {
            //          _os << "took full step, delta truncated from " << temp << " to " << delta << std::endl;
            //       }
#endif
            //    }
            // }
         }
         else if ( ( rho < _xiLO ) ||
                  ( rho > _xiUO ) ) 
         { 
            // decrease delta
            success = this->decrDelta(strm, delta, normfull, took_full);
         }
      } // pred_change == 0
      
      reject = false;

      // do not make this >= , may have res and res_0 both zero and that is ok
      if ( ( actual_change > 0e0 ) && 
         ( _rejectResIncrease ) ) 
      { 
         // residual increasing, need to reject solution
         // have already decreased delta (rho < 0)
         reject = true;
      }
      
      return success;
   }

   __snls_hdev__ bool updateDelta
   (
      void    * strm      ,
      double  & delta     ,
      double    res       ,
      double    res_0     ,
      double    pred_resid,
      bool    & reject    , // reject_prev
      bool      took_full , // use_nr
      double    normfull   // nr2norm
      ) const
   {
      bool  success       = true;
      double actual_change = res - res_0;
      double pred_change   = pred_resid - res_0;

      if ( pred_change == 0e0 ) 
      {
         if ( delta >= _deltaMax ) {
            // things are going badly enough that the solver should probably stop
#ifdef __cuda_host_only__
            if (strm) { *((std::ostream *) strm) << "predicted change is zero and delta at max" << std::endl; }
#endif
            success = false ;
         }
         else {
#ifdef __cuda_host_only__
            if (strm) { *((std::ostream *) strm) << "predicted change is zero, forcing delta larger" << std::endl; }
#endif
            delta = fmin( delta * _xiForcedIncDelta, _deltaMax );
            // ierr = IERR_UPD_DELTA_PRZ_p; // do not report this as a failure
         }
      }
      else 
      {
         double rho = actual_change / pred_change;
#ifdef __cuda_host_only__
         if (strm) { *((std::ostream *) strm) << "rho = " << rho << std::endl; }
#endif
         if ( ( rho > _xiLG ) &&
            ( actual_change < 0e0 ) &&
            ( rho < _xiUG ) ) {
            if ( ! took_full ) {
               // increase delta
               this->incrDelta(strm, delta);
            }
            // instead of truncating, just do not increase, as 
            // truncating can cause trouble
            // else { 
            // 
            //    double temp = delta;
            //    double temp_b = normfull * _xiIncDelta;
            //    if (temp_b < delta) {
            //       delta = temp_b;
#ifdef __cuda_host_only__
            //       if ( _os != NULL ) {
            //          _os << "took full step, delta truncated from " << temp << " to " << delta << std::endl;
            //       }
#endif
            //    }
            // }
         }
         else if ( ( rho < _xiLO ) ||
                  ( rho > _xiUO ) ) 
         { 
            // decrease delta
            success = this->decrDelta(strm, delta, normfull, took_full);
         }
      } // pred_change == 0
      
      reject = false;

      // do not make this >= , may have res and res_0 both zero and that is ok
      if ( ( actual_change > 0e0 ) && 
         ( _rejectResIncrease ) ) 
      { 
         // residual increasing, need to reject solution
         // have already decreased delta (rho < 0)
         reject = true;
      }
      
      return success;
   }

private:
   
   __snls_hdev__ void checkParams() const
   {
      if ( ( _deltaMin <= 0 ) ||
         ( _deltaMax <= _deltaMin ) ||
         ( _xiLG <= _xiLO ) ||
         ( _xiUG >= _xiUO ) ||
         ( _xiIncDelta <= 1 ) ||
         ( ( _xiDecDelta >= 1 ) || ( _xiDecDelta <= 0 ) ) ||
         ( _xiForcedIncDelta <= 1 ) 
         ) {
         SNLS_FAIL("TrDeltaControl_Batch::checkParams", "bad setting") ;
      }
   }

public:
   
   double _xiLG, _xiUG, _xiIncDelta, _xiLO, _xiUO, _xiDecDelta, _xiForcedIncDelta, _deltaInit, _deltaMin, _deltaMax;
   bool  _rejectResIncrease ;
};

/** Helper templates to ensure compliant CRJ implementations */
template<typename CRJ, typename = void>
struct has_valid_computeRJ : std::false_type { static constexpr bool value = false;};

template<typename CRJ>
struct has_valid_computeRJ <
   CRJ,typename std::enable_if<
       std::is_void<
           decltype(std::declval<CRJ>().computeRJ(std::declval<rview2d&>(), std::declval<rview3d&>(), std::declval<const rview2d&>(), 
                    std::declval<rview1b&>(), std::declval<const chai::ManagedArray<SNLSStatus_t>&>(),
                    std::declval<const int>(), std::declval<const int>(), std::declval<const int>())) 
       >::value
       ,
       void
   >::type
>: std::true_type { static constexpr bool value = true;};

template<typename CRJ, typename = void>
struct has_ndim : std::false_type { static constexpr bool value = false;};

template<typename CRJ>
struct has_ndim <
   CRJ,typename std::enable_if<
       std::is_same<
           decltype(CRJ::nDimSys),
           const int  
       >::value
       ,
       void
   >::type
>: std::true_type { static constexpr bool value = true;};

// trust region type solver, dogleg approximation
// for dense general Jacobian matrix
//
// CRJ should :
// 	have member function
// 		     __snls_hdev__ computeRJ( double* const r, double* const J, const double* const x, 
//                                     bool* const rJSuccess, const int offset, const int x_offset, const int nbatch ) ;
// 		computeRJ function returns true for successful evaluation
// 		TODO ... J becomes a RAJA::View ?
//	have trait nDimSys
//
// TODO ...*** specialize to N=1 case, nad N=2 also?
//
template< class CRJ >
class SNLSTrDlDenseG_Batch 
{
   public:
      static_assert(has_valid_computeRJ<CRJ>::value, "The CRJ implementation in SNLSTrDlDenseG_Batch needs to implement void computeRJ( rview2d &r, rview3d &J, const rview2d &x,"
                                                      " rview1b &rJSuccess, const chai::ManagedArray<SNLSStatus_t> &status, const int offset, const int x_offset, const int nbatch )");
      static_assert(has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");

   public:
   /// constructor which requires the number of points to be set
   /// or else it defaults to just using 1 for batch solves
   SNLSTrDlDenseG_Batch(CRJ &crj, uint npts = 1, uint dflt_int_batch = 50000) :
               _crj(crj),
               _mfevals(0), _nIters(0), _nJFact(0),
               _deltaControl(nullptr),
               _outputLevel(0),
               _os(nullptr),
               _npts(npts),
               _int_batch_size(dflt_int_batch),
               _x(nullptr, npts, CRJ::nDimSys),
               _res(nullptr, npts),
               _delta(nullptr, npts),
               _residual(nullptr, dflt_int_batch, CRJ::nDimSys),
               _Jacobian(nullptr, dflt_int_batch, CRJ::nDimSys, CRJ::nDimSys),
               _rjSuccess(nullptr, dflt_int_batch)
   {
      init();
   };

   void init() {
      memoryManager& mm = memoryManager::getInstance();
      const auto es = snls::Device::GetCHAIES();
      const int num_allocs = _int_batch_size * (11 + (5 * CRJ::nDimSys) + CRJ::nDimSys * CRJ::nDimSys) + 
                             _npts * (2 + CRJ::nDimSys);
      wrk_data = mm.allocManagedArray<double>(num_allocs);
      _status = mm.allocManagedArray<SNLSStatus_t>(_npts);
      _fevals = mm.allocManagedArray<int>(_npts);

      wrkb_data = mm.allocManagedArray<bool>(3 * _int_batch_size);
      _rjSuccess.set_data(RSETUP(wrkb_data, es, 0));

      int offset = 0;
      _offset_work = _npts * (2 + CRJ::nDimSys)
                   + _int_batch_size * (CRJ::nDimSys + CRJ::nDimSys * CRJ::nDimSys);
      _x.set_data(RSETUP(wrk_data, es, offset));
      offset += _npts * CRJ::nDimSys;
      _res.set_data(RSETUP(wrk_data, es, offset));
      offset += _npts;
      _delta.set_data(RSETUP(wrk_data, es, offset));
      offset += _npts;

      _residual.set_data(RSETUP(wrk_data, es, offset));
      offset += _int_batch_size * CRJ::nDimSys;
      _Jacobian.set_data(RSETUP(wrk_data, es, offset));

      SNLS_FORALL(i, 0, _npts, {
         _fevals[i] = 0;
         for (int j = 0; j < CRJ::nDimSys; j++){
            _x(i, j) = 0.0;
         }
         _delta(i) = 1e8;
         _res(i) = 1e20;
         _status[i] = SNLSStatus_t::unConverged;
      });
   };
   /// destructor needs to dealloc the _x, _r, and _status variables
   ~SNLSTrDlDenseG_Batch() {
      //
      // memoryManager& mm = memoryManager::getInstance();
      wrk_data.free();
      wrkb_data.free();
      _fevals.free();
      _status.free();
      // if ( _outputLevel > 1 && _os != nullptr ) {
      //    *_os << "Function and Jacobian factorizations: " << _fevals << " " << _nJFact << std::endl;
      // }
   };

   public:
      CRJ &_crj ;
      static const int _nDim = CRJ::nDimSys ;
      // Not clear how we want to handle these in a batch job sense of things at least for 
      // NFEvals, RhoLast, Delta, and Res... Do we return the largest one or do we just
      // change this to return a const pointer to the array these values are located at?         
      int     getNDim   () const { return(_nDim   ); };
      int     getMaxNFEvals() const { return(_mfevals ); };
      const chai::ManagedArray<int> getNFEvals() const { return _fevals; };
      const rview1d& getDelta() const { return _delta; };
      const rview1d& getRes() const { return _res; };
      // These allow us to make use of the already exisiting 
      // work allocations for these variables
      rview2d& getResidualVec() { return _residual; }
      rview3d& getJacobianMat() { return _Jacobian; }
      rview1b& getrjSuccessVec() { return _rjSuccess; }

      /// setX can be used to set the initial guess for all of the points used in the batch job
      // fix me so that this has a forall loop over this
      inline void setX( const double* const x) {
         SNLS_FORALL(ipts, 0, _npts, {
            for (int iX = 0; iX < _nDim; ++iX) {
               _x(ipts, iX) = x[ipts * _nDim + iX] ;
            }
         });
      }

      inline void setX(const chai::ManagedArray<double> &x) {
         SNLS_FORALL(ipts, 0, _npts, {
            for (int iX = 0; iX < _nDim; ++iX) {
               _x(ipts, iX) = x[ipts * _nDim + iX] ;
            }
         });
      }
      /// getX can be used to get solution for all of the points used in the batch job
      // fix me so that this has a forall loop over this
      inline void getX( double* const x) const {
         SNLS_FORALL(ipts, 0, _npts, {
            for (int iX = 0; iX < _nDim; ++iX) {
               x[ipts * _nDim + iX] = _x(ipts, iX);
            }
         });
      };

      inline void getX( chai::ManagedArray<double> &x) const {
         SNLS_FORALL(ipts, 0, _npts, {
            for (int iX = 0; iX < _nDim; ++iX) {
               x[ipts * _nDim + iX] = _x(ipts, iX);
            }
         });
      };

      /**
       * Must call setupSolver before calling solve
       */
      void   setupSolver(int maxIter,
                         double tolerance,
                         TrDeltaControl_Batch * deltaControl,
                         int outputLevel=0) {
         SNLS_FORALL(i, 0, _npts, {
            _status[i] = SNLSStatus_t::unConverged ;
            _fevals[i] = 0;
         });

         _maxIter = maxIter ;
         _tolerance = tolerance ;

         _deltaControl = deltaControl ;

         this->setOutputlevel( outputLevel ) ;

      }

      void   setOutputlevel( int    outputLevel ) {
         _outputLevel = outputLevel ;
         _os          = nullptr ;
         //
         if ( _outputLevel > 0 ) {
            _os = &(std::cout) ;
         }
      }

      // solve returns bool for whether or not all systems solved successfully
      //
      // on exit, _res is consistent with _x
      bool solve() {
         // We'll want to chunk this up into pieces so that we don't become memory hogs
         
         if ( _deltaControl == nullptr ) {
            SNLS_FAIL("solve", "_deltaControl not set") ;
         }
         
         SNLS_FORALL(i, 0, _npts, {
            _status[i] = SNLSStatus_t::unConverged;
            _fevals[i] = 0;
         });

         const int numblks = (_npts + _int_batch_size - 1)/ _int_batch_size;

         int offset = 0;

         // All of our temporary variables needed for the batch solve
         // They are also all chai::ManagedArray<T>
         // This greatly simplifies our memory management story
         // The data will automatically migrate to the location it needs
         // whether it's on the host or device.
         const auto es = snls::Device::GetCHAIES();
         rview1b use_nr(RSETUP(wrkb_data, es, _int_batch_size), _int_batch_size);
         rview1b reject_prev(RSETUP(wrkb_data, es, 2 * _int_batch_size), _int_batch_size);

         int woffset = _offset_work;
         const int off2d = _int_batch_size * _nDim;
         const int off1d = _int_batch_size;
         // 11 * batch_size + 5 * (batch_size * ndim) + 1 * batch_size * ndim*ndim
         // internal memory usage for solver...
         // if ndim = 8 and batch_size = 500k then this is roughly 460MBs
         // We're probably close to 500 MBs total then for the solver at size
         // if batch_size and npts are equal
         rview1d res_0(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d nr2norm(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d alpha(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d norm_s_sd_opt(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d norm_grad(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d norm_grad_inv(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d qa(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d qb(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d Jg2(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d res_cauchy(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;
         rview1d pred_resid(RSETUP(wrk_data, es, woffset), _int_batch_size);
         woffset += off1d;

         rview2d nrStep(RSETUP(wrk_data, es, woffset), _int_batch_size, _nDim);
         woffset += off2d;
         rview2d grad(RSETUP(wrk_data, es, woffset), _int_batch_size, _nDim);
         woffset += off2d;
         rview2d delx(RSETUP(wrk_data, es, woffset), _int_batch_size, _nDim);
         woffset += off2d;
         rview2d solx(RSETUP(wrk_data, es, woffset), _int_batch_size, _nDim);
         // offset += off2d;

         int m_fevals = 0;
         int m_nJFact = 0;
         int m_nIters = 0;

         for(int iblk = 0; iblk < numblks; iblk++) {
            // fix me: modify these to become arrays sets???
            _mfevals = 0 ;
            _nJFact = 0 ;
            _nIters = 0 ;
            // Checks to see if we're the last bulk and if not use the initial batch size
            // if we are we need to either use initial batch size or the npts%initial_batch_size
            const int fin_batch = (_npts % _int_batch_size == 0) ? _int_batch_size : _npts % _int_batch_size;
            const int batch_size = (iblk != numblks - 1) ? _int_batch_size : fin_batch;

            //Reinitialize all of our data back to these values
            SNLS_FORALL_T(i, 512, 0, batch_size, {
               _rjSuccess(i) = true;
               reject_prev(i) = false;
               use_nr(i) = false;
               nr2norm(i) = 0.0;
               alpha(i) = 0.0;
               norm_s_sd_opt(i) = 0.0;
               norm_grad(i) = 0.0;
               norm_grad_inv(i) = 0.0;
               qa(i) = 0.0;
               qb(i) = 0.0;

               _delta(i + offset) = _deltaControl->getDeltaInit();

               // Setting solx to initial x value
               for (int j = 0; j < _nDim; j++)
               {
                  solx(i, j) = _x(offset + i, j);
               }
            });

            this->computeRJ(_residual, _Jacobian, _rjSuccess, offset, batch_size); // at _x

            SNLS_FORALL_T(i, 512, 0, batch_size, {
               if ( !(_rjSuccess(i)) ) {
                  _status[i + offset] = SNLSStatus_t::initEvalFailure ;
               }
               // this breaks out of the internal lambda and is essentially a loop continue
               if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  _res(i + offset) = this->normvec(&_residual.data[i * _nDim]) ;
                  res_0(i) = _res(i + offset);
            });
            // #ifdef __cuda_host_only__
            //          if (_os) { *_os << "res = " << _res << std::endl ; }
            // #endif
            //
            //We'll need somewhere to have a reduction operation to check and see that everything is done
            while ( _nIters < _maxIter ) {
               //
               _nIters += 1 ;
               SNLS_FORALL_T(i, 224, 0, batch_size, 
               { // start of batch compute kernel 1
                  // this breaks out of the internal lambda and is essentially a loop continue
                  if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  if ( !reject_prev(i) ) {
                     //
                     // have a newly accepted solution point
                     // compute information for step determination
                     // fix me: this won't work currently
                     this->computeSysMult( &_Jacobian.data[i * _nXnDim], &_residual.data[i * _nDim], &grad.data[i * _nDim], true );
                     // used to keep :
                     //	ngrad[iX] = -grad[iX] ;
                     // 	nsd[iX]   = ngrad[iX] * norm_grad_inv ; 
                     
                     // find Cauchy point
                     //
                     {
                        double norm2_grad = this->normvecSq( &grad.data[i * _nDim] ) ;
                        norm_grad(i) = sqrt( norm2_grad ) ;
                        {
                           double ntemp[_nDim] ; 
                           this->computeSysMult( &_Jacobian.data[i * _nXnDim], &grad.data[i * _nDim], &ntemp[0], false ) ; // was -grad in previous implementation, but sign does not matter
                           Jg2(i) = this->normvecSq( &ntemp[0] ) ;
                        }
                        
                        alpha(i) = 1e0 ;
                        res_cauchy(i) = res_0(i);
                        if ( Jg2(i) > 0e0 ) { 
                           alpha(i) = norm2_grad / Jg2(i) ;
                           res_cauchy(i) = sqrt(fmax(res_0(i) * res_0(i) - alpha(i) * norm2_grad, 0.0)) ;
                        }

                        // optimal step along steapest descent is alpha*ngrad
                        norm_s_sd_opt(i) = alpha(i) * norm_grad(i);
                        
                        norm_grad_inv(i) = 1e0 ;
                        if ( norm_grad(i) > 0e0 ) {
                           norm_grad_inv(i) = 1e0 / norm_grad(i);
                        }
                        
                     }

                     this->computeNewtonStep( &_Jacobian.data[i * _nXnDim], &_residual.data[i * _nDim], &nrStep.data[i * _nDim] ) ;
                     nr2norm(i) = this->normvec( &nrStep.data[i * _nDim] ) ;
                     //
                     // as currently implemented, J is not factored in-place and computeSysMult no loner works ;
                     // or would have to be reworked for PLU=J

                     {

                        // inline the use of p instead of computing it
                        //
                        // double p[_nDim] ;
                        // for (int iX = 0; iX < _nDim; ++iX) {
                        //    p[iX] = nrStep[iX] + alpha*grad[iX] ; // nrStep - alpha*ngrad
                        // }

                        {
                           double gam0 = 0e0 ;
                           double norm2_p = 0.0 ;
                           for (int iX = 0; iX < _nDim; ++iX) {
                              double p_iX = nrStep(i, iX) + alpha(i) * grad(i, iX);
                              norm2_p += p_iX * p_iX ;
                              gam0 += p_iX * grad(i, iX);
                           }
                           gam0 = -gam0 * norm_grad_inv(i);

                           qb(i) = 2.0 * gam0 * norm_s_sd_opt(i);

                           qa(i) = norm2_p ;
                           
                        }
                           
                     }

                  } // !reject_prev
                  // New batch compute kernel 1

                  // compute the step given _delta
                  //
                  // start of batch compute kernel 2
                  // this breaks out of the internal lambda and is essentially a loop continue
                  if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  use_nr(i) = false;
                  if ( nr2norm(i) <= _delta(i + offset) ) {
                     // use Newton step
                     use_nr(i) = true ;

                     for (int iX = 0; iX < _nDim; ++iX) {
                        delx(i, iX) = nrStep(i, iX);
                     }
                     pred_resid(i) = 0e0 ;

   #ifdef __cuda_host_only__
                     if ( _os != nullptr ) {
                        *_os << "trying newton step" << std::endl ;
                     }
   #endif
                  }
                  else { // use_nr

                     // step along dogleg path
                  
                     if ( norm_s_sd_opt(i) >= _delta(i + offset) ) {

                        // use step along steapest descent direction

                        {
                           double factor = -_delta(i + offset) * norm_grad_inv(i);
                           for (int iX = 0; iX < _nDim; ++iX) {
                              delx(i, iX) = factor * grad(i, iX);
                           }
                        }

                        {
                           double val = -(_delta(i + offset) * norm_grad(i)) 
                                      + 0.5 * _delta(i + offset) * _delta(i + offset) * Jg2(i)
                                      * (norm_grad_inv(i) * norm_grad_inv(i));
                           pred_resid(i) = sqrt(fmax(2.0 * val + res_0(i) * res_0(i), 0.0)) ;
                        }
               
                     }
                     else{

                        // qc and beta depend on delta
                        //
                        double qc = norm_s_sd_opt(i) * norm_s_sd_opt(i) - _delta(i + offset) * _delta(i + offset);
                        //
                        double beta = (-qb(i) + sqrt(qb(i) * qb(i) - 4.0 * qa(i) * qc))/(2.0 * qa(i)) ;
   #ifdef SNLS_DEBUG
                        if ( beta > 1.0 || beta < 0.0 ) {
                           SNLS_FAIL(__func__, "beta not in [0,1]") ;
                        }
   #endif
                        beta = fmax(0.0, fmin(1.0, beta)) ; // to deal with and roundoff

                        // delx[iX] = alpha*ngrad[iX] + beta*p[iX] = beta*nrStep[iX] - (1.0-beta)*alpha*grad[iX]
                        //
                        {
                           double omb  = 1.0 - beta ;
                           double omba = omb * alpha(i);
                           for (int iX = 0; iX < _nDim; ++iX) {
                              delx(i, iX) = beta * nrStep(i, iX) - omba * grad(i, iX);
                           }
                           pred_resid(i) = omb * res_cauchy(i);
                        }
                     } // if norm_s_sd_opt >= delta

                  } // use_nr
                  // end of batch compute kernel 2
                  // start of batch compute kernel 3
                  // this breaks out of the internal lambda and is essentially a loop continue
                  if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  this->update( &delx.data[i * _nDim], i, offset ) ;
                  reject_prev(i) = false ;
               }); // end of batch compute kernel 3
                  //
                  
               // start of batch compute kernel 4
               this->computeRJ(_residual, _Jacobian, _rjSuccess, offset, batch_size) ; // at _x
               // end of batch compute kernel 4
               SNLS_FORALL_T(i, 256, 0, batch_size, 
               { // start of batch compute kernel 5
                  // this breaks out of the internal lambda and is essentially a loop continue
                  if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  if ( !(_rjSuccess(i)) ) {
                     // got an error doing the evaluation
                     // try to cut back step size and go again
                     bool deltaSuccess = _deltaControl->decrDelta(_os, _delta(i + offset), nr2norm(i), use_nr(i) ) ;
                     if ( ! deltaSuccess ) {
                        _status[i + offset] = deltaFailure ;
                        _fevals[i + offset] = _mfevals;
                        return; // equivalent to a continue in a while loop
                     }
                     reject_prev(i) = true ;
                  }
                  else {
                     _res(i + offset) = this->normvec(&_residual.data[i * _nDim]);

                     // allow to exit now, may have forced one iteration anyway, in which
                     // case the delta update can do funny things if the residual was
                     // already very small 
                     // Now _x shouldn't change after we are no longer unconverged
                     // due to all of the exits we have at the start of the loops
                     // but just in case we save this value and assign it back when we exit 
                     if ( _res(i + offset) < _tolerance ) {
                        _status[i + offset] = converged ;
                        for (int j = 0; j < _nDim; j++) {
                           solx(i, j) = _x(offset + i, j);
                        }
                        _fevals[i + offset] = _mfevals;
                        return; // equivalent to a continue in a while loop
                     }

                     {
                        bool deltaSuccess = _deltaControl->updateDelta(_os,
                                                                     _delta(i + offset), _res(i + offset), res_0(i), pred_resid(i),
                                                                     reject_prev(i), use_nr(i), nr2norm(i)) ;
                        if ( ! deltaSuccess ) {
                           _status[i + offset] = deltaFailure ;
                           _fevals[i + offset] = _mfevals;
                           return; // equivalent to a continue in a while loop
                        }
                     }
                  }
                   // end of batch compute kernel 5
                   // start of batch compute kernel 6
                  // this breaks out of the internal lambda and is essentially a loop continue
                  if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  if ( reject_prev(i) ) { 
                     _res(i + offset) = res_0(i);
                     this->reject( &delx.data[i * _nDim], i, offset ) ;
                  }
               // end of batch compute kernel 6/5
               // end of delta step...
               //
               // start of batch compute kernel 7
                  // this breaks out of the internal lambda and is essentially a loop continue
                  if( _status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  res_0(i) = _res(i + offset);
               }); // end of batch compute kernel 7

               // If this is true then that means all of the batched items
               // either failed or converged, and so we can exit and start the new batched data
               if(status_exit<true, SNLS_GPU_THREADS>(offset, batch_size)) {
                  break;
               }
               
            } // _nIters < _maxIter

            SNLS_FORALL_T(i, 512, 0, batch_size * _nDim, {
               _x.data[offset * _nDim + i] = solx.data[i];
            });

            if(m_fevals < _mfevals) m_fevals = _mfevals;
            if(m_nJFact < _nJFact) m_nJFact = _nJFact;
            if(m_nIters < _nIters) m_nIters = _nIters;

            offset += batch_size;

         } // end of batch loop

	      _mfevals = m_fevals;
         _nJFact = m_nJFact;
         _nIters = m_nIters;

         bool converged = status_exit<false, SNLS_GPU_THREADS>(0, _npts);

         // fix me have this return a boolean
         return converged ;
      
      }

      // convenience wrapper, for the current _x
      // no longer returns a bool but has a bool array argument
      inline void computeRJ(rview2d &r,
                            rview3d &J,
                            rview1b &rJSuccess,
                            const int offset,
                            const int batch_size) {
         
         _mfevals++ ;
         // We'll probably want to modify this as well to take in a bool array
         // that it's responsible for setting
         // fix me rJSuccess needs to be passed into _crj.computeRJ
         this->_crj.computeRJ(r, J, _x, rJSuccess, _status, offset, offset, batch_size);
         
#ifdef SNLS_DEBUG
         // // Needs to be rethought of for how to do this for the vectorized format...
         // if ( _outputLevel > 2 && _os != nullptr ) {
         //    // do finite differencing
         //    // assume system is scaled such that perturbation size can be standard
         //    auto mm = snls::memoryManager::getInstance();
         //    chai::ManagedArray<double> J_FD = mm.allocManagedArray<double>(J.size());
         //    // Dummy variable since we got rid of using raw pointers
         //    // The default initialization has a size of 0 which is what we want.
         //    chai::ManagedArray<double> J_dummy;
         //    chai::ManagedArray<double> r_pert = mm.allocManagedArray<double>(r.size());
         //    chai::ManagedArray<double> x_pert = mm.allocManagedArray<double>(_x.size());

         //    const double pert_val     = 1.0e-7 ;
         //    const double pert_val_inv = 1.0/pert_val;

         //    for ( int iX = 0; iX < _nDim ; ++iX ) {
         //       SNLS_FORALL(iBatch, 0, batch_size, {
         //          const int off = SNLS_VOFF(iBatch, _nDim);
         //          const int toff = SNLS_TOFF(iBatch, offset, _nDim);
         //          for ( int jX = 0; jX < _nDim ; ++jX ) {
         //             x_pert[off + jX] = _x[toff + jX] ;
         //          }
         //          x_pert[off + iX] = x_pert[off + iX] + pert_val;
         //       });
         //       this->_crj.computeRJ(r_pert, J_dummy, x_pert, rJSuccess, offset, 0, batch_size);
         //       const bool success = reduced_rjSuccess<SNLS_GPU_THREADS>(rJSuccess, batch_size);
         //       if ( !success ) {
         //          SNLS_FAIL(__func__, "Problem while finite-differencing");
         //       }
         //       SNLS_FORALL(iBatch, 0, batch_size, {
         //          const int off = SNLS_VOFF(iBatch, _nDim);
         //          const int moff = SNLS_MOFF(iBatch, _nXnDim);
         //          for ( int iR = 0; iR < _nDim ; iR++ ) {
         //             J_FD[moff + SNLSTRDLDG_J_INDX(iR, iX, _nDim)] = pert_val_inv * ( r_pert[off + iR] - r[off + iR] );
         //          }
         //       });
         //    }

         //    const double* J_data = J.data(chai::ExecutionSpace::CPU);
         //    const double* J_FD_data = J_FD.data(chai::ExecutionSpace::CPU);
         //    for (int iBatch = 0; iBatch < batch_size; iBatch++) {
         //       const int moff = SNLS_MOFF(iBatch, _nXnDim);
         //       *_os << "Batch item # " << iBatch << std::endl;
         //       *_os << "J_an = " << std::endl ; printMatJ( &J_data[moff],    *_os );
         //       *_os << "J_fd = " << std::endl ; printMatJ( &J_FD_data[moff], *_os );
         //    }

         //    // Clean-up the memory these objects use.
         //    x_pert.free();
         //    r_pert.free();
         //    J_FD.free();
         //    J_dummy.free();

         //    // put things back the way they were ;
         //    this->_crj.computeRJ(r, J, _x, rJSuccess, offset, offset, batch_size);
         // } // _os != nullptr
#endif         
         
      }
      
   private :
   
     __snls_hdev__ inline void  computeNewtonStep (double* const       J,
                                      const double* const r,
                                      double* const       newton  ) {
         
         _nJFact++ ;
         
#if HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
// This version of the Newton solver uses the LAPACK solver DGETRF() and DGETRS()
// 
// Note that we can replace this with a custom function if there are performance 
// specializations (say for a known fixed system size)
   
         // row-major storage
         const char trans = 'T';

         // LAPack is probably not the most efficient for the system sizes of interest ;
         // even simple linpack dgefa and dgesl would probably be better ;
         // but for now, just go with it

         int info=0;
         int ipiv[_nDim] ;
         DGETRF(&_nDim, &_nDim, J, &_nDim, ipiv, &info) ;

         if ( info != 0 ) { SNLS_FAIL(__func__, "info non-zero from dgetrf"); }

         // std::copy( r, r + _nDim, newton );
         for (int iX = 0; iX < _nDim; ++iX) {
            newton[iX] = - r[iX] ; 
         }

         int nRHS=1; info=0;
         DGETRS(&trans, &_nDim, &nRHS, J, &_nDim, ipiv, newton, &_nDim, &info);

         if ( info != 0 ) { SNLS_FAIL(__func__, "info non-zero from lapack::dgetrs()") ; }

#else
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)

         {
            const int n = _nDim;

            int   err = SNLS_LUP_Solve<n>(J, newton, r);
            if (err<0) {
               SNLS_FAIL(__func__," fail return from LUP_Solve()") ;
            }
            //
            for (int i=0; (i<n); ++i) { newton[i] = -newton[i]; }

         }
#endif
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
      }
      
      __snls_hdev__ inline void  computeSysMult(const double* const J, 
                                  const double* const v,
                                  double* const       p,
                                  bool                transpose ) {
         
         // row-major storage
         bool sysByV = not transpose ;
   
         if ( sysByV ) {
            int ipX = 0 ;
            for (int pX = 0; pX < _nDim; ++pX) {
               p[pX] = 0e0 ;
               for (int kX = 0; kX < _nDim; ++kX) {
                  p[pX] += J[kX+ipX] * v[kX] ;
               }
               ipX += _nDim ; // ipX = pX*_nDim ;
            }
         }
         else {
            for (int pX = 0; pX < _nDim; ++pX) {
               p[pX] = 0e0 ;
            }
            int ikX = 0 ;
            for (int kX = 0; kX < _nDim; ++kX) {
               for (int pX = 0; pX < _nDim; ++pX) {
                  p[pX] += J[ikX+pX] * v[kX] ;
               }
               ikX += _nDim ; // ikW = kX*_nDim ;
            }
         }
      }
      // fix me this needs to be udated to be a compute kernel overall batches
      __snls_hdev__ inline void  update(const double* const delX, const int ielem, const int offset ) {
         const int toffset = ielem + offset;
         //delX is already offset
         for (int iX = 0; iX < _nDim; ++iX) {
            _x(toffset, iX) = _x(toffset, iX) + delX[iX] ;
         }
      }
      // fix me this needs to be provided an offset value
      __snls_hdev__ inline void  reject(const double* const delX, const int ielem, const int offset ) {
         const int toffset = ielem + offset;
         // delX is already offset
         for (int iX = 0; iX < _nDim; ++iX) {
            _x(toffset, iX) = _x(toffset, iX) - delX[iX] ;
         }
      }
      
      __snls_hdev__ inline double normvec(const double* const v) {
         return sqrt( this->normvecSq(v) ) ;
      }
      
      __snls_hdev__ inline double normvecSq(const double* const v) {
         double a = 0e0;
         for (int iX = 0; iX < _nDim; ++iX) {
            a += v[iX]*v[iX] ;
         }
         return a ;
      }
      
   public:

      /// Performs a bitwise reduction and operation on _status to see if the
      /// current batch can exit.
      template <const bool batch_loop, const int NUMTHREADS>
      inline bool status_exit(const int offset,
                              const int batch_size)
      {
         // Additional backends can be added as seen within the MFEM_FORALL
         // which this was based on.
            
         // Device::Backend makes use of a global variable
         // so as long as this is set in one central location
         // and you don't have multiple Device objects changing
         // the backend things should just work no matter where this
         // is used.
         bool red_add = false;
         const int end = offset + batch_size;
         SNLSStatus_t*  status = _status.data(Device::GetCHAIES());
#ifdef HAVE_RAJA
         switch(Device::GetBackend()) {
#ifdef RAJA_ENABLE_CUDA
            case(ExecutionStrategy::CUDA): {
               //RAJA::ReduceBitAnd<RAJA::cuda_reduce, bool> output(init_val);
               RAJA::ReduceSum<RAJA::cuda_reduce, int> output(0);
               RAJA::forall<RAJA::cuda_exec<512>>(RAJA::RangeSegment(offset, end), [=] __snls_device__ (int i) {
                  if(!batch_loop) { 
                     if(isConverged(status[i])) output += 1;
                  }
                  else {
                     if(status[i] != SNLSStatus_t::unConverged) output += 1;
                  }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
   #endif
   #if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
            case(ExecutionStrategy::OPENMP): {
               //RAJA::ReduceBitAnd<RAJA::omp_reduce_ordered, bool> output(init_val);
               RAJA::ReduceSum<RAJA::omp_reduce_ordered, int> output(0);
               RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(offset, end), [=] (int i) {
                  if(!batch_loop) { 
                     if(isConverged(status[i])) output += 1;
                  }
                  else {
                     if(status[i] != SNLSStatus_t::unConverged) output += 1;
                  }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
   #endif
            case(ExecutionStrategy::CPU):
            default: {
               //RAJA::ReduceBitAnd<RAJA::seq_reduce, bool> output(init_val);
               RAJA::ReduceSum<RAJA::seq_reduce, int> output(0);
               RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(offset, end), [=] (int i) {
                  if(!batch_loop) { 
                     if(isConverged(status[i])) output += 1;
                  }
                  else {
                     if(status[i] != SNLSStatus_t::unConverged) output += 1;
                  }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
         } // End of switch
      #endif
         return red_add;
      } // end of status_exitable

      /// Performs a bitwise reduction and operation on _status to see if the
      /// current batch can exit.
      template <const int NUMTHREADS>
      inline bool reduced_rjSuccess(rview1b &rJSuccess,
                                    const int batch_size)
      {
         // Additional backends can be added as seen within the MFEM_FORALL
         // which this was based on.
            
         // Device::Backend makes use of a global variable
         // so as long as this is set in one central location
         // and you don't have multiple Device objects changing
         // the backend things should just work no matter where this
         // is used.
         bool red_add = false;
         const int end = batch_size;
#ifdef HAVE_RAJA
         switch(Device::GetBackend()) {
#ifdef RAJA_ENABLE_CUDA
            case(ExecutionStrategy::CUDA): {
               //RAJA::ReduceBitAnd<RAJA::cuda_reduce, bool> output(init_val);
               RAJA::ReduceSum<RAJA::cuda_reduce, int> output(0);
               RAJA::forall<RAJA::cuda_exec<512>>(RAJA::RangeSegment(0, end), [=] __snls_device__ (int i) {
                  if (rJSuccess(i)) { output += 1; }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
#endif
#if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
            case(ExecutionStrategy::OPENMP): {
               //RAJA::ReduceBitAnd<RAJA::omp_reduce_ordered, bool> output(init_val);
               RAJA::ReduceSum<RAJA::omp_reduce_ordered, int> output(0);
               RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, end), [=] (int i) {
                  if (rJSuccess(i)) { output += 1; }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
#endif
            case(ExecutionStrategy::CPU):
            default: {
               //RAJA::ReduceBitAnd<RAJA::seq_reduce, bool> output(init_val);
               RAJA::ReduceSum<RAJA::seq_reduce, int> output(0);
               RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, end), [=] (int i) {
                  if (rJSuccess(i)) { output += 1; }
               });
               red_add = (output.get() == batch_size) ? true : false;
               break;
            }
         } // End of switch
#endif
         return red_add;
      } // end of status_exitable

      // Returns a pointer to the status array now on the host.
      // The user is responsible for deleting this ptr after their done using it
      // through the use of the SNLS::memoryManager::dealloc<T>(T* ptr) function  
      SNLSStatus_t * getStatusHost() {
         return _status.data(chai::ExecutionSpace::CPU);
      }
   
#ifdef SNLS_DEBUG
#ifdef __cuda_host_only__
      void  printVecX (const double* const y, std::ostream & oss ) {
         oss << std::setprecision(14) ;
         for ( int iX=0; iX<_nDim; ++iX) {
            oss << y[iX] << " " ;
         }
         oss << std::endl ;
      }
      
      void  printMatJ (const double* const A, std::ostream & oss ) {
         oss << std::setprecision(14) ;
         for ( int iX=0; iX<_nDim; ++iX) {
            for ( int jX=0; jX<_nDim; ++jX) {
               oss << std::setw(21) << std::setprecision(11) << A[SNLSTRDLDG_J_INDX(iX,jX,_nDim)] << " " ;
            }
            oss << std::endl ;
         } 
      }
#endif
#endif

   // fix me lots of the below need to become arrays of values...
   public:
      rview2d _x;

   protected:
      static const int _nXnDim = _nDim * _nDim ;
      
      int _mfevals, _nIters, _nJFact ;
      chai::ManagedArray<int> _fevals;
      rview1d _delta;
      rview1d _res;
      rview2d _residual;
      rview3d _Jacobian;
      rview1b _rjSuccess;
      chai::ManagedArray<bool>   wrkb_data;
      chai::ManagedArray<double> wrk_data;

   private:
      TrDeltaControl_Batch* _deltaControl ;

      int    _offset_work;
      int    _maxIter ;
      double _tolerance ;
      int    _outputLevel ;
      uint   _int_batch_size ;
      uint   _npts ;

      // _rhoLast is not really needed -- but is kept for debug and testing purposes
      // fix me given the above we should probably delete this then
      double _rhoLast ;

      std::ostream* _os ;

      chai::ManagedArray<SNLSStatus_t> _status;
};
  } // namespace batch
} // namespace snls

#endif  // SNLS_TRDLDG_BATCH_H
