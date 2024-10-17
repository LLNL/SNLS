// -*-c++-*-
#pragma once
// SNLS_base includes the type defs of the raja views
#include "SNLS_base.h"
#if defined(SNLS_RAJA_PORT_SUITE)
#include "SNLS_linalg.h"
#include "SNLS_lup_solve.h"
#include "SNLS_TrDelta.h"
#include "SNLS_kernels_batch.h"

#include "SNLS_device_forall.h"
#include "SNLS_view_types.h"
#include "SNLS_memory_manager.h"
#include "RAJA/RAJA.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/managed_ptr.hpp"

#include <stdlib.h>
#include <iostream>
#ifdef __snls_host_only__
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
/// This macro provides a simple way to set-up raja views given a chai
/// array (wdata), the chai execution space (es), and the offset into
/// that chai array for the raja view
#define SNLS_RSETUP(wdata, es, offset) &(wdata.data(es))[offset]
// Macros related to the number of local and global single and vector
// data for either double or bool data
#define SNLS_BATCH_LOCAL_SPS 4
#define SNLS_BATCH_GLOBAL_SPS 2
#define SNLS_BATCH_LOCAL_VPS 4
#define SNLS_BATCH_BOOL_PTS 3

namespace snls {
namespace batch{

// trust region type solver, dogleg approximation
// for dense general Jacobian matrix
template< class CRJ >
class SNLSTrDlDenseG_Batch 
{
   public:
      static_assert(snls::batch::has_valid_computeRJ<CRJ>::value, "The CRJ implementation in SNLSTrDlDenseG_Batch needs to implement void computeRJ( rview2d &r, rview3d &J, const rview2d &x,"
                                                                  " rview1b &rJSuccess, const chai::ManagedArray<SNLSStatus_t> &status, const int offset, const int nbatch )");
      static_assert(snls::batch::has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");

   public:
   /// constructor which requires the number of points to be set
   /// or else it defaults to just using 1 for batch solves
   SNLSTrDlDenseG_Batch(CRJ &crj, uint npts = 1, uint dflt_initial_batch = 50000) :
               _crj(crj),
               _x(nullptr, npts, CRJ::nDimSys),
               _mfevals(0), _nIters(0),
               _delta(nullptr, npts),
               _res(nullptr, npts),
               _residual(nullptr, dflt_initial_batch, CRJ::nDimSys),
               _Jacobian(nullptr, dflt_initial_batch, CRJ::nDimSys, CRJ::nDimSys),
               _rjSuccess(nullptr, dflt_initial_batch),
               _deltaControl(nullptr),
               _outputLevel(0),
               _initial_batch_size(dflt_initial_batch),
               _npts(npts),
               _os(nullptr)
   {
      init();
   }

   void init() {
      // Create all of the working arrays at initializations to reduce the number
      // of mallocs we need to create through out the life of the solver object
      memoryManager& mm = memoryManager::getInstance();
      const auto es = snls::Device::GetInstance().GetCHAIES();
      // Values multiplied by _initial_batch_size are related to the various
      // working arrays needed such as the residual and Jacobian matrices
      // Values multiplied _npts are those useful to the user such as the
      // solution variable _x, _delta, and l2 norm of residual (_res)
      // Working arrays used in the solve function are:
      // 1d arrays [_initial_batch_size]
      // res_0, nr_norm, Jg2, pred_resid
      // 2d arrays [_initial_batch_size, CRJ::nDimSys]
      // nrStep, grad, delx, _residual
      // 3d arrays [_initial_batch_size, CRJ::nDimSys, CRJ::nDimSys]
      // _Jacobian
      const int num_allocs = _initial_batch_size * (SNLS_BATCH_LOCAL_SPS + (SNLS_BATCH_LOCAL_VPS * CRJ::nDimSys) + CRJ::nDimSys * CRJ::nDimSys) +
                             _npts * (SNLS_BATCH_GLOBAL_SPS + CRJ::nDimSys);
      wrk_data = mm.allocManagedArray<double>(num_allocs);
      _status = mm.allocManagedArray<SNLSStatus_t>(_npts);
      _fevals = mm.allocManagedArray<int>(_npts);

      // These are boolean working arrays used in the solve routine
      // _rjSuccess is the only one accessible for external uses
      // 1d arrays [_initial_batch_size]
      // _rjSuccess, use_nr, reject_prev
      wrkb_data = mm.allocManagedArray<bool>(SNLS_BATCH_BOOL_PTS * _initial_batch_size);
      _rjSuccess.set_data(SNLS_RSETUP(wrkb_data, es, 0));

      int offset = 0;
      _offset_work = _npts * (SNLS_BATCH_GLOBAL_SPS + CRJ::nDimSys)
                   + _initial_batch_size * (CRJ::nDimSys + CRJ::nDimSys * CRJ::nDimSys);
      _x.set_data(SNLS_RSETUP(wrk_data, es, offset));
      offset += _npts * CRJ::nDimSys;
      _res.set_data(SNLS_RSETUP(wrk_data, es, offset));
      offset += _npts;
      _delta.set_data(SNLS_RSETUP(wrk_data, es, offset));
      offset += _npts;

      _residual.set_data(SNLS_RSETUP(wrk_data, es, offset));
      offset += _initial_batch_size * CRJ::nDimSys;
      _Jacobian.set_data(SNLS_RSETUP(wrk_data, es, offset));

      auto& x = _x;
      auto& delta = _delta;
      auto& res = _res;
      auto& status = _status;
      auto& fevals = _fevals;

      snls::forall<SNLS_GPU_BLOCKS>(0, _npts, [=] __snls_hdev__ (int i) {
         fevals[i] = 0;
         for (int j = 0; j < CRJ::nDimSys; j++){
            x(i, j) = 0.0;
         }
         delta(i) = 1e8;
         res(i) = 1e20;
         status[i] = SNLSStatus_t::unConverged;
      });
   }
   /// destructor needs to dealloc the wrk_data, wrkb_data, _feval, and _status variables
   ~SNLSTrDlDenseG_Batch() {
      wrk_data.free();
      wrkb_data.free();
      _fevals.free();
      _status.free();
   }

   public:
      CRJ &_crj ;
      static constexpr int _nDim = CRJ::nDimSys ;
      /// The size of the nonlinear system of equations being solved for
      int     getNDim   () const { return(_nDim   ); }
      /// Returns the maximum of function evaluations across all the nonlinear system solves
      int     getMaxNFEvals() const { return(_mfevals ); }
      /// Returns the maximum of jacobian evaluations across all the nonlinear system solves
      int     getMaxNJEvals() const { return(_mfevals ); }
      /// Returns the function evaluation array for each point
      const chai::ManagedArray<int> getNFEvals() const { return _fevals; }
      /// Returns the jacobian evaluation array for each point
      const chai::ManagedArray<int> getNJEvals() const { return _fevals; }
      /// Returns the size of the delta step used as part of the dogleg solve of the
      /// PDE
      const rview1d& getDelta() const { return _delta; }
      /// Returns the L2 norm of the residual vector of the nonlinear systems being solved for
      const rview1d& getRes() const { return _res; }
      /// The working array for the residual vector
      /// Useful if one wants to do a computeRJ call outside of the solve func
      /// It has dimensions of (_intial_batch_size, _nDim) and follows c array striding
      rview2d& getResidualVec() { return _residual; }
      /// The working array for the jacobian matrix
      /// Useful if one wants to do a computeRJ call outside of the solve func
      /// It has dimensions of (_intial_batch_size, _nDim, _nDim) and follows c array striding
      rview3d& getJacobianMat() { return _Jacobian; }
      /// The working array for the rjSuccess vector
      /// Useful if one wants to do a computeRJ call outside of the solve func
      /// It has dimensions of (_intial_batch_size) and follows c array striding
      rview1b& getRJSuccessVec() { return _rjSuccess; }

      /// setX can be used to set the initial guess for all of the points used in the batch job
      inline void setX( const double* const x) {
         auto& lx = _x;
         constexpr size_t nDim = _nDim;
         snls::forall<SNLS_GPU_BLOCKS>(0, _npts, [=] __snls_hdev__ (int ipts) {
            for (size_t iX = 0; iX < nDim; ++iX) {
               lx(ipts, iX) = x[ipts * nDim + iX] ;
            }
         });
      }

      /// setX can be used to set the initial guess for all of the points used in the batch job
      inline void setX(const chai::ManagedArray<double> &x) {
         auto& lx = _x;
         constexpr size_t nDim = _nDim;
         snls::forall<SNLS_GPU_BLOCKS>(0, _npts, [=] __snls_hdev__ (int ipts) {
            for (size_t iX = 0; iX < nDim; ++iX) {
               lx(ipts, iX) = x[ipts * nDim + iX] ;
            }
         });
      }

      /// getX can be used to get solution for all of the points used in the batch job
      inline void getX( double* const x) const {
         auto& lx = _x;
         constexpr size_t nDim = _nDim;
         snls::forall<SNLS_GPU_BLOCKS>(0, _npts, [=] __snls_hdev__ (int ipts) {
            for (size_t iX = 0; iX < nDim; ++iX) {
               x[ipts * nDim + iX] = lx(ipts, iX);
            }
         });
      }

      /// getX can be used to get solution for all of the points used in the batch job
      inline void getX( chai::ManagedArray<double> &x) const {
         auto& lx = _x;
         constexpr size_t nDim = _nDim;
         snls::forall<SNLS_GPU_BLOCKS>(0, _npts, [=] __snls_hdev__ (int ipts) {
            for (size_t iX = 0; iX < nDim; ++iX) {
               x[ipts * nDim + iX] = lx(ipts, iX);
            }
         });
      }

      /**
       * Must call setupSolver before calling solve
       */
      void setupSolver(int maxIter,
                         double tolerance,
                         TrDeltaInput& tdi,
                         int outputLevel=0) {

         _maxIter = maxIter ;
         _tolerance = tolerance ;
         // Need to see if this will actually work at all...
         chai::ManagedArray<snls::TrDeltaInput> temp(1);
         temp.data(chai::ExecutionSpace::CPU);
         temp[0] = tdi;
         _deltaControl = chai::make_managed<snls::TrDeltaControl>(chai::unpack(temp));

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

      /// solve returns bool for whether or not all systems solved successfully
      ///
      /// on exit, _res is consistent with _x
      bool solve() {
         if ( _deltaControl == nullptr ) {
            SNLS_FAIL("solve", "_deltaControl not set") ;
         }

         auto& rjSuccess = _rjSuccess;
         auto& status = _status;
         auto& res = _res;
         auto& residual = _residual;
         auto deltaControl = _deltaControl;
         auto& delta = _delta;
         auto& fevals = _fevals;
         auto& Jacobian = _Jacobian;
         auto& x = _x;
         auto& tolerance = _tolerance;
         constexpr size_t nXnDim = _nXnDim;
         constexpr size_t nDim = _nDim;

         snls::forall<SNLS_GPU_BLOCKS>(0, _npts, [=] __snls_hdev__ (int i) {
            status[i] = SNLSStatus_t::unConverged;
            fevals[i] = 0;
         });

         const int numblks = (_npts + _initial_batch_size - 1)/ _initial_batch_size;

         int offset = 0;

         // All of our temporary variables needed for the batch solve
         // We make use of the working arrays created initially to reuse
         // memory if multiple solves are called during the life of this object
         const auto es = snls::Device::GetInstance().GetCHAIES();
         rview1b use_nr(SNLS_RSETUP(wrkb_data, es, _initial_batch_size), _initial_batch_size);
         rview1b reject_prev(SNLS_RSETUP(wrkb_data, es, 2 * _initial_batch_size), _initial_batch_size);

         int woffset = _offset_work;
         const int off2d = _initial_batch_size * _nDim;
         const int off1d = _initial_batch_size;

         rview1d res_0(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d nr_norm(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d Jg_2(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;
         rview1d pred_resid(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size);
         woffset += off1d;

         rview2d nrStep(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size, _nDim);
         woffset += off2d;
         rview2d grad(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size, _nDim);
         woffset += off2d;
         rview2d delx(SNLS_RSETUP(wrk_data, es, woffset), _initial_batch_size, _nDim);
         woffset += off2d;

         int m_fevals = 0;
         int m_nIters = 0;

         for(int iblk = 0; iblk < numblks; iblk++) {

            _mfevals = 0 ;
            _nIters = 0 ;
            // Checks to see if we're the last block and if not use the initial batch size
            // if we are we need to either use initial batch size or the npts%initial_batch_size
            const int fin_batch = (_npts % _initial_batch_size == 0) ? _initial_batch_size : _npts % _initial_batch_size;
            const int batch_size = (iblk != numblks - 1) ? _initial_batch_size : fin_batch;

            // Reinitialize all of our data back to these values
            snls::forall<512>(0, batch_size, [=] __snls_hdev__ (int i) {
               rjSuccess(i) = true;
               reject_prev(i) = false;
               use_nr(i) = false;
               nr_norm(i) = 0.0;
               Jg_2(i) = 0.0;

               delta(i + offset) = deltaControl->getDeltaInit();
            });

            this->computeRJ(residual, Jacobian, rjSuccess, offset, batch_size); // at _x

            snls::forall<512>(0, batch_size, [=] __snls_hdev__ (int i) {
               if ( !(rjSuccess(i)) ) {
                  status[i + offset] = SNLSStatus_t::initEvalFailure ;
               }
               // this breaks out of the internal lambda and is essentially a loop continue
               if( status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  res(i + offset) = snls::linalg::norm<nDim>(&residual.get_data()[i * nDim]);
                  res_0(i) = res(i + offset);
            });

            while ( _nIters < _maxIter ) {
               //
               _nIters += 1 ;
               // Start of batch compute kernel 1
               // This loop contains the LU solve and the
               // update to gradient term if previous solution wasn't rejected.
               // Previously, we had this kernel fused with the rest of the dogleg
               // algo and update x portion of code. However, we couldn't easily split
               // up the kernels if we did the above...
               // We  might be able to move some or all of this code to other kernels in the code
               // to reduce number of kernels launches which could help with performance.
               snls::forall<SNLS_GPU_BLOCKS>(0, batch_size, [=] __snls_hdev__ (int i)
               {  // start of cauchy point calculations
                  // this breaks out of the internal lambda and is essentially a loop continue
                  if( status[i + offset] != SNLSStatus_t::unConverged){ return; }
                  if ( !reject_prev(i) ) {
                     snls::linalg::matTVecMult<nDim, nDim>(&Jacobian.get_data()[i * nXnDim], &residual.get_data()[i * nDim], &grad.get_data()[i * nDim]);
                     {
                        double ntemp[nDim] ;
                        snls::linalg::matVecMult<nDim, nDim>(&Jacobian.get_data()[i * nXnDim], &grad.get_data()[i * nDim], ntemp); // was -grad in previous implementation, but sign does not matter
                        Jg_2(i) = snls::linalg::dotProd<nDim>(ntemp, ntemp);
                     }
                     const bool sol_stat = this->computeNewtonStep( &Jacobian.get_data()[i * nXnDim], &residual.get_data()[i * nDim], &nrStep.get_data()[i * nDim] ) ;
                     if (!sol_stat) {
                        status[i + offset] = SNLSStatus_t::linearSolveFailure;
                        return;
                     }
                     nr_norm(i) = snls::linalg::norm<nDim>( &nrStep.get_data()[i * nDim] );
                  }
               }); // end of batch compute kernel 1
               // Computes the batch version of the dogleg code and updates the solution variable x
               snls::batch::dogleg<nDim>(offset, batch_size, status, delta, res_0, nr_norm, Jg_2, grad, nrStep,
                                          delx, x, pred_resid, use_nr);
               // This calculates the new residual and jacobian given the updated x value
               this->computeRJ(residual, Jacobian, rjSuccess, offset, batch_size) ; // at _x

               // updates delta based on a trust region
               // if the solution is rejected than x is also returned to its previous value
               snls::batch::updateDelta<nDim>(offset, batch_size, _mfevals, deltaControl,
                                               residual, pred_resid, nr_norm, use_nr, rjSuccess, tolerance,
                                               delx, x, res_0, res, delta, reject_prev, fevals, status);

               // If this is true then that means all of the batched items
               // either failed or converged, and so we can exit and start the new batched data
               if(status_exit<true>(offset, batch_size)) {
                  break;
               }
            } // _nIters < _maxIter

            if(m_fevals < _mfevals) m_fevals = _mfevals;
            if(m_nIters < _nIters) m_nIters = _nIters;

            offset += batch_size;

         } // end of batch loop

         _mfevals = m_fevals;
         _nIters = m_nIters;

         bool converged = status_exit<false>(0, _npts);
         return converged ;
      }

      /// Computes the residual, jacobian, and whether or not 
      /// the computation was a success using the current _x
      /// state of the solver. If there is an offset to the 
      /// _x variable that is supplied as well as the current
      /// batch size.
      inline void computeRJ(rview2d &r,
                            rview3d &J,
                            rview1b &rJSuccess,
                            const int offset,
                            const int batch_size) {
         
         _mfevals++ ;
         auto& x = _x;
         auto& status = _status;
         this->_crj.computeRJ(r, J, x, rJSuccess, status, offset, batch_size);
#ifdef SNLS_DEBUG
         // This is not efficient at all but it'll work for debugging cases
         if ( _outputLevel > 2 && _os != nullptr ) {
            // do finite differencing
            // assume system is scaled such that perturbation size can be standard
            auto mm = snls::memoryManager::getInstance();
            chai::ManagedArray<double> cJ_FD = mm.allocManagedArray<double>(_initial_batch_size * _nXnDim);
            // Need to copy the Jacobian data over to a local variable.
            // Since, I'm not aware of a way to just copy a portion of underlying
            // chai::managedArray over to host/device
            chai::ManagedArray<double> cJ = mm.allocManagedArray<double>(_initial_batch_size * _nXnDim);
            const int tot_npts = _initial_batch_size * _nXnDim;
            snls::forall<SNLS_GPU_BLOCKS>(0, tot_npts, [=] __snls_hdev__ (int i) {
               // If we want to completely avoid inner loops
               // we have to do the following to get the correct
               // indexing into J.
               // Although, we do incur the cost of a few divisions this
               // way. Nonetheless, this sort of way of doing things can
               // really boost performance on the GPU if our compute kernels
               // are friendly towards this sort of thing.
               // const int ipt = i / _nXnDim;
               // const int imod = (i % _nXnDim);
               // const int iX  = imod / _nDim;
               // const int jX  = imod % _nDim;
               // cJ[i] = J(ipt, iX, jX);
               // Alternatively, we can just use the underlying data here.
               cJ[i] = J.get_data()[i];
            });
            // Dummy variable since we got rid of using raw pointers
            // The default initialization has a size of 0 which is what we want.
            rview3d J_dummy(nullptr, 0, 0, 0);
            chai::ManagedArray<double> cr_pert = mm.allocManagedArray<double>(_initial_batch_size * _nDim);
            chai::ManagedArray<double> cx_pert = mm.allocManagedArray<double>(_npts * _nDim);
            const auto es = snls::Device::GetInstance().GetCHAIES();
            rview2d r_pert(SNLS_RSETUP(cr_pert, es, 0), _initial_batch_size, _nDim);
            rview2d x_pert(SNLS_RSETUP(cx_pert, es, 0), _npts, _nDim);
            rview3d J_FD(SNLS_RSETUP(cJ_FD, es, 0), _initial_batch_size, _nDim, _nDim);

            const double pert_val     = 1.0e-7 ;
            const double pert_val_inv = 1.0/pert_val;

            for ( int iX = 0; iX < _nDim ; ++iX ) {
               snls::forall<SNLS_GPU_BLOCKS>(0, batch_size, [=] __snls_hdev__ (int iBatch) {
                  for ( int jX = 0; jX < _nDim ; ++jX ) {
                     x_pert(offset + iBatch, jX) = x(offset + iBatch, jX);
                  }
                  x_pert(offset + iBatch, iX) = x_pert(offset + iBatch, iX) + pert_val;
               });
               this->_crj.computeRJ(r_pert, J_dummy, x_pert, rJSuccess, status, offset, batch_size);
               const bool success = reduced_rjSuccess(rJSuccess, batch_size);
               if ( !success ) {
                  SNLS_FAIL(__func__, "Problem while finite-differencing");
               }
               snls::forall<SNLS_GPU_BLOCKS>(0, batch_size, [=] __snls_hdev__ (int iBatch) {
                  for ( int iR = 0; iR < _nDim ; iR++ ) {
                     J_FD(iBatch, iR, iX) = pert_val_inv * ( r_pert(iBatch, iR) - r(iBatch, iR) );
                  }
               });
            }

            // Terribly inefficient here if running on the GPU...
            const double* J_FD_data = cJ_FD.data(chai::ExecutionSpace::CPU);
            const double* J_data = cJ.data(chai::ExecutionSpace::CPU);
#if defined(__snls_host_only__)
            for (int iBatch = 0; iBatch < batch_size; iBatch++) {
               const int moff = SNLS_MOFF(iBatch, _nXnDim);
               *_os << "Batch item # " << iBatch << std::endl;
               *_os << "J_an = " << std::endl ; snls::linalg::printMat<_nDim>( &J_data[moff],    *_os ) ;
               *_os << "J_fd = " << std::endl ; snls::linalg::printMat<_nDim>( &J_FD_data[moff], *_os ) ;
            }
#endif

            // Clean-up the memory these objects use.
            cx_pert.free();
            cr_pert.free();
            cJ_FD.free();
            cJ.free();

            // put things back the way they were ;
            this->_crj.computeRJ(r, J, x, rJSuccess, status, offset, batch_size);
         } // _os != nullptr
#endif
      }
   private :
     __snls_hdev__ inline bool  computeNewtonStep (double* const       J,
                                      const double* const r,
                                      double* const       newton  ) {
#if HAVE_LAPACK && SNLS_USE_LAPACK && defined(__snls_host_only__)
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

         if ( info != 0 ) {
            SNLS_WARN(__func__, "info non-zero from dgetrf");
            return false;
         }

         for (int iX = 0; iX < _nDim; ++iX) {
            newton[iX] = - r[iX] ; 
         }

         int nRHS=1; info=0;
         DGETRS(&trans, &_nDim, &nRHS, J, &_nDim, ipiv, newton, &_nDim, &info);

         if ( info != 0 ) {
            SNLS_WARN(__func__, "info non-zero from lapack::dgetrs()");
            return false;
         }

#else
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__snls_host_only__)

         {
            const int n = _nDim;

            int   err = SNLS_LUP_Solve<n>(J, newton, r);
            if (err<0) {
               SNLS_WARN(__func__," fail return from LUP_Solve()");
               return false;
            }
            for (int i=0; (i<n); ++i) { newton[i] = -newton[i]; }
         }
#endif
         return true;
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
      }
   public:

      /// Performs a bitwise reduction and operation on _status to see if the
      /// current batch can exit.
      template <const bool batch_loop, const int NUMBLOCKS=512>
      inline bool status_exit(const int offset,
                              const int batch_size)
      {
         bool red_add = false;
         const int end = offset + batch_size;
         SNLSStatus_t*  status = _status.data(Device::GetInstance().GetCHAIES());
         switch(Device::GetInstance().GetBackend()) {
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
            case(ExecutionStrategy::GPU): {
               //RAJA::ReduceBitAnd<RAJA::cuda_reduce, bool> output(init_val);
#if defined(RAJA_ENABLE_CUDA)
               using gpu_reduce = RAJA::cuda_reduce;
               using gpu_policy = RAJA::cuda_exec<NUMBLOCKS>;
#else
               using gpu_reduce = RAJA::hip_reduce;
               using gpu_policy = RAJA::hip_exec<NUMBLOCKS>;
#endif
               RAJA::ReduceSum<gpu_reduce, int> output(0);
               RAJA::forall<gpu_policy>(RAJA::RangeSegment(offset, end), [=] __snls_device__ (int i) {
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
         return red_add;
      } // end of status_exitable

      /// Performs a bitwise reduction and operation on _status to see if the
      /// FD portion of computeRJ was successful
      template <const int NUMBLOCKS=512>
      inline bool reduced_rjSuccess(rview1b &rJSuccess,
                                    const int batch_size)
      {
         bool red_add = false;
         const int end = batch_size;
         switch(Device::GetInstance().GetBackend()) {
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
            case(ExecutionStrategy::GPU): {
               //RAJA::ReduceBitAnd<RAJA::cuda_reduce, bool> output(init_val);
#if defined(RAJA_ENABLE_CUDA)
               using gpu_reduce = RAJA::cuda_reduce;
               using gpu_policy = RAJA::cuda_exec<NUMBLOCKS>;
#else
               using gpu_reduce = RAJA::hip_reduce;
               using gpu_policy = RAJA::hip_exec<NUMBLOCKS>;
#endif
               RAJA::ReduceSum<gpu_reduce, int> output(0);
               RAJA::forall<gpu_policy>(RAJA::RangeSegment(0, end), [=] __snls_device__ (int i) {
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
         return red_add;
      } // end of status_exitable

      // Returns a pointer to the status array now on the host.
      // The user is responsible for deleting this ptr after their done using it
      // through the use of the SNLS::memoryManager::dealloc<T>(T* ptr) function  
      SNLSStatus_t * getStatusHost() {
         return _status.data(chai::ExecutionSpace::CPU);
      }

   public:
      rview2d _x;

   protected:
      static constexpr int _nXnDim = _nDim * _nDim ;
      
      int _mfevals, _nIters;
      chai::ManagedArray<int> _fevals;
      rview1d _delta;
      rview1d _res;
      rview2d _residual;
      rview3d _Jacobian;
      rview1b _rjSuccess;
      chai::ManagedArray<bool>   wrkb_data;
      chai::ManagedArray<double> wrk_data;

   private:
      chai::managed_ptr<TrDeltaControl> _deltaControl ;

      int    _offset_work;
      int    _maxIter ;
      double _tolerance ;
      int    _outputLevel ;
      uint   _initial_batch_size ;
      uint   _npts ;

      std::ostream* _os ;

      chai::ManagedArray<SNLSStatus_t> _status;
};
} // namespace batch
} // namespace snls

#endif //SNLS_RAJA_PORT_SUITE