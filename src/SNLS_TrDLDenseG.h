// -*-c++-*-

#ifndef SNLS_TRDLDG_H
#define SNLS_TRDLDG_H

#include "SNLS_base.h"
#include "SNLS_linalg.h"
#include "SNLS_lup_solve.h"
#include "SNLS_TrDelta.h"
#include "SNLS_kernels.h"

#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)
#include "RAJA/RAJA.hpp"
#include "SNLS_device_forall.h"
#endif

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

namespace snls {

// trust region type solver, dogleg approximation
// for dense general Jacobian matrix
//
// CRJ should :
// 	have member function
// 		     __snls_hdev__ bool computeRJ( double* const r, double* const J, const double* const x ) ;
// 		computeRJ function returns true for successful evaluation
// 		TODO ... J becomes a RAJA::View ?
//	have trait nDimSys
//
// TODO ...*** specialize to N=1 case, and N=2 also?
//
template< typename CRJ, int nDimSys = CRJ::nDimSys >
class SNLSTrDlDenseG 
{
   public:
      static_assert(has_valid_computeRJ<CRJ>::value || has_valid_computeRJ_lambda<CRJ>::value, "The CRJ implementation in SNLSTrDlDenseG needs to implement bool computeRJ( double* const r, double* const J, const double* const x ) or be a lambda function that takes in the same arguments");
      // static_assert(has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");

   public:
   // constructor
   __snls_hdev__ SNLSTrDlDenseG(CRJ &crj) :
               _crj(crj),
               _fevals(0), _nIters(0), _nJFact(0), _delta(1e8), _res(1e20),
               _deltaControl(nullptr),
               _outputLevel(0),
               _rhoLast(0.0),
               _os(nullptr),
               _status(unConverged)
               {
               }
   // destructor
   __snls_hdev__ ~SNLSTrDlDenseG() {
#ifdef __snls_host_only__
      if ( _outputLevel > 1 && _os != nullptr ) {
         *_os << "Function and Jacobian factorizations: " << _fevals << " " << _nJFact << std::endl;
      }
#endif
   }

   public:
      CRJ &_crj ;
      static constexpr int _nDim = nDimSys;

      __snls_hdev__ int     getNDim   () const { return(_nDim   ); }
      __snls_hdev__ int     getNFEvals() const { return(_fevals ); }
      __snls_hdev__ int     getNJEvals() const { return(_fevals ); }
      __snls_hdev__ double  getRhoLast() const { return(_rhoLast); }
      __snls_hdev__ double  getDelta  () const { return(_delta  ); }
      __snls_hdev__ double  getRes    () const { return(_res    ); }

      // setX can be used to set the initial guess
      __snls_hdev__ inline void setX( const double* const x ) {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x[iX] = x[iX] ;
         }
      }
   
      __snls_hdev__ inline void getX( double* const x ) const {
         for (int iX = 0; iX < _nDim; ++iX) {
            x[iX] = _x[iX] ;
         }
      } 

      /**
       * Must call setupSolver before calling solve
       */
      __snls_hdev__ void   setupSolver(int              maxIter,
                                       double           tolerance,
                                       TrDeltaControl * deltaControl,
                                       int              outputLevel=0 ) {
   
         _status = unConverged ;
         _fevals = 0 ;

         _maxIter = maxIter ;
         _tolerance = tolerance ;

         _deltaControl = deltaControl ;

         this->setOutputlevel( outputLevel ) ;

      }

      __snls_hdev__ void   setOutputlevel( int    outputLevel ) {
         _outputLevel = outputLevel ;
         _os          = nullptr ;
         //
#ifdef __snls_host_only__
         if ( _outputLevel > 0 ) {
            _os = &(std::cout) ;
         }
#endif
      }

#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)
      /**
       * @brief Solve for the current CRJ/_x, single-threaded -- byte-identical
       * behavior to this method before cooperative support existed.
       *
       * A one-line forwarder onto the same solveImpl() that solveTeam()
       * below uses, specialized to a "team of one": tid=0, nthreads=1, a
       * trivial (and, per snls::TeamActivityConsensus's own contract,
       * zero-cost) TeamActivityConsensus<1>, and no external scratch. See
       * solveImpl()'s doc comment for why this degeneration is exact, not
       * approximate.
       *
       * @return the solver's exit status, as before.
       */
      __snls_hdev__ SNLSStatus_t solve() {
         snls::TeamActivityConsensus<1> soloConsensus;
         return solveImpl(0, 1, 0, soloConsensus, nullptr, nullptr);
      }

      /**
       * @brief Solve for the current CRJ/_x, using a team of `nthreads`
       * cooperating threads instead of one thread alone.
       *
       * The caller (typically an external application's own
       * snls::forall_team-based kernel, or an equivalent mechanism it
       * builds itself, since SNLS does not own the outer kernel for this
       * solver -- see design doc §3.3) must ensure:
       *  -# All `nthreads` threads calling solveTeam() together for one
       *     point reference the SAME SNLSTrDlDenseG instance (e.g.
       *     constructed once in RAJA_TEAM_SHARED memory, gated by
       *     tid==0, with a teamSync() before any thread calls
       *     solveTeam()) -- not one instance per thread. This is what
       *     makes _x/_status (and every other member this method reads or
       *     writes) genuinely shared across the team, exactly the way
       *     `active`/`i`-indexed state is naturally shared in
       *     SNLSTrDlDenseG_Batch.
       *  -# `tid`, `nthreads`, and `teamBase` are exactly what
       *     snls::forall_team (or the caller's equivalent) handed this
       *     thread for this point.
       *  -# `extScratch`/`extPiv`, if provided, are pointers this caller
       *     has already resolved to its own teamBase-indexed slot of a
       *     larger, block-wide buffer (mirroring the caller-provided-
       *     scratch pattern in SNLS_lup_solve.h's cooperative overloads --
       *     RAJA_TEAM_SHARED memory is block-scoped, not team-scoped, so
       *     a buffer this method declared internally could not safely be
       *     indexed per-team on its own). If null (the default), this
       *     method declares its own internal team-shared scratch instead
       *     -- safe ONLY if at most one team occupies the physical block
       *     this call executes in (always true for solve()'s nthreads==1
       *     use; true for a solveTeam() use only if the caller's own
       *     packing guarantees itemsPerBlock==1, e.g. NTEAM==NUMBLOCKS
       *     exactly). A caller packing multiple teams per block MUST
       *     supply both.
       *
       * @tparam Consensus deduced; the concrete snls::TeamActivityConsensus<N>
       *                   specialization the caller's own dispatch
       *                   constructed, sized to however many teams are
       *                   actually co-resident in that dispatch.
       *
       * @param[in] tid        this thread's role within its own team,
       *                       0..nthreads-1.
       * @param[in] nthreads   how many threads are cooperating on this
       *                       solve.
       * @param[in] teamBase   which team-slot this thread's team occupies
       *                       within the physical block, as handed to the
       *                       body by snls::forall_team -- only consulted
       *                       when extScratch/extPiv are null and more
       *                       than one team could be co-resident (see the
       *                       precondition above).
       * @param[in,out] consensus block-wide activity consensus (see
       *                       snls::TeamActivityConsensus), used to decide
       *                       when every team sharing this block is done
       *                       rather than exiting on a purely
       *                       per-point-local condition.
       * @param[in,out] extScratch optional, caller-owned team-shared
       *                       scratch of at least _nXnDim + _nDim
       *                       doubles (Jacobian then residual). See the
       *                       precondition above.
       * @param[in,out] extPiv     optional, caller-owned team-shared
       *                       scratch of at least _nDim + 1 ints.
       *
       * @return the solver's exit status.
       */
      template <typename Consensus>
      __snls_hdev__ SNLSStatus_t solveTeam(int tid, int nthreads, int teamBase,
                                            Consensus& consensus,
                                            double* extScratch = nullptr,
                                            int* extPiv = nullptr) {
         return solveImpl(tid, nthreads, teamBase, consensus, extScratch, extPiv);
      }
#else
      // solve returns status
      //
      // on exit, _res is consistent with _x
      __snls_hdev__ SNLSStatus_t solve() {

         if ( _deltaControl == nullptr ) {
            SNLS_FAIL("solve", "_deltaControl not set") ;
         }

         _status = unConverged ;
         _fevals = 0 ;
         _nJFact = 0 ;
         _nIters = 0 ;

         _delta = _deltaControl->getDeltaInit() ;
#ifdef __snls_host_only__
         if (_os) { *_os << "initial delta = " << _delta << std::endl ; }
#endif

         double residual[_nDim], Jacobian[_nXnDim] ;
         //
         {
            bool rjSuccess = this->computeRJ(residual, Jacobian) ; // at _x
            if ( !(rjSuccess) ) {
               _status = initEvalFailure ;
               return _status ;
            }
         }
         _res = snls::linalg::norm<_nDim>(residual);
         double res_0 = _res ;
#ifdef __snls_host_only__
         if (_os) { *_os << "res = " << _res << std::endl ; }
#endif

         bool reject_prev = false ;
         //
         // data for sorting out the step
         double nrStep[_nDim];
         double grad[_nDim];
         double delx[_nDim];
         double Jg_2;

         //
         while ( _nIters < _maxIter ) {
            //
            _nIters += 1 ;

            // This is done outside this step so that these operations can be done with varying solve
            // techniques such as LU/QR or etc...
            if(!reject_prev) {
               // So the LU solve does things in-place which causes issues when calculating the grad term...
               // So, we need to pull this out and perform this operation first
               snls::linalg::matTVecMult<_nDim, _nDim>(Jacobian, residual, grad);
               {
                  double ntemp[_nDim] ;
                  snls::linalg::matVecMult<_nDim, _nDim>(Jacobian, grad, ntemp);
                  Jg_2 = snls::linalg::dotProd<_nDim>(ntemp, ntemp);
               }
               const bool sol_stat = this->computeNewtonStep( Jacobian, residual, nrStep );
               if (!sol_stat) {
                  _status = SNLSStatus_t::linearSolveFailure;
                  break;
               }

            }
            //
            double pred_resid;
            bool use_nr = false;

            // If the step was rejected nrStep will be the same value as previously, and so we can just recalculate nr_norm here.
            const double nr_norm = snls::linalg::norm<_nDim>(nrStep);

            // computes the updated delta x, predicated residual error, and whether or not NR method was used.
            snls::dogleg<_nDim>(_delta, res_0, nr_norm, Jg_2, grad, nrStep,
                                delx, _x, pred_resid, use_nr, _os);
            reject_prev = false;

            //
            {
               bool rjSuccess = this->computeRJ(residual, Jacobian) ; // at _x
               snls::updateDelta<_nDim>(_deltaControl, residual, res_0, pred_resid, nr_norm, _tolerance, use_nr, rjSuccess,
                                        _delta, _res, _rhoLast, reject_prev, _status, _os);
               if(_status != SNLSStatus_t::unConverged) { break; }
            }

            if ( reject_prev ) {
#ifdef __snls_host_only__
               if ( _os != nullptr ) {
                  *_os << "rejecting solution" << std::endl ;
               }
#endif
               _res = res_0 ;
               this->reject( delx ) ;
            }
            //
            res_0 = _res;

         } // _nIters < _maxIter

         return _status ;

      }
#endif // SNLS_RAJA_PORT_SUITE || SNLS_RAJA_ONLY

      // convenience wrapper, for the current _x
      __snls_hdev__ inline bool computeRJ(double* const r,
                                          double* const J ) {
         
         _fevals++ ;
         bool retval;
         if constexpr(has_valid_computeRJ<CRJ>::value) {
            retval = this->_crj.computeRJ(r, J, _x);
         } else {
            retval = this->_crj(r, J, _x);
         }
         
#ifdef SNLS_DEBUG
#ifdef __snls_host_only__
         if ( _outputLevel > 2 && _os != nullptr ) {
            // do finite differencing
            // assume system is scaled such that perturbation size can be standard

            double r_base[_nDim]; 
            for ( int jX = 0; jX < _nDim ; ++jX ) {
               r_base[jX] = r[jX] ;
            }
            
            const double pert_val     = 1.0e-7 ;
            const double pert_val_inv = 1.0/pert_val ;
            
            double J_FD[_nXnDim] ;
            
            for ( int iX = 0; iX < _nDim ; ++iX ) {
               double r_pert[_nDim];
               double x_pert[_nDim];
               for ( int jX = 0; jX < _nDim ; ++jX ) {
                  x_pert[jX] = _x[jX] ;
               }
               x_pert[iX] = x_pert[iX] + pert_val ;
               bool retvalThis;
               if constexpr(has_valid_computeRJ<CRJ>::value) {
                  retvalThis = this->_crj.computeRJ(r_pert, nullptr, x_pert);
               } else {
                  retvalThis = this->_crj(r_pert, nullptr, x_pert);
               }
               if ( !retvalThis ) {
                  SNLS_FAIL(__func__, "Problem while finite-differencing");
               }
               for ( int iR = 0; iR < _nDim ; iR++ ) {
                  J_FD[SNLS_NN_INDX(iR,iX,_nDim)] = pert_val_inv * ( r_pert[iR] - r_base[iR] ) ;
               }
            }
            
            *_os << "J_an = " << std::endl ; snls::linalg::printMat<_nDim>( J,    *_os ) ;
            *_os << "J_fd = " << std::endl ; snls::linalg::printMat<_nDim>( J_FD, *_os ) ;

            // put things back the way they were ;
            if constexpr(has_valid_computeRJ<CRJ>::value) {
               retval = this->_crj.computeRJ(r, J, _x);
            } else {
               retval = this->_crj(r, J, _x);
            }
            
         } // _os != nullptr
#endif
#endif         
         return retval ;
         
      }
      
   private :

#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)
      /**
       * @brief One unified Newton-loop implementation behind both solve()
       * (tid=0, nthreads=1, a trivial consensus) and solveTeam() (real
       * cooperation) -- there is exactly one loop, not a simple one and a
       * separately-maintained cooperative one (design doc §4.5).
       *
       * **Why the nthreads==1 case is exactly, not just approximately,
       * today's behavior**: snls::TeamActivityConsensus<1>::report()/
       * anyActive() degenerate to exactly `myActive` (here: exactly
       * `_status==unConverged`) at zero extra cost, and
       * RAJA::LaunchContext::teamSync() is a no-op whenever nthreads==1
       * (host, or a lone device team of one) -- so this loop, specialized
       * to that case, compiles down to the same sequence of operations as
       * the original solve() loop. This is verified by an explicit
       * bit-identical regression test rather than merely asserted.
       *
       * **What is, and is not, team-shared, and why**: the elimination
       * step inside the cooperative LU solve genuinely needs every
       * thread to see the same Jacobian/residual/pivot memory, so those
       * become team-shared scratch (extScratch/extPiv, or self-managed
       * under the single-team-per-block precondition documented on
       * solveTeam()). `_status` and `_reject_prev` are also read to gate
       * whether the cooperative, teamSync()-containing computeNewtonStep()
       * call happens at all this iteration -- they must be consistently
       * visible to every thread in the team too, which is why
       * `_reject_prev` is a class member (naturally shared once the
       * whole object is, per solveTeam()'s precondition) rather than a
       * local, and why the loop re-derives `active` from `_status` fresh
       * every pass instead of maintaining a separate per-thread "am I
       * still active" local: only `tid==0` ever assigns such a local, so
       * every other thread's own copy would never change, and would keep
       * entering this teamSync()-containing block on later passes even
       * after `tid==0` (having already updated `_status`) stopped --
       * exactly the divergent-barrier-count hazard this whole design
       * exists to avoid, just one level deeper than the batch solver's
       * `active` flag needs to worry about, since here it also gates a
       * cooperative call across the SAME team, not just across a kernel's
       * different points. Re-deriving `active` from the team-shared
       * `_status` every pass sidesteps this entirely. Everything else here
       * (nrStep, grad, delx, Jg_2, res_0,
       * nr_norm, pred_resid, use_nr) is read and written by `tid==0`
       * only, throughout, so it stays an ordinary per-thread-private
       * local exactly as it was before -- there is nothing for another
       * thread to ever see.
       *
       * @tparam Consensus deduced; see solveTeam()'s doc comment.
       *
       * @param[in]     tid        see solveTeam().
       * @param[in]     nthreads   see solveTeam().
       * @param[in]     teamBase   see solveTeam().
       * @param[in,out] consensus  see solveTeam().
       * @param[in,out] extScratch see solveTeam().
       * @param[in,out] extPiv     see solveTeam().
       *
       * @return the solver's exit status.
       */
      template <typename Consensus>
      __snls_hdev__ SNLSStatus_t solveImpl(int tid, int nthreads, int teamBase,
                                            Consensus& consensus,
                                            double* extScratch, int* extPiv) {

         // Team-shared scratch for the Jacobian/residual/pivot state the
         // cooperative LU solve needs every thread in the team to see --
         // see solveTeam()'s doc comment for the extScratch/extPiv
         // precondition when they are null.
         RAJA_TEAM_SHARED double localScratch[_nXnDim + _nDim];
         RAJA_TEAM_SHARED int    localPiv[_nDim + 1];
         double* scratch  = extScratch ? extScratch : localScratch;
         int*    piv      = extPiv     ? extPiv     : localPiv;
         double* Jacobian = scratch;
         double* residual = scratch + _nXnDim;
         (void)teamBase; // only meaningful to the self-managed (null) scratch case's precondition, not to any indexing done here

         if (tid == 0) {
            if ( _deltaControl == nullptr ) {
               SNLS_FAIL("solve", "_deltaControl not set") ;
            }
            _status = unConverged ;
            _fevals = 0 ;
            _nJFact = 0 ;
            _nIters = 0 ;
            _reject_prev = false ;

            _delta = _deltaControl->getDeltaInit() ;
#ifdef __snls_host_only__
            if (_os) { *_os << "initial delta = " << _delta << std::endl ; }
#endif
            bool rjSuccess = this->computeRJ(residual, Jacobian) ; // at _x
            if ( !(rjSuccess) ) {
               _status = initEvalFailure ;
            } else {
               _res = snls::linalg::norm<_nDim>(residual);
#ifdef __snls_host_only__
               if (_os) { *_os << "res = " << _res << std::endl ; }
#endif
            }
         }
         RAJA::LaunchContext{}.teamSync(); // publish _status/_res/residual/Jacobian

         if ( _status == initEvalFailure ) {
            // Uniform across every thread in the team (every thread just
            // read the same, just-published _status) -- safe to return
            // here, before any teamSync()-containing loop work has begun.
            return _status ;
         }

         // tid==0-only state from here down: read and written by tid==0
         // alone throughout, so an ordinary per-thread-private local is
         // exactly as safe as it was before cooperation existed.
         double res_0 = _res ;
         double nrStep[_nDim];
         double grad[_nDim];
         double delx[_nDim];
         double Jg_2 = 0.0;

         while ( true ) {
            // Re-derived fresh every pass from the team-shared _status
            // and _nIters, rather than threaded through as a separate
            // per-thread local -- see this method's doc comment for why.
            // The _nIters < _maxIter bound matches the original solve()
            // loop's `while (_nIters < _maxIter)` -- without it, a
            // problem that never converges or fails outright (only
            // possible with a poorly-conditioned CRJ/starting guess) would
            // spin here forever instead of giving up as it does today.
            const bool active = ( _status == SNLSStatus_t::unConverged ) && ( _nIters < _maxIter );
            consensus.report(tid, teamBase, active);
            if ( !consensus.anyActive() ) { break; }

            if ( active ) {
               if (tid == 0) { _nIters += 1; }

               if (!_reject_prev) {
                  if (tid == 0) {
                     // So the LU solve does things in-place which causes issues when calculating the grad term...
                     // So, we need to pull this out and perform this operation first
                     snls::linalg::matTVecMult<_nDim, _nDim>(Jacobian, residual, grad);
                     double ntemp[_nDim] ;
                     snls::linalg::matVecMult<_nDim, _nDim>(Jacobian, grad, ntemp);
                     Jg_2 = snls::linalg::dotProd<_nDim>(ntemp, ntemp);
                  }
               }
               RAJA::LaunchContext{}.teamSync(); // before the cooperative call, all threads unconditionally

               // computeNewtonStep runs unconditionally either way (calling
               // it a data-dependent number of times per thread, e.g.
               // skipping it entirely when _reject_prev is true, would
               // make a DIFFERENT team sharing this physical block --
               // which has no reason to share this team's _reject_prev
               // value, since it is a separate point with its own
               // SNLSTrDlDenseG instance -- diverge on how many times it
               // calls this teamSync()-containing function). But when
               // _reject_prev is true, nrStep must stay exactly as it
               // was: Jacobian/residual at this point are the REJECTED
               // trial's values, not the (reverted) current _x's, so a
               // fresh Newton step computed from them would be
               // meaningless, not merely redundant -- see this method's
               // doc comment. So its output goes to a throwaway
               // per-thread scratch buffer instead of the persistent
               // nrStep whenever this point is not actually recomputing.
               double newtonScratch[_nDim];
               double* newtonDest = _reject_prev ? newtonScratch : nrStep;

               const bool sol_stat = this->computeNewtonStep(tid, nthreads, piv, Jacobian, residual, newtonDest);
               RAJA::LaunchContext{}.teamSync(); // after

               if (tid == 0) {
                  if (!sol_stat) {
                     _status = SNLSStatus_t::linearSolveFailure;
                  } else {
                     double pred_resid;
                     bool use_nr = false;

                     // If the step was rejected nrStep will be the same value as previously, and so we can just recalculate nr_norm here.
                     const double nr_norm = snls::linalg::norm<_nDim>(nrStep);

                     // computes the updated delta x, predicated residual error, and whether or not NR method was used.
                     snls::dogleg<_nDim>(_delta, res_0, nr_norm, Jg_2, grad, nrStep,
                                         delx, _x, pred_resid, use_nr, _os);
                     _reject_prev = false;

                     bool rjSuccess = this->computeRJ(residual, Jacobian) ; // at _x
                     snls::updateDelta<_nDim>(_deltaControl, residual, res_0, pred_resid, nr_norm, _tolerance, use_nr, rjSuccess,
                                              _delta, _res, _rhoLast, _reject_prev, _status, _os);

                     if ( _reject_prev ) {
#ifdef __snls_host_only__
                        if ( _os != nullptr ) {
                           *_os << "rejecting solution" << std::endl ;
                        }
#endif
                        _res = res_0 ;
                        this->reject( delx ) ;
                     }
                     res_0 = _res;
                  }
               }
            }
            RAJA::LaunchContext{}.teamSync(); // so every thread's next-pass reads of _status/_reject_prev are consistent
         } // consensus loop

         return _status ;
      }

      /**
       * @brief Compute the Newton step via a team of `nthreads`
       * cooperating threads -- the cooperative counterpart of the plain
       * computeNewtonStep() below. See SNLSTrDlDenseG_Batch's
       * computeNewtonStep for the identical shape and reasoning; this is
       * the non-batch solver's version of the same refactor.
       *
       * @param[in]     tid      this thread's role within its own team.
       * @param[in]     nthreads how many threads are cooperating.
       * @param[in,out] piv      n+1, team-shared pivot-vector scratch
       *                         (caller-provided; see SNLS_lup_solve.h's
       *                         cooperative SNLS_LUP_Solve() for why).
       * @param[in,out] J        n*n, row-major, team-shared scratch.
       * @param[in]     r        n, the residual vector.
       * @param[out]    newton   n, the computed Newton step.
       *
       * @return true on success, false if the LU solve failed.
       */
      __snls_hdev__ inline bool  computeNewtonStep (int tid, int nthreads,
                                                    int* const          piv,
                                                    double* const       J,
                                                    const double* const r,
                                                    double* const       newton  ) {

         _nJFact++ ;

#if HAVE_LAPACK && SNLS_USE_LAPACK && defined(__snls_host_only__)
         // UNTOUCHED -- host-only, never reached with nthreads>1 in
         // practice since the GPU packed path never compiles
         // __snls_host_only__ code; tid/nthreads/piv are unused here.
         // row-major storage
         const char trans = 'T';

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

         {
            const int n = _nDim;
            constexpr double tol = 1e-50; // matches the plain overload's default

            int   err = SNLS_LUP_Solve<n>(J, piv, newton, r, tid, nthreads, tol);
            if (err<0) {
               SNLS_WARN(__func__," fail return from LUP_Solve()");
               return false;
            }
            for (int i=0; (i<n); ++i) { newton[i] = -newton[i]; }
         }
#endif
         return true;
      }
#else
      __snls_hdev__ inline bool  computeNewtonStep (double* const       J,
                                                    const double* const r,
                                                    double* const       newton  ) {

         _nJFact++ ;

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

         // std::copy( r, r + _nDim, newton );
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
            //
            for (int i=0; (i<n); ++i) { newton[i] = -newton[i]; }

         }
#endif
// HAVE_LAPACK && SNLS_USE_LAPACK && defined(__cuda_host_only__)
         return true;
      }
#endif // SNLS_RAJA_PORT_SUITE || SNLS_RAJA_ONLY

      __snls_hdev__ inline void  reject(const double* const delX ) {
         for (int iX = 0; iX < _nDim; ++iX) {
            _x[iX] = _x[iX] - delX[iX] ;
         }
      }

   public:
      double _x[_nDim] ;

   protected:
      static constexpr int _nXnDim = _nDim * _nDim ;

      int _fevals, _nIters, _nJFact ;
      double _delta, _res ;
#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)
      /// Whether the previous step was rejected -- a class member (rather
      /// than solveImpl()'s own local, as it was before cooperative
      /// support existed) specifically so it is naturally team-shared
      /// once the whole object is (see solveTeam()'s precondition):
      /// it gates whether the cooperative computeNewtonStep() call
      /// happens at all, so every thread in the team must see the same
      /// value. Reset at the start of every solveImpl() call.
      bool _reject_prev ;
#endif

   private:
      TrDeltaControl* _deltaControl ;

      int   _maxIter    ;
      double _tolerance  ;
      int   _outputLevel;

      // _rhoLast is not really needed -- but is kept for debug and testing purposes
      double _rhoLast ;

#ifdef __snls_host_only__
      std::ostream* _os ;
#else
      char* _os ; // do not use
#endif

      SNLSStatus_t  _status ;
};

} // namespace snls

#endif  // SNLS_TRDLDG_H
