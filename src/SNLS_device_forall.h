/**************************************************************************
 Module:  MS_device_forall
 Purpose: Provides an abstraction layer over various different execution
          backends to allow for an easy way to write code for either the
          CPU, OpenMP, or the GPU using a single piece of code.
 ***************************************************************************/

#ifndef SNLS_device_forall_h
#define SNLS_device_forall_h

#ifndef SNLS_GPU_BLOCKS
#define SNLS_GPU_BLOCKS 256
#endif

#include "SNLS_config.h"

#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)

#include "SNLS_gpu_portability.h"
#include "SNLS_unused.h"
#include "RAJA/RAJA.hpp"

#if defined(SNLS_RAJA_PORT_SUITE)

#include "chai/config.hpp"
#include "chai/ExecutionSpaces.hpp"

#endif

#include <variant>

   // Implementation of SNLS's "parallel for" (forall) device/host kernel
   // interfaces supporting RAJA and sequential backends which is based on
   // the MFEM_FORALL macro.
   // An example would be something like:
   // SNLS_FORALL(i, 0, 50, {var[i] = 0;});
   
/// The SNLS_FORALL wrapper where GPU blocks are set to a default value
#define SNLS_FORALL(i, st, end, ...)                  \
snls::SNLS_ForallWrap<SNLS_GPU_BLOCKS>(		         \
st,                                                   \
end,                                                  \
snls::Device::GetInstance().GetDefaultRAJAResource(), \
[=] __snls_device__ (int i) {__VA_ARGS__},            \
[=] (int i) {__VA_ARGS__})

/// The SNLS_FORALL wrapper that allows one to change the number of GPU blocks
#define SNLS_FORALL_T(i, threads, st, end, ...)       \
snls::SNLS_ForallWrap<threads>(			               \
st,                                                   \
end,                                                  \
snls::Device::GetInstance().GetDefaultRAJAResource(), \
[=] __snls_device__ (int i) {__VA_ARGS__},            \
[=] (int i) {__VA_ARGS__})

// offset a vector / matrix to correct starting memory location given:
// what local element we're on, what offset that local element is from the global 0,
// and then the unrolled size of the vector / matrix
#define SNLS_TOFF(ielem, offset, ndim) (offset * ndim) + (ielem * ndim)
#define SNLS_VOFF(ielem, ndim) (ielem * ndim)
#define SNLS_MOFF(ielem, ndim2) (ielem * ndim2)

#if defined(SNLS_RAJA_PORT_SUITE)
using ches = chai::ExecutionSpace;
#else
namespace chai_fake {
   enum class ExecutionSpace {CPU, GPU};
}
using ches = chai_fake::ExecutionSpace;
#endif

namespace snls {

   using rhost_res = RAJA::resources::Host;
#if defined(__snls_gpu_active__)
#if defined(RAJA_ENABLE_CUDA)
         using rgpu_res = RAJA::resources::Cuda;
#else //defined(RAJA_ENABLE_HIP)
         using rgpu_res = RAJA::resources::Hip;
#endif
#endif

   using rres = std::variant<
      rhost_res
#if defined(__snls_gpu_active__)
      ,
      rgpu_res
#endif
   >;

   using rrese = RAJA::resources::Event;

   /// ExecutionStrategy defines how one would like the
   /// computations done.
   /// CPU refers to serial executions of for loops on the Host
   /// through RAJA forall abstractions
   /// OPENMP refers to parallel executuons of for loops on the 
   /// Host using OpenMP through RAJA forall abstractions
   /// GPU refers to parallel executions of for loops on the Device
   /// using GPU through RAJA forall abstractions
   enum class ExecutionStrategy { CPU, GPU, OPENMP };

   /// This has largely been inspired by the MFEM device
   /// class, since they make use of it with their FORALL macro
   /// It's recommended to only have one object for the lifetime
   /// of multiple models being used, so no clashing with
   /// multiple objects can occur in regards to which models
   /// run on what ExecutionStrategy backend.
   class Device {
      public:
         static Device& GetInstance();

         ///
         /// Get the current execution strategy
         ///
         /// @return   the current execution strategy
         ///
         ExecutionStrategy GetBackend() { return m_es; }

         ///
         /// Set the current execution strategy
         ///
         /// @param[in]   es   New execution strategy
         ///
         void SetBackend(ExecutionStrategy es) { m_es = es; }

         ///
         /// Get CHAI execution space corresponding to the execution strategy
         ///
         /// @return   the current CHAI execution space
         ///
         ches GetCHAIES();

         /// Return the default RAJA resource corresponding to the execution strategy
         /// @return a RAJA resource set
         ///
         rres GetDefaultRAJAResource();

         /// Return a new RAJA resource corresponding to the execution strategy
         /// Note: When the GPU execution strategy is set, we do not return the default
         ///       resource set / stream but instead cycle through the 15 other streams that
         ///       RAJA / camp has created ahead of time.
         /// @return a RAJA resource set
         ///
         rres GetRAJAResource();

         /// Utilizing the provided RAJA resource variant, it waits for the RAJA resource event
         /// to be completed. Note, a few safety constraints apply here:
         /// 1.) You must not change the execution space to a different one until all the kernels
         ///     have finished running.
         /// 2.) The event supplied to this wait for must correspond to a forallwrap that utilized the
         ///     same RAJA resource being supplied here.
         ///     Note: If one did not supply a resource set to the one of the forall calls then it is likely
         ///     using the default value.
         void WaitFor(rres& res, rrese* event);

         ///
         /// Wait for all work enqueued in resource to complete
         ///
         void Wait(rres& res);

         ///
         /// Delete copy constructor
         ///
         Device(const Device&) = delete;

         ///
         /// Delete copy assignment operator
         ///
         Device& operator=(const Device&) = delete;

      private:
         ///
         /// Current execution strategy
         ///
         ExecutionStrategy m_es;

         /// Default host resource set
         rhost_res m_host_res;
#if defined(__snls_gpu_active__)
         /// Default GPU resource set
         rgpu_res m_gpu_res;
#endif
         ///
         /// Default constructor
         ///
         Device();

         ///
         /// Destructor
         ///
         ~Device() = default;
   };

   /// The forall kernel body wrapper. It should be noted that one
   /// limitation of this wrapper is that the lambda captures can
   /// only capture functions / variables that are publically available
   /// if this is called within a class object.
   template <const int NUMBLOCKS, const bool ASYNC = false, typename DBODY, typename HBODY>
   inline rrese SNLS_ForallWrap(const int st,
                               const int end,
                               rres  resv,
                               DBODY && UNUSED_GPU(d_body),
                               HBODY &&h_body)
   {
      // Additional backends can be added as seen within the MFEM_FORALL
      // which this was based on.
      
      // Device::Backend makes use of a global variable
      // so as long as this is set in one central location
      // and you don't have multiple Device objects changing
      // the backend things should just work no matter where this
      // is used.
      switch(Device::GetInstance().GetBackend()) {
#if defined(__snls_gpu_active__)
         case(ExecutionStrategy::GPU): {
#if defined(RAJA_ENABLE_CUDA)
            using gpu_exec_policy = RAJA::cuda_exec<NUMBLOCKS, ASYNC>;
#else
            using gpu_exec_policy = RAJA::hip_exec<NUMBLOCKS, ASYNC>;
#endif
            auto res = std::get<rgpu_res>(resv);
            return RAJA::forall<gpu_exec_policy>(res, RAJA::RangeSegment(st, end), std::forward<DBODY>(d_body));
         }
#endif
#if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
         case(ExecutionStrategy::OPENMP): {
            auto res = std::get<rhost_res>(resv);
            return RAJA::forall<RAJA::omp_parallel_for_exec>(res, RAJA::RangeSegment(st, end), std::forward<HBODY>(h_body));
         }
#endif
         case(ExecutionStrategy::CPU):
         default: {
            // Moved from a for loop to raja forall so that the chai ManagedArray
            // would automatically move the memory over
            auto res = std::get<rhost_res>(resv);
            return RAJA::forall<RAJA::seq_exec>(res, RAJA::RangeSegment(st, end), std::forward<HBODY>(h_body));
         }
      } // End of switch
      return rrese{};
   } // end of forall wrap

   /// An alternative to the macro forall interface and copies more or less the 
   /// MFEM's team alternative design as well. This new formulation should allow for better debug information.
   /// So, it should allow better debug information and also better control over our lambda
   /// functions and what we capture in them.
   /// One is required to provide the desired RAJA::resources::Resource through an SNLS resource variant
   template <const int NUMBLOCKS=SNLS_GPU_BLOCKS, const bool ASYNC=false, typename BODY>
   inline rrese forall(const int st,
                      const int end,
                      rres res,
                      BODY &&body)
   {
      return SNLS_ForallWrap<NUMBLOCKS, ASYNC>(st, end, res, std::forward<BODY>(body), std::forward<BODY>(body));
   }

   /// This is essentially the same as forall variant that uses the resource set, but it
   /// uses whatever is the default resource / stream for either the GPU or host.
   /// Note, under the hood it calls the forall variant that uses the resource set but provides the
   /// default resource / stream for either the GPU or host.
   template <const int NUMBLOCKS=SNLS_GPU_BLOCKS, const bool ASYNC=false, typename BODY>
   inline rrese forall(const int st,
                      const int end,
                      BODY &&body)
   {
      return forall(st, end, Device::GetInstance().GetDefaultRAJAResource(), std::forward<BODY>(body));
   }

   /// This method allows one to pass in an execution strategy so which the forall will swap over to
   /// only for this one call. Once the forall call finishes, it will revert back to the original
   /// execution strategy. 
   /// Note, under the hood, it makes a call to the forall call that uses the default resource set.
   /// Note 2, this method does not allow for async calls and does not return a RAJA resource event
   /// that one can check. Since, we can't later on easily wait on the resource event...
   template <const int NUMBLOCKS=SNLS_GPU_BLOCKS, typename BODY>
   inline void forall_strat(const int st,
                            const int end,
                            ExecutionStrategy strat,
                            BODY &&body)
   {
      auto prev_strat = Device::GetInstance().GetBackend();
      Device::GetInstance().SetBackend(strat);
      forall(st, end, std::forward<BODY>(body));
      Device::GetInstance().SetBackend(prev_strat);
   }

   /**
    * @brief Block-packing arithmetic used by snls::forall_team to
    * cooperatively pack NTEAM-wide "teams" of threads -- one team per
    * iteration index -- into physical CUDA/HIP blocks sized to the
    * caller's own NUMBLOCKS, without ever shrinking NUMBLOCKS itself.
    *
    * NTEAM is the number of threads that cooperate (via
    * RAJA::LaunchContext::teamSync() and RAJA_TEAM_SHARED memory) on a
    * single iteration index. It has no default and is independent of any
    * particular problem size solved inside the body; it is up to the
    * caller to pick and tune it.
    *
    * **Packing arithmetic**:
    * ```
    * itemsPerBlock  = max(1, NUMBLOCKS / NTEAM)
    * effectiveBlock = itemsPerBlock * NTEAM
    * ```
    * `effectiveBlock` equals NUMBLOCKS whenever NTEAM evenly divides it;
    * otherwise it comes out a little smaller (rounding loss) -- except when
    * NTEAM > NUMBLOCKS, in which case itemsPerBlock floors to 1 and
    * effectiveBlock is forced *up* to NTEAM, exceeding the original
    * request.
    *
    * @tparam NUMBLOCKS the CUDA/HIP block size the caller already uses for
    *                   its kernel (e.g. SNLS_GPU_BLOCKS); never shrunk by
    *                   this packing.
    * @tparam NTEAM     the number of threads that cooperate on one
    *                   iteration index.
    *
    * @note NTEAM must be > 0 and <= 1024 (the max threads/block on
    *       essentially all current CUDA/HIP hardware); both are enforced
    *       with a static_assert rather than left to fail at launch time.
    *
    * @see forall_team() for the dispatch built on top of this packing.
    */
   template <int NUMBLOCKS, int NTEAM>
   struct ForallTeamPacking {
      static_assert(NTEAM > 0, "NTEAM must be positive");
      static_assert(NTEAM <= 1024,
                    "NTEAM exceeds the max threads/block on essentially all "
                    "current CUDA/HIP hardware");

      /** @brief Number of complete, independent teams packed into one
       *  physical block -- kept as large as NUMBLOCKS/NTEAM allows so the
       *  *block* stays a good occupancy size even though each individual
       *  *team* is small. */
      static constexpr int itemsPerBlock  = (NUMBLOCKS / NTEAM) > 0 ? (NUMBLOCKS / NTEAM) : 1;

      /** @brief The actual CUDA/HIP block size used, itemsPerBlock*NTEAM. */
      static constexpr int effectiveBlock = itemsPerBlock * NTEAM;
   };

   /**
    * @brief Block-wide "is any team in my block still active" consensus,
    * used by a forall_team body that has its own multi-iteration loop to
    * decide when every team sharing its physical block is done, rather
    * than exiting on a purely per-team-local condition.
    *
    * Once several teams cooperate within one block via teamSync(), a team
    * that exits its own loop early (because *it* individually is done)
    * while a block-mate is still iterating cannot safely skip that
    * block-mate's remaining teamSync() calls -- doing so produces a
    * mismatched __syncthreads() count within the same block, which is
    * undefined behavior (in practice, a hang). TeamActivityConsensus makes
    * every team in a block agree, once per outer iteration, on whether
    * *any* team still has work left, so every team's loop keeps an
    * identical teamSync() call count until the whole block finishes
    * together.
    *
    * **Mechanics**: a trivial (no user-declared constructor) aggregate, so
    * it is safe for a caller to declare one directly as
    * `RAJA_TEAM_SHARED TeamActivityConsensus<N> consensus;` inside device
    * code -- every thread in the block executes that same declaration,
    * but because RAJA_TEAM_SHARED expands to `__shared__` there, all of
    * them refer to the same single instance. Being trivial matters: a
    * non-trivial constructor would run redundantly once per thread on
    * that one shared instance, which is unnecessary at best and a data
    * race at worst. Anywhere RAJA_TEAM_SHARED instead expands to nothing
    * (CPU/OpenMP), the same declaration is just an ordinary local
    * variable, and report()'s two teamSync() calls are no-ops.
    *
    * @tparam NTEAMS the number of teams actually co-resident in the
    *                specific forall_team dispatch that constructs this
    *                instance -- itemsPerBlock for the main dispatch, or 1
    *                for both the remainder dispatch and the CPU/OpenMP
    *                fallback (see forall_team). Must match exactly; never
    *                a larger bound "just in case" -- reducing over a slot
    *                no team in this dispatch ever writes reads
    *                uninitialized shared memory, and garbage bits can
    *                make the reduction falsely "still active" forever,
    *                hanging the dispatch permanently.
    *
    * @note For NTEAMS==1 (only one team ever present -- the remainder
    *       dispatch, the CPU/OpenMP fallback, or a lone, non-packed
    *       caller), the reduction degenerates to exactly `myActive` at
    *       zero extra cost beyond the early-exit check it replaces --
    *       this is what lets a plain, single-threaded solve() call share
    *       one loop implementation with the cooperative solveTeam().
    *
    * @see forall_team() for how the two GPU dispatches and the CPU/OpenMP
    *      fallback each construct a correctly-sized instance.
    */
   template <int NTEAMS>
   struct TeamActivityConsensus {
      static_assert(NTEAMS > 0, "NTEAMS must be positive");

      /** @brief Per-team "I still have work to do" flags, one slot per
       *  team co-resident in this block; each team writes only its own
       *  m_active[teamBase] slot, so no atomics are ever needed. */
      bool m_active[NTEAMS];

      /** @brief Block-wide OR-reduction of m_active, valid for every
       *  thread to read via anyActive() only after report() returns. */
      bool m_anyActive;

      /**
       * @brief Report this team's activity state and (re-)compute the
       * block-wide consensus.
       *
       * Must be called by every thread in every team sharing this block,
       * once per outer iteration, with an identical call count across the
       * whole block -- this function itself contains two teamSync()
       * calls, so skipping it conditionally is exactly as much a hazard
       * as skipping a raw teamSync() call would be (see forall_team's
       * contract).
       *
       * @param[in] tid      this thread's role within its own team,
       *                     0..nthreads-1 (as handed to the forall_team
       *                     body).
       * @param[in] teamBase which team-slot (0..NTEAMS-1) this thread's
       *                     team occupies (as handed to the forall_team
       *                     body).
       * @param[in] myActive whether this thread's own team still has work
       *                     left to do; only tid==0's value is consulted,
       *                     since it is uniform across a team by
       *                     construction (it depends only on the team's
       *                     shared index i, never on tid itself).
       */
      __snls_hdev__ void report(int tid, int teamBase, bool myActive)
      {
         if (tid == 0) {
            m_active[teamBase] = myActive;
         }
         RAJA::LaunchContext{}.teamSync();
         if (tid == 0 && teamBase == 0) {
            bool any = false;
            for (int t = 0; t < NTEAMS; ++t) {
               any = any || m_active[t];
            }
            m_anyActive = any;
         }
         RAJA::LaunchContext{}.teamSync();
      }

      /**
       * @brief Whether any team sharing this block reported itself active
       * on the most recent report() call.
       *
       * Safe to call from any thread; only meaningful after report() has
       * returned at least once.
       *
       * @return true if at least one team co-resident in this block was
       *         still active as of the most recent report().
       */
      __snls_hdev__ bool anyActive() const { return m_anyActive; }
   };

   /**
    * @brief Like snls::forall, but hands the body a team of NTEAM
    * cooperating threads per iteration index instead of one independent
    * thread per index.
    *
    * body signature:
    * ```
    * [=] __snls_hdev__ (int i, int tid, int nthreads, int teamBase,
    *                     auto& consensus) { ... }
    * ```
    * (`consensus`'s concrete type varies by dispatch -- see below -- so a
    * body that touches it must accept it generically, e.g. via `auto&`.)
    *
    * **Body parameters**:
    * - `i` -- which iteration index (point) this thread is contributing
    *   to. ALWAYS a valid index in [st, end) -- identical guarantee to
    *   plain snls::forall; never needs a bounds check. Multiple
    *   invocations with the SAME i (different tid) are cooperating on that
    *   one index.
    * - `tid` -- this thread's role within its own team, 0..nthreads-1. By
    *   convention, tid==0 does any "exactly one thread should do this"
    *   bookkeeping.
    * - `nthreads` -- how many threads, total, are cooperating on THIS i.
    *   Equal to the compile-time NTEAM on the GPU packed dispatches, and 1
    *   on the CPU/OpenMP fallback.
    * - `teamBase` -- which "slot" (0..itemsPerBlock-1) within the current
    *   physical block this thread's team occupies. Needed only if the
    *   body declares its own RAJA_TEAM_SHARED scratch shared across the
    *   itemsPerBlock teams co-resident in one block; otherwise ignore it.
    * - `consensus` -- an already-constructed snls::TeamActivityConsensus,
    *   sized correctly for however many teams are actually co-resident in
    *   this specific dispatch (main vs. remainder vs. CPU/OpenMP
    *   fallback). Only needed if the body has its own multi-iteration
    *   loop that must know "has every team sharing my block finished, or
    *   is someone still working" -- a single-pass body can ignore it
    *   entirely.
    *
    * **Dispatch strategy**: internally issues up to two exact,
    * ragged-tail-free launches -- a main dispatch sized to a multiple of
    * itemsPerBlock, plus (only if needed) a remainder dispatch of one team
    * per block for whatever is left over -- rather than a single padded
    * launch, so every physically-launched thread always receives a valid
    * `i` and reaches exactly the teamSync() calls its block-mates do. Each
    * of the two GPU dispatches, and the CPU/OpenMP fallback, constructs
    * its own snls::TeamActivityConsensus sized to the number of teams
    * actually present in that dispatch (itemsPerBlock, 1, and 1
    * respectively) -- never a larger, shared bound.
    *
    * @tparam NUMBLOCKS the CUDA/HIP block size the caller already uses;
    *                   never shrunk by this dispatch (see
    *                   snls::ForallTeamPacking).
    * @tparam NTEAM     the number of cooperating threads per team; no
    *                   default, independent of any problem size solved
    *                   inside the body -- tune per workload.
    * @tparam BODY      deduced; the callable described above.
    *
    * @param[in] st   first iteration index (inclusive).
    * @param[in] end  last iteration index (exclusive).
    * @param[in] body callable invoked once per (index, thread-in-team)
    *                 pair; see the body-parameter list above for what it
    *                 is handed.
    *
    * **Contract for callers**:
    * -# `i` is ALWAYS valid (never a padding/out-of-range index).
    * -# If the body has its own per-index "nothing to do here" fast path
    *    (already converged, whatever), that condition must NEVER skip or
    *    wrap a call that contains a teamSync() -- directly, via a
    *    cooperative SNLS_LUP_* call, or via `consensus.report()` (which
    *    itself contains two teamSync() calls). Compute an "active" flag
    *    and use it only to gate which writes actually take effect; the
    *    loop/teamSync() *structure* must be identical for every thread in
    *    the block, always.
    * -# NTEAM is independent of any particular problem size solved inside
    *    the body -- there is no default relating them. Any index-strided
    *    loop inside a cooperative solve must stride over `nthreads`, not
    *    assume a 1:1 tid-to-row mapping, to handle NTEAM being smaller OR
    *    larger than that solve's own dimension.
    * -# If the body has its own multi-iteration loop, its termination must
    *    go through `consensus`, not a per-team-local exit condition (see
    *    snls::TeamActivityConsensus).
    *
    * @note On CPU/OpenMP, nthreads is always 1 and teamBase is always 0 --
    *       a "team of one" degenerates to today's independent-thread
    *       behavior with zero risk, and (for a single-pass body without
    *       its own loop) an early-return fast path IS safe there
    *       specifically, since nthreads==1 implies no other team shares
    *       this call.
    *
    * @see ForallTeamPacking for the itemsPerBlock/effectiveBlock arithmetic.
    * @see TeamActivityConsensus for the `consensus` parameter's contract.
    */
   template <int NUMBLOCKS, int NTEAM, typename BODY>
   inline void forall_team(int st, int end, BODY&& body)
   {
      switch (Device::GetInstance().GetBackend()) {
#if defined(__snls_gpu_active__)
         case ExecutionStrategy::GPU: {
            // Split into a main dispatch (an exact multiple of
            // itemsPerBlock, hence of effectiveBlock -- no ragged tail
            // possible) plus, only if needed, a remainder dispatch sized
            // to exactly NTEAM threads/block (one team per block -- also
            // exact, by the same argument, since remainder*NTEAM is
            // trivially a multiple of NTEAM). Neither dispatch ever lets
            // RAJA skip a physical thread that would otherwise fail to
            // reach a teamSync() its block-mates do reach.
            using Packing = ForallTeamPacking<NUMBLOCKS, NTEAM>;
            constexpr int itemsPerBlock  = Packing::itemsPerBlock;
            constexpr int effectiveBlock = Packing::effectiveBlock;
            const int totalItems = end - st;
            const int fullCount = itemsPerBlock * (totalItems / itemsPerBlock);
            const int remainder = totalItems - fullCount;

            if (fullCount > 0) {
               snls::forall<effectiveBlock>(0, fullCount * NTEAM,
                  [=] __snls_hdev__ (int gidx) {
                     const int localIdx = gidx % effectiveBlock;
                     const int i        = st + gidx / NTEAM;
                     const int tid      = gidx % NTEAM;
                     const int teamBase = localIdx / NTEAM;
                     // One instance per physical block (§3.6/§10): every
                     // thread in the block executes this same declaration,
                     // but RAJA_TEAM_SHARED makes them all refer to the
                     // same storage. Sized to itemsPerBlock -- exactly the
                     // number of teams co-resident in this dispatch.
                     RAJA_TEAM_SHARED TeamActivityConsensus<itemsPerBlock> consensus;
                     body(i, tid, NTEAM, teamBase, consensus);
                  });
            }
            if (remainder > 0) {
               snls::forall<NTEAM>(0, remainder * NTEAM,
                  [=] __snls_hdev__ (int gidx) {
                     const int i   = st + fullCount + gidx / NTEAM;
                     const int tid = gidx % NTEAM;
                     // Exactly one team per block in the remainder
                     // dispatch (§3.7) -- trivial by construction.
                     RAJA_TEAM_SHARED TeamActivityConsensus<1> consensus;
                     body(i, tid, NTEAM, 0, consensus);
                  });
            }
            break;
         }
#endif
         case ExecutionStrategy::OPENMP:
         case ExecutionStrategy::CPU:
         default: {
            // No packing, no barrier hazard -- a "team of one" per index.
            snls::forall<NUMBLOCKS>(st, end, [=] __snls_hdev__ (int i) {
               RAJA_TEAM_SHARED TeamActivityConsensus<1> consensus;
               body(i, 0, 1, 0, consensus);
            });
            break;
         }
      }
   }

}
#endif // SNLS_RAJA_PORT_SUITE || SNLS_RAJA_ONLY
#endif /* SNLS_device_forall_h */
