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

}
#endif // SNLS_RAJA_PORT_SUITE || SNLS_RAJA_ONLY
#endif /* SNLS_device_forall_h */
