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

#if defined(SNLS_RAJA_PERF_SUITE)

#include "SNLS_gpu_portability.h"
#include "RAJA/RAJA.hpp"
#include "chai/config.hpp"
#include "chai/ExecutionSpaces.hpp"

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

namespace snls {

   // Provide some simple shortcuts in-case people need something beyond the default
   template<typename T>
   using rview1 = RAJA::View<T, RAJA::Layout<1>>;
   template<typename T>
   using rview2 = RAJA::View<T, RAJA::Layout<2>>;
   template<typename T>
   using rview3 = RAJA::View<T, RAJA::Layout<3>>;

   using rview1b = rview1<bool>;
   using rview1d = rview1<double>;
   using rview2d = rview2<double>;
   using rview3d = rview3<double>;
   using crview1d = rview1<const double>;
   using crview2d = rview2<const double>;
   using crview3d = rview3<const double>;

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

   // We really don't care what View class we're using as the sub-view just wraps it up
   // and then allows us to take a slice/window of the original view
   // Should probably work out some form of SFINAE to ensure T that we're templated on
   // is an actual View that we can use.
   template<class T>
   class subview {
   public:
      // Delete the default constructor as that wouldn't be a valid object
      __snls_hdev__
      subview() = delete;
      // where we don't want any offset within the subview itself
      __snls_hdev__
      subview(const int index, T& view) : m_view(view), m_index(index), m_offset(0) {};
      // sometimes you might want to have an initial offset in your subview when constructing
      // your subview in which everything appears as 0 afterwards
      __snls_hdev__
      subview(const int index, const size_t offset, T& view) : m_view(view), m_index(index), m_offset(offset) {};

      ~subview() = default;

      // Could probably add default copy constructors as well here if need be...

      // Let the compiler figure out the correct return type here as the one from
      // RAJA at least for regular Views is non-trivial
      // make the assumption here that we're using row-major memory order for views
      // so m_index is in the location of the slowest moving index as this is the default
      // for RAJA...
      template <typename... Args>
      __snls_hdev__
      inline
      constexpr
      auto&
      operator()(Args... args) const
      {
         // The use of m_offset here provides us the equivalent of a rolling
         // subview/window if our application needs it
         return m_view(m_index, m_offset + args...);
      }

      // If we need to have like a rolling subview/window type class then
      // we'd need some way to update the offset in our slowest moving index
      // in the subview (so not m_view's slowest index)
      __snls_hdev__
      inline
      void set_offset(const int offset) const
      {
         // Might want an assert in here for debugs to make sure that this is within
         // the bounds of what m_view expects is a valid offset
         m_offset = offset;
      }

   private:
      // Internally we shouldn't be modifying the view itself so let's make it constant
      const T& m_view;
      const int m_index = 0;
      mutable size_t m_offset = 0;
   };


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
         chai::ExecutionSpace GetCHAIES();

         /// Return the default RAJA resource corresponding to the execution strategy
         /// @return a RAJA resource set
         ///
         rres GetDefaultRAJAResource();

         /// Return a new RAJA resource corresponding to the execution strategy
         /// @return a RAJA resource set
         ///
         rres GetRAJAResource();

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
   template <const int NUMBLOCKS, const bool async = false, typename DBODY, typename HBODY>
   inline void SNLS_ForallWrap(const int st,
                               const int end,
                               rres  resv,
                               DBODY &&d_body,
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
            RAJA::forall<gpu_exec_policy>(res, RAJA::RangeSegment(st, end), std::forward<DBODY>(d_body));
            break;
         }
#endif
#if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
         case(ExecutionStrategy::OPENMP): {
            auto res = std::get<rhost_res>(resv);
            RAJA::forall<RAJA::omp_parallel_for_exec>(res, RAJA::RangeSegment(st, end), std::forward<HBODY>(h_body));
            break;
         }
#endif
         case(ExecutionStrategy::CPU):
         default: {
            // Moved from a for loop to raja forall so that the chai ManagedArray
            // would automatically move the memory over
            auto res = std::get<rhost_res>(resv);
            RAJA::forall<RAJA::seq_exec>(res, RAJA::RangeSegment(st, end), std::forward<HBODY>(h_body));
            break;
         }
      } // End of switch
   } // end of forall wrap

   /// An alternative to the macro forall interface and copies more or less the 
   /// MFEM's team alternative design as well. This new formulation should allow for better debug information.
   /// So, it should allow better debug information and also better control over our lambda
   /// functions and what we capture in them.
   /// Note, under the hood this will make use of whatever is the default resource / stream
   /// for either the GPU or host.
   template <const int NUMBLOCKS=SNLS_GPU_BLOCKS, const bool ASYNC=false, typename BODY>
   inline void forall(const int st,
                      const int end,
                      BODY &&body)
   {
      SNLS_ForallWrap<NUMBLOCKS, ASYNC>(st, end, Device::GetInstance().GetDefaultRAJAResource(), std::forward<BODY>(body), std::forward<BODY>(body));
   }

   /// Essentially the same as the earlier forall(...) call except one can provide
   /// the desired RAJA::resources::Resource through an SNLS resource variant
   template <const int NUMBLOCKS=SNLS_GPU_BLOCKS, const bool ASYNC=false, typename BODY>
   inline void forall(const int st,
                      const int end,
                      rres res,
                      BODY &&body)
   {
      SNLS_ForallWrap<NUMBLOCKS, ASYNC>(st, end, res, std::forward<BODY>(body), std::forward<BODY>(body));
   }

}
#endif // SNLS_RAJA_PERF_SUITE
#endif /* SNLS_device_forall_h */
