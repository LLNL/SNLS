/**************************************************************************
 Module:  MS_device_forall
 Purpose: Provides an abstraction layer over various different execution
          backends to allow for an easy way to write code for either the
          CPU, OpenMP, or the GPU using a single piece of code.
 ***************************************************************************/

#ifndef SNLS_device_forall_h
#define SNLS_device_forall_h

#ifndef SNLS_GPU_THREADS
#define SNLS_GPU_THREADS 256
#endif

#include "SNLS_config.h"

#if defined(SNLS_RAJA_PERF_SUITE)
#include "RAJA/RAJA.hpp"
#include "chai/config.hpp"
#include "chai/ExecutionSpaces.hpp"

// Implementation of SNLS's "parallel for" (forall) device/host kernel
// interfaces supporting RAJA and sequential backends which is based on
// the MFEM_FORALL macro.
// An example would be something like:
// SNLS_FORALL(i, 0, 50, {var[i] = 0;});

#ifdef SNLS_RAJA_EXPT_FEATURES

using snls_policy_list = camp::list<RAJA::loop_exec
#if defined(RAJA_ENABLE_OPENMP)
                               ,RAJA::omp_parallel_for_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                               ,RAJA::cuda_exec<SNLS_GPU_THREADS>
#endif
                               >;
  
/// The SNLS_FORALL wrapper where GPU threads are set to a default value
#define SNLS_FORALL(i, st, end, ...)            \
snls::SNLS_ForallWrap<SNLS_GPU_THREADS, false>(	\
st,                                             \
end,                                            \
[=] __snls_hdev__ (int i) {__VA_ARGS__})

/// The SNLS_FORALL wrapper that allows one to change the number of GPU threads
#define SNLS_FORALL_T(i, threads, st, end, ...)  \
snls::SNLS_ForallWrap<threads, true>(			    \
st,                                              \
end,                                             \
[=] __snls_hdev__ (int i) {__VA_ARGS__})

#else

#define SNLS_FORALL(i, st, end, ...)           \
snls::SNLS_ForallWrap<SNLS_GPU_THREADS>(		  \
st,                                            \
end,                                           \
[=] __snls_device__ (int i) {__VA_ARGS__},     \
[=] (int i) {__VA_ARGS__})

/// The SNLS_FORALL wrapper that allows one to change the number of GPU threads
#define SNLS_FORALL_T(i, threads, st, end, ...)  \
snls::SNLS_ForallWrap<threads>(			          \
st,                                              \
end,                                             \
[=] __snls_device__ (int i) {__VA_ARGS__},       \
[=] (int i) {__VA_ARGS__})

#endif

// offset a vector / matrix to correct starting memory location given:
// what local element we're on, what offset that local element is from the global 0,
// and then the unrolled size of the vector / matrix
#define SNLS_TOFF(ielem, offset, ndim) (offset * ndim) + (ielem * ndim)
#define SNLS_VOFF(ielem, ndim) (ielem * ndim)
#define SNLS_MOFF(ielem, ndim2) (ielem * ndim2)

namespace snls {

   typedef RAJA::View<bool, RAJA::Layout<1> > rview1b;
   typedef RAJA::View<double, RAJA::Layout<1> > rview1d;
   typedef RAJA::View<double, RAJA::Layout<2> > rview2d;
   typedef RAJA::View<double, RAJA::Layout<3> > rview3d;

   /// ExecutionStrategy defines how one would like the
   /// computations done.
   /// CPU refers to serial executions of for loops on the Host
   /// through RAJA forall abstractions
   /// OPENMP refers to parallel executuons of for loops on the 
   /// Host using OpenMP through RAJA forall abstractions
   /// CUDA refers to parallel executions of for loops on the Device
   /// using CUDA through RAJA forall abstractions
   enum class ExecutionStrategy { CPU, CUDA, OPENMP };
   /// This has largely been inspired by the MFEM device
   /// class, since they make use of it with their FORALL macro
   /// It's recommended to only have one object for the lifetime
   /// of multiple models being used, so no clashing with
   /// multiple objects can occur in regards to which models
   /// run on what ExecutionStrategy backend.
   class Device {
      private:
         static Device device_singleton;
         ExecutionStrategy _es;
         static Device& Get() { return device_singleton; }
      public:
#ifdef __CUDACC__
         Device() : _es(ExecutionStrategy::CUDA) {}
#else
         Device() : _es(ExecutionStrategy::CPU) {}
#endif
         Device(ExecutionStrategy es) : _es(es) {
            Get()._es = es;
         }
         void SetBackend(ExecutionStrategy es) { Get()._es = es; }
         static inline ExecutionStrategy GetBackend() { return Get()._es; }

         static inline chai::ExecutionSpace GetCHAIES() 
         {
            switch (Get()._es) {
#ifdef __CUDACC__
               case ExecutionStrategy::CUDA: {
                  return chai::ExecutionSpace::GPU;
               }
#endif
#ifdef OPENMP_ENABLE
               case ExecutionStrategy::OPENMP:
#endif
               case ExecutionStrategy::CPU:
               default: {
                  return chai::ExecutionSpace::CPU;
               }
            }
         }

         static inline int GetPolicyNumber() {
            int pol_id = 0;
            if (Get()._es == ExecutionStrategy::CPU) {
               return pol_id;
            }
            pol_id++;
#ifdef OPENMP_ENABLE
            if (Get()._es == ExecutionStrategy::OPENMP) {
               return pol_id;
            }
            pol_id++;
#endif
#ifdef __CUDACC__
            if (Get()._es == ExecutionStrategy::CUDA) {
               return pol_id;
            }
#endif
            return pol_id;
         }

         ~Device() {}
   };



#ifdef SNLS_RAJA_EXPT_FEATURES
   /// The forall kernel body wrapper. It should be noted that one
   /// limitation of this wrapper is that the lambda captures can
   /// only capture functions / variables that are publicly available
   /// if this is called within a class object.
   template <const int NUMTHREADS, const bool GPU_THREADS, 
             typename HDBODY>
   inline void SNLS_ForallWrap(const int st,
                               const int end,
                               HDBODY &&hd_body)
   {
      // Additional backends can be added as seen within the MFEM_FORALL
      // which this was based on.
      
      // Device::Backend makes use of a global variable
      // so as long as this is set in one central location
      // and you don't have multiple Device objects changing
      // the backend things should just work no matter where this
      // is used.
      if (GPU_THREADS) {
         switch(Device::GetBackend()) {
   #ifdef RAJA_ENABLE_CUDA
            case(ExecutionStrategy::CUDA): {
               RAJA::forall<RAJA::cuda_exec<NUMTHREADS>>(RAJA::RangeSegment(st, end), hd_body);
               break;
            }
   #endif
            default: {
               // Moved from a for loop to raja forall so that the chai ManagedArray
               // would automatically move the memory over
               const int pol_id = Device::GetPolicyNumber();
               RAJA::expt::dynamic_forall<snls_policy_list>(pol_id, RAJA::RangeSegment(st, end), hd_body);
               break;
            }
         } // End of switch
      }
      else {
         const int pol_id = Device::GetPolicyNumber();
         RAJA::expt::dynamic_forall<snls_policy_list>(pol_id, RAJA::RangeSegment(st, end), hd_body);
      }
   } // end of forall wrap
#else
   /// The forall kernel body wrapper. It should be noted that one
   /// limitation of this wrapper is that the lambda captures can
   /// only capture functions / variables that are publically available
   /// if this is called within a class object.
   template <const int NUMTHREADS, typename DBODY, typename HBODY>
   inline void SNLS_ForallWrap(const int st,
                               const int end,
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
      switch(Device::GetBackend()) {
#ifdef RAJA_ENABLE_CUDA
         case(ExecutionStrategy::CUDA): {
            RAJA::forall<RAJA::cuda_exec<NUMTHREADS>>(RAJA::RangeSegment(st, end), d_body);
            break;
         }
#endif
#if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
         case(ExecutionStrategy::OPENMP): {
            RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(st, end), h_body);
            break;
         }
#endif
         case(ExecutionStrategy::CPU):
         default: {
            // Moved from a for loop to raja forall so that the chai ManagedArray
            // would automatically move the memory over
            RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(st, end), h_body);
            break;
         }
      } // End of switch
   } // end of forall wrap
#endif // SNLS_RAJA_EXPT_FEATURES
}
#endif // SNLS_RAJA_PERF_SUITE
#endif /* SNLS_device_forall_h */
