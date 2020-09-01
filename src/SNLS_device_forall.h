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

#ifdef HAVE_RAJA
#include "RAJA/RAJA.hpp"
#endif

#ifdef OPENMP_ENABLE
#include <omp.h>
#endif

   // Implementation of SNLS's "parallel for" (forall) device/host kernel
   // interfaces supporting RAJA and sequential backends which is based on
   // the MFEM_FORALL macro.
   // An example would be something like:
   // SNLS_FORALL(i, 0, 50, {var[i] = 0;});
   
/// The SNLS_FORALL wrapper where GPU threads are set to a default value
#define SNLS_FORALL(i, st, end, ...)           \
snls::SNLS_ForallWrap<SNLS_GPU_THREADS>(		  \
st,                                            \
end,                                           \
[=] __snls_device__ (int i) {__VA_ARGS__},     \
[&] (int i) {__VA_ARGS__})

/// The MFEM_FORALL wrapper that allows one to change the number of GPU threads
#define SNLS_FORALL_T(i, threads, st, end, ...)  \
snls::SNLS_ForallWrap<threads>(			          \
st,                                              \
end,                                             \
[=] __snls_device__ (int i) {__VA_ARGS__},       \
[&] (int i) {__VA_ARGS__})

// offset a vector / matrix to correct starting memory location given:
// what local element we're on, what offset that local element is from the global 0,
// and then the unrolled size of the vector / matrix
#define SNLS_TOFF(ielem, offset, ndim) (offset * ndim) + (ielem * ndim)
#define SNLS_VOFF(ielem, ndim) (ielem * ndim)
#define SNLS_MOFF(ielem, ndim2) (ielem * ndim2)

namespace snls {
   enum class ExecutionStrategy { CPU, CUDA, OPENMP };
   /// This has largely been inspired by the MFEM device
   /// class, since they make use of it with their FORALL macro
   /// It's recommended to only have one object for the lifetime
   /// of the material models being used, so no clashing with
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
         ~Device() {
#ifdef __CUDACC__
            Get()._es = ExecutionStrategy::CUDA;
#else
            Get()._es = ExecutionStrategy::CPU;
#endif
         }
   };
      
#ifndef HAVE_RAJA
#ifdef __CUDACC__
template <typename BODY> __snls_global__ static
void CuKernel(const int st, const int end, BODY body)
{
   const int k = blockDim.x * blockIdx.x + threadIdx.x + st;
   if (k >= end) { return; }
   body(k);
}

template <const int BLCK, typename DBODY>
void CuWrap(const int st, const int end, DBODY &&d_body)
{
   if ((end - st) == 0) { return; }
   const int GRID = ((end - st) + BLCK - 1) / BLCK;
   CuKernel<<<GRID,BLCK>>>(st, end, d_body);
   cudaDeviceSynchronize();         // (wait for threads to complete)
   CUDART_CHECK( cudaGetLastError() );   // (check for device errors)
}
#endif

#ifdef OPENMP_ENABLE
/// OpenMP backend
template <typename HBODY>
void OmpWrap(const int st, const int end, HBODY &&h_body)
{
   #pragma omp parallel for
   for (int k = st; k < end; k++)
   {
      h_body(k);
   }
}
#endif
#endif

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
#ifdef HAVE_RAJA
   #ifdef RAJA_ENABLE_CUDA
         case(ExecutionStrategy::CUDA): {
            RAJA::forall<RAJA::cuda_exec<NUMTHREADS>>(RAJA::RangeSegment(st, end), d_body);
            break;
         }
   #endif
   #ifdef RAJA_ENABLE_OPENMP && OPENMP_ENABLE
         case(ExecutionStrategy::OPENMP): {
            RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(st, end), h_body);
            break;
         }
   #endif
#else
   #ifdef __CUDACC__
         case(ExecutionStrategy::CUDA): {
            // true denotes asynchronous kernel
            CuWrap<NUMTHREADS>(st, end, d_body);
            break;
         }
   #endif
   #ifdef OPENMP_ENABLED
         case(ExecutionStrategy::OPENMP): {
            OmpWrap(st, end, h_body);
            break;
         }
   #endif
#endif
         case(ExecutionStrategy::CPU):
         default: {
            for (int k = st; k < end; k++) { h_body(k); }
            break;
         }
      } // End of switch
   } // end of forall wrap
}


#endif /* SNLS_device_forall_h */
