/**************************************************************************
 Module:  SNLS_device_forall
 Purpose: Provides an abstraction layer over various different execution
          backends to allow for an easy way to write code for either the
          CPU, OpenMP, or the GPU using a single piece of code.
 ***************************************************************************/

#include "SNLS_device_forall.h"
#if defined(SNLS_RAJA_PERF_SUITE)
namespace snls {
   Device& Device::GetInstance() {
      static Device s_device;
      return s_device;
   }

   Device::Device() :
#if defined(__snls_gpu_active__)
      m_es{ExecutionStrategy::GPU},
      m_host_res{rhost_res::get_default()},
      m_gpu_res{rgpu_res::get_default()}
#else
      m_es{ExecutionStrategy::CPU},
      m_host_res{rhost_res::get_default()}
#endif
   {
   }

   chai::ExecutionSpace Device::GetCHAIES() {
      switch (m_es) {
#if defined(__snls_gpu_active__)
         case ExecutionStrategy::GPU:
            return chai::ExecutionSpace::GPU;
#endif

#if defined(OPENMP_ENABLE)
         case ExecutionStrategy::OPENMP:
#endif
         case ExecutionStrategy::CPU:
         default:
            return chai::ExecutionSpace::CPU;
      }
   }
   rres Device::GetRAJAResource() {
      switch (m_es) {
#if defined(__snls_gpu_active__)
         case ExecutionStrategy::GPU:
         {
            return rres(m_gpu_res);

         }
#endif

#if defined(OPENMP_ENABLE)
         case ExecutionStrategy::OPENMP:
#endif
         case ExecutionStrategy::CPU:
         default:
            return rres(m_host_res);
      }
   }
}
#endif
