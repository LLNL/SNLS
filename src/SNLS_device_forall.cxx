/**************************************************************************
 Module:  SNLS_device_forall
 Purpose: Provides an abstraction layer over various different execution
          backends to allow for an easy way to write code for either the
          CPU, OpenMP, or the GPU using a single piece of code.
 ***************************************************************************/

#include "SNLS_device_forall.h"
#include "SNLS_unused.h"
#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)
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

   ches Device::GetCHAIES() {
      switch (m_es) {
#if defined(__snls_gpu_active__)
         case ExecutionStrategy::GPU:
            return ches::GPU;
#endif

#if defined(OPENMP_ENABLE)
         case ExecutionStrategy::OPENMP:
#endif
         case ExecutionStrategy::CPU:
         default:
            return ches::CPU;
      }
   }

   rres Device::GetDefaultRAJAResource() {
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

   rres Device::GetRAJAResource() {
      // We should keep track of which stream we're on so we don't pass in the default by accident.
      // camp seems to only have 16 streams total and rotates through them unless we supply
      // an int value and then it goes with that choice
      static int UNUSED_GPU(stream) = 0;
      switch (m_es) {
#if defined(__snls_gpu_active__)
         case ExecutionStrategy::GPU:
         {
            // We're updating the stream value based on the same way camp does
            // One key difference is that we check to make sure we're not on the default value
            // and if so we increment the stream to avoid clashing with kernels that are using
            // the default stream.
            stream = (stream + 1) % 16;
            if (stream == 0) stream += 1;
            return rres(rgpu_res{stream});

         }
#endif

#if defined(OPENMP_ENABLE)
         case ExecutionStrategy::OPENMP:
#endif
         case ExecutionStrategy::CPU:
         default:
            return rres(rhost_res{});
      }
   }

   void Device::WaitFor(rres& res, rrese* event) {
      switch (m_es) {
#if defined(__snls_gpu_active__)
         case ExecutionStrategy::GPU:
         {
            std::get<rgpu_res>(res).wait_for(event);
            break;
         }
#endif
#if defined(OPENMP_ENABLE)
         case ExecutionStrategy::OPENMP:
#endif
         case ExecutionStrategy::CPU:
         default:
            std::get<rhost_res>(res).wait_for(event);
      }
   }

   void Device::Wait(rres& res) {
      switch (m_es) {
#if defined(__snls_gpu_active__)
         case ExecutionStrategy::GPU:
         {
            std::get<rgpu_res>(res).wait();
            break;
         }
#endif
#if defined(OPENMP_ENABLE)
         case ExecutionStrategy::OPENMP:
#endif
         case ExecutionStrategy::CPU:
         default:
            std::get<rhost_res>(res).wait();
      }
   }
  
}
#endif
