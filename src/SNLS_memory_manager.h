/**************************************************************************
 Module:  SNLS_memory_manager
 Purpose: Provides a memory manager that is based on the Umpire memory
          management library. 
 ***************************************************************************/

#ifndef SNLS_memory_manager_h
#define SNLS_memory_manager_h

#include "SNLS_config.h"
#include "SNLS_gpu_portability.h"
#include "SNLS_device_forall.h"

#if defined(SNLS_RAJA_PORT_SUITE)
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <chai/config.hpp>
#include <chai/ManagedArray.hpp>

namespace snls {
   /*! A global memory manager designed to be used with the Umpire library.
    *  It provides methods to allocate on the host(H) or device(D);
    *  copy data between the HtoD, DtoH, HtoH, or DtoD; and deallocation.
    *  Through Umpire, we can keep track of what platform the pointer came
    *  from which it easy to make sure we're not sending a device pointer to
    *  a host kernel and vice versa.
    * 
    *  If Umpire can not be used on the system then this class simply provides
    *  methods to allocate data, copy data, or deallocate on the H. If the system
    *  can make use of a D then it should be capable of compiling Umpire. Umpire
    *  has host configs all the way back to gcc-4.9.3 and clang-3.9.1 for TOSS 3
    *  systems, and all the way back to gcc-4.9.3 and clang-3.8.1 for CHAOS 5
    *  systems.
    */
   class memoryManager {
      public:
         /// Returns a reference to the global memoryManager object
         __snls_host__
         static memoryManager& getInstance();
         /*! Changes the internal host allocator to be one that
          *  corresponds with the integer id provided. This method
          *  should be preferably called before the class is initialized
          *  as complete.
          *  This host allocator should hopefully not be a pooled memory allocator
          *  due to performance reasons.  
          */
         __snls_host__
         void setHostAllocator(int id);
         /*! Changes the internal device allocator to be one that
          *  corresponds with the integer id provided. This method
          *  should be preferably called before the class is initialized
          *  as complete.
          *  This device allocator should hopefully be a pooled memory allocator
          *  due to performance reasons.
          */
         __snls_host__
         void setDeviceAllocator(int id);
         /// Tells the class that it is now considered completely initialized
         __snls_host__
         void complete() { _complete = true; }
         /// Returns a boolean for whether or not the class is complete
         __snls_host__
         bool getComplete() { return _complete; }


         template<typename T>
         __snls_host__
         inline
         chai::ManagedArray<T> allocManagedArray(std::size_t size=0)
         {
            es = snls::Device::GetInstance().GetCHAIES();
            chai::ManagedArray<T> array(size, 
            std::initializer_list<chai::ExecutionSpace>{chai::CPU
#if defined(__snls_gpu_active__)
               , chai::GPU
#endif
               },
               std::initializer_list<umpire::Allocator>{_host_allocator
#if defined(__snls_gpu_active__)
               , _device_allocator
#endif
               },
               es);

            return array;
         }

         template<typename T>
         __snls_host__
         inline
         chai::ManagedArray<T>* allocPManagedArray(std::size_t size=0)
         {
            es = snls::Device::GetInstance().GetCHAIES();
            auto array = new chai::ManagedArray<T>(size, 
            std::initializer_list<chai::ExecutionSpace>{chai::CPU
#if defined(__snls_gpu_active__)
               , chai::GPU
#endif
               },
               std::initializer_list<umpire::Allocator>{_host_allocator
#if defined(__snls_gpu_active__)
               , _device_allocator
#endif
               },
               es);

            return array;

         }
         __snls_host__
         virtual ~memoryManager(){}
     private:
         memoryManager();
         bool _complete;
         umpire::Allocator _host_allocator;
#if defined(__snls_gpu_active__)
         umpire::Allocator _device_allocator;
#endif
         umpire::ResourceManager& _rm;
         chai::ExecutionSpace es;
   };
}
#endif // HAVE_RAJA_PERF_SUITE
#endif
