/**************************************************************************
 Module:  SNLS_memory_manager
 Purpose: Provides a memory manager that is based on the Umpire memory
          management library. 
 ***************************************************************************/

#include "SNLS_memory_manager.h"
#include "SNLS_unused.h"
#include "SNLS_port.h"
#include <cstring>

#if defined(SNLS_RAJA_PORT_SUITE)

namespace snls {
  
   /** Returns a reference to the global memoryManager object
       Borrowed this implementation from how Umpire deals with its
       ResourceManager class. 
    */
   __snls_host__
   memoryManager& memoryManager::getInstance()
   {
      static memoryManager mm;
      return mm;
   }

   memoryManager::memoryManager() :
   _complete(false),
   _rm(umpire::ResourceManager::getInstance())
   {
      const int initial_size = (1024 * 1024 * 1024);
      _host_allocator = _rm.makeAllocator<umpire::strategy::QuickPool>
                         ("SNLS_HOST_pool", _rm.getAllocator("HOST"),
                          initial_size);
      // _host_allocator = _rm.getAllocator("HOST");
#if defined(__snls_gpu_active__)
      // Do we want to make this pinned memory instead?
      _device_allocator = _rm.makeAllocator<umpire::strategy::QuickPool>
	                      ("SNLS_DEVICE_pool", _rm.getAllocator("DEVICE"),
                          initial_size);
#endif
      es = snls::Device::GetInstance().GetCHAIES();
   }
  
   /** Changes the internal host allocator to be one that
    *  corresponds with the integer id provided. This method
    *  should be preferably called before the class is initialized
    *  as complete.
    *  This host allocator should hopefully not be a pooled memory allocator
    *  due to performance reasons.  
    */
   __snls_host__
   void memoryManager::setHostAllocator(int id)
   {
      if(_rm.getAllocator(id).getPlatform() == umpire::Platform::host) {
         _host_allocator = _rm.getAllocator(id);
      } else {
         SNLS_FAIL("memoryManager::setHostAllocator", "The supplied id should be associated with a host allocator");
      }
   }
   /** Changes the internal device allocator to be one that
    *  corresponds with the integer id provided. This method
    *  should be preferably called before the class is initialized
    *  as complete.
    *  This device allocator should hopefully be a pooled memory allocator
    *  due to performance reasons.
    */
   __snls_host__
   void memoryManager::setDeviceAllocator(int UNUSED_GPU(id))
   {
#if defined(__snls_gpu_active__)
      // We don't want to disassociate our default device allocator from
      // Umpire just in case it still has memory associated with it floating around.
      if((_rm.getAllocator(id).getPlatform() == umpire::Platform::cuda) || (_rm.getAllocator(id).getPlatform() == umpire::Platform::hip)) {
         _device_allocator = _rm.getAllocator(id);
      } else {
         SNLS_FAIL("memoryManager::setDeviceAllocator", "The supplied id should be associated with a device allocator");
      }
#else
      SNLS_FAIL("memoryManager::setDeviceAllocator", "SNLS was not compiled with Umpire and GPU capabilities");
#endif
   }
}

#endif // HAVE_RAJA_PERF_SUITE
