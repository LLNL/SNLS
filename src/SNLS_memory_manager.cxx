/**************************************************************************
 Module:  SNLS_memory_manager
 Purpose: Provides a memory manager that is based on the Umpire memory
          management library. 
 ***************************************************************************/

#include "SNLS_memory_manager.h"
#include "SNLS_device_forall.h"
#include "SNLS_port.h"

#ifdef HAVE_UMPIRE
#include "umpire/strategy/DynamicPool.hpp"
#endif

#include <cstring>

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

#ifdef HAVE_UMPIRE
   memoryManager::memoryManager() :
   _complete(false),
   _rm(umpire::ResourceManager::getInstance())
   {
      _host_allocator = _rm.makeAllocator<umpire::strategy::DynamicPool>
                         ("MSLib_HOST_pool", _rm.getAllocator("HOST"));
      // _host_allocator = _rm.getAllocator("HOST");
#ifdef __CUDACC__
      // Do we want to make this pinned memory instead?
      _device_allocator = _rm.makeAllocator<umpire::strategy::DynamicPool>
	                      ("MSLib_DEVICE_pool", _rm.getAllocator("DEVICE"));
      es = chai::ExecutionSpace::GPU;
#else
      es = chai::ExecutionSpace::CPU;
#endif
   }
#else
   memoryManager::memoryManager() :
   _complete(false){}
#endif
  
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
#ifdef HAVE_UMPIRE
      if(_rm.getAllocator(id).getPlatform() == umpire::Platform::host) {
         _host_allocator = _rm.getAllocator(id);
      } else {
         SNLS_FAIL("memoryManager::setHostAllocator", "The supplied id should be associated with a host allocator");
      }
#else
      SNLS_FAIL("memoryManager::setHostAllocator", "SNLS was not compiled with Umpire");
#endif
   }
   /** Changes the internal device allocator to be one that
    *  corresponds with the integer id provided. This method
    *  should be preferably called before the class is initialized
    *  as complete.
    *  This device allocator should hopefully be a pooled memory allocator
    *  due to performance reasons.
    */
   __snls_host__
   void memoryManager::setDeviceAllocator(int id)
   {
#ifdef __CUDACC__
#ifdef HAVE_UMPIRE
      // We don't want to disassociate our default device allocator from
      // Umpire just in case it still has memory associated with it floating around.
      if(_rm.getAllocator(id).getPlatform() == umpire::Platform::cuda) {
         _device_allocator = _rm.getAllocator(id);
      } else {
         SNLS_FAIL("memoryManager::setDeviceAllocator", "The supplied id should be associated with a device allocator");
      }
#else
      SNLS_FAIL("memoryManager::setDeviceAllocator", "SNLS was not compiled with Umpire");
#endif
#endif
   }
   /*! 
    * \brief Copies data from one umpire pointer to another pointer.
    *   
    *  If you are unsure if this pointer exists within an Umpire
    *  allocator use the isUmpirePointer() method to find out.
    *  If neither pointer was created with Umpire consider
    *  using the copyHost2Device, copyDevice2Host, copyHost2Host, or 
    *  copyDevice2Device, since they will do the appropriate copy
    *  for you.
    *
    * The dst_ptr must be large enough to accommodate size bytes of data.
    *
    * \param src_ptr Source pointer.
    * \param dst_ptr Destination pointer.
    * \param size Data size in bytes. 
    */
#ifdef HAVE_UMPIRE
   __snls_host__
   void memoryManager::copy(void* src_ptr, void* dst_ptr, std::size_t size)
   {
      if (!_rm.hasAllocator(src_ptr) || !_rm.hasAllocator(dst_ptr))
      {
	 SNLS_FAIL("memoryManager::copy","One of the provided pointers was not created with Umpire.");
      }
      _rm.copy(dst_ptr, src_ptr, size);
   }
#endif
#ifdef __CUDACC__
   /// Copies data from a host pointer over to a device pointer. The device pointer
   /// should be large enough to hold the data.
   __snls_host__
   void memoryManager::copyHost2Device(const void* host_ptr,
				                           void* device_ptr,
				                           std::size_t size)
   {
      CUDART_CHECK(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice));
   }
   /// Copies data from a device pointer over to a host pointer. The host pointer
   /// should be large enough to hold the data.
   __snls_host__
   void memoryManager::copyDevice2Host(const void* device_ptr,
                                       void* host_ptr,
                                       std::size_t size)
   {
      CUDART_CHECK(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost));
   }
   /// Copies data from a device pointer over to another device pointer.
   /// The destination device pointer should be large enough to hold the data.
   __snls_host__
   void memoryManager::copyDevice2Device(const void* src_ptr,
                                         void* dst_ptr,
                                         std::size_t size)
   {
      CUDART_CHECK(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice));
   }
#endif
   /// Copies data from a host pointer over to another host pointer. The destination
   /// host pointer should be large enough to hold the data.
   __snls_host__
   void memoryManager::copyHost2Host(const void* src_ptr,
                                     void* dst_ptr,
                                     std::size_t size)
   {
      // What would be a nice way to error check this? Would memcpy_s be an acceptable
      // function to use?
      std::memcpy(dst_ptr, src_ptr, size);
   }
#ifdef HAVE_UMPIRE
   void memoryManager::deallocUmpire(void *ptr){
      if (!_rm.hasAllocator(ptr))
      {
         SNLS_FAIL("memoryManager::deallocUmpire", "The provided pointer was not created with Umpire.");
      }
      _rm.deallocate(ptr);
   }
#endif
   /// Tells one whether the given pointer was allocated using Umpire.
   __snls_host__
   bool memoryManager::isUmpirePointer(void* ptr)
   {
#ifdef HAVE_UMPIRE
      return _rm.hasAllocator(ptr);
#else
      return false;
#endif
   }
}
