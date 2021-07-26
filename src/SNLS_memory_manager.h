/**************************************************************************
 Module:  SNLS_memory_manager
 Purpose: Provides a memory manager that is based on the Umpire memory
          management library. 
 ***************************************************************************/

#ifndef SNLS_memory_manager_h
#define SNLS_memory_manager_h

#include "SNLS_cuda_portability.h"
#include "SNLS_device_forall.h"

#ifdef HAVE_UMPIRE
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#endif
#ifdef HAVE_CHAI
#include "chai/ManagedArray.hpp"
#endif

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
            es = snls::Device::GetCHAIES();
            chai::ManagedArray<T> array(size, 
            std::initializer_list<chai::ExecutionSpace>{chai::CPU
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
               , chai::GPU
#endif
               },
               std::initializer_list<umpire::Allocator>{_host_allocator
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
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
            auto array = new chai::ManagedArray<T>(size, 
            std::initializer_list<chai::ExecutionSpace>{chai::CPU
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
               , chai::GPU
#endif
               },
               std::initializer_list<umpire::Allocator>{_host_allocator
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
               , _device_allocator
#endif
               },
               es);

            return array;

         }
         /*! Allocates a pointer with a given size / num pts through the use
	       *  of the host allocator. This ptr should later be deallocated
	       *  using the appropriate dealloc() function.
	       */
         template<typename T>
         __snls_host__
         inline
         T* allocHost(std::size_t size=0)
         {
#ifdef HAVE_UMPIRE
            return static_cast<T*>(_host_allocator.allocate(size * sizeof(T)));
#else
            return new T[size];
#endif
         }
#ifdef __CUDACC__
         /*! Allocates a pointer with a given size / num pts through the use
	       *  of the device allocator. This ptr should later be deallocated using
	       *  the appropriate dealloc() function.
	       *  If the need arises later on we could also add an umpire::DeviceAllocator,
	       *  so we can do allocations within device kernels rather than doing allocations
	       *  on the host for the device memory.
	       */
         template<typename T>
         __snls_host__
	      inline
         T* allocDevice(std::size_t size=0)
         {
#ifdef HAVE_UMPIRE
            return static_cast<T*>(_device_allocator.allocate(size * sizeof(T)));
#else
            T *ptr;
            CUDART_CHECK(cudaMalloc((void * *) &ptr, size * sizeof(T)));
	         cudaDeviceSynchronize();
	         return ptr;
#endif
         }
#endif
         /*! Uses the class default allocator to allocate data using
          *   the umpire resource manager.
          *   
          *   The default allocator depends on what the global value of
          *   the Device class. If it is set to CPU or OPENMP then the
          *   host allocator is used. If it is set to CUDA then it makes
          *   use of the device allocator
          */
         template<typename T>
         __snls_host__
         T* alloc(std::size_t size=0)
         {
            // Device::Backend makes use of a global variable
            // so as long as this is set in one central location
            // and you don't have multiple Device objects changing
            // the backend things should just work no matter where this
            // is used.
            switch(Device::GetBackend()) {
#ifdef __CUDACC__
               case(ExecutionStrategy::CUDA): {
                  return allocDevice<T>(size);
               }
#endif
               case(ExecutionStrategy::OPENMP):
               case(ExecutionStrategy::CPU):
               default: {
                 return allocHost<T>(size);
               }
            }
         } 
	 
	      /*! 
          * \brief Copies data from one umpire pointer to another pointer.
	       *   
          *  If you are unsure if this pointer exists within an Umpire
          *  allocator use the isUmpirePointer() method to find out.
          *  If neither pointer was created with Umpire consider
	       *  using the copyHost2Device, copyDevice2Host, copyHost2Host, or 
          *  copyDevice2Device, since they will do the appropriate copy
          *  for you. This function will fail if either pointer wasn't allocated
          *  by Umpire.
          *
          * The dst_ptr must be large enough to accommodate size bytes of data.
          *
          * \param src_ptr Source pointer.
          * \param dst_ptr Destination pointer.
          * \param size Data size in bytes. 
          */
#ifdef HAVE_UMPIRE
	      __snls_host__
	      void copy(void* src_ptr, void* dst_ptr, std::size_t size=0);
#endif
         /// Copies data from a host pointer over to a device pointer. The device pointer
	      /// should be large enough to hold the data.
#ifdef __CUDACC__
	      __snls_host__
	      void copyHost2Device(const void* host_ptr, void* device_ptr, std::size_t size=0);
         /// Copies data from a device pointer over to a host pointer. The host pointer
	      /// should be large enough to hold the data.
	      __snls_host__
	      void copyDevice2Host(const void* device_ptr, void* host_ptr, std::size_t size=0);
         /// Copies data from a device pointer over to another device pointer.
	      /// The destination device pointer should be large enough to hold the data.
         __snls_host__
	      void copyDevice2Device(const void* src_ptr, void* dst_ptr, std::size_t size=0);
#endif
	      /// Copies data from a host pointer over to another host pointer. The destination
	      /// host pointer should be large enough to hold the data.
	      __snls_host__
	      void copyHost2Host(const void* src_ptr, void* dst_ptr, std::size_t size=0);
         /// Tells one whether the given pointer was allocated using Umpire.
	      __snls_host__
	      bool isUmpirePointer(void* ptr);
         /// Deallocate a pointer given the ExecutionSpace if Umpire isn't available.
	      /// This version requires is templated on the typename and Umpire's isn't templated at all.
         /// If Umpire is available then this deallocates using Umpire's internal model.
         template<typename T>
         __snls_host__
         void dealloc(T* ptr)
         {
            if(ptr){
#ifdef HAVE_UMPIRE
               deallocUmpire(ptr);
#else
               switch(Device::GetBackend()) {
#ifdef __CUDACC__
                  case(ExecutionStrategy::CUDA): {
                     CUDART_CHECK(cudaFree(ptr));
                     break;
                  }
#endif
                  case(ExecutionStrategy::OPENMP):
                  case(ExecutionStrategy::CPU):
                  default: {
                     delete[] ptr;
                     break;
                  }
               }
#endif
            }
            ptr = nullptr;
         }
	      __snls_host__
	      virtual ~memoryManager(){}
     private:
         memoryManager();
         bool _complete;
#ifdef HAVE_UMPIRE
         void deallocUmpire(void* ptr);
	      umpire::Allocator _host_allocator;
#ifdef __CUDACC__
	      umpire::Allocator _device_allocator;
#endif
	      umpire::ResourceManager& _rm;
#endif
#ifdef HAVE_CHAI
         chai::ExecutionSpace es;
#endif
   };

}

#endif
