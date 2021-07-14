#ifndef PROB_GPU_THREADS
#define PROB_GPU_THREADS 256
#endif

#include "RAJA/RAJA.hpp"

#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "chai/ManagedArray.hpp"

/// The PROB_FORALL wrapper where GPU threads are set to a default value
#define PROB_FORALL(i, st, end, ...)           \
PROB_ForallWrap<PROB_GPU_THREADS>(		  \
st,                                            \
end,                                           \
[=] __device__ (int i) {__VA_ARGS__},     \
[&] (int i) {__VA_ARGS__})

/// The MFEM_FORALL wrapper that allows one to change the number of GPU threads
#define PROB_FORALL_T(i, threads, st, end, ...)  \
PROB_ForallWrap<threads>(			          \
st,                                              \
end,                                             \
[=] __device__ (int i) {__VA_ARGS__},       \
[&] (int i) {__VA_ARGS__})
   /// This has largely been inspired by the MFEM device
   /// class, since they make use of it with their FORALL macro
   /// It's recommended to only have one object for the lifetime
   /// of the material models being used, so no clashing with
   /// multiple objects can occur in regards to which models
   /// run on what ExecutionSpace backend.

   class Device {
      private:
         static Device device_singleton;
         chai::ExecutionSpace _es;
         static Device& Get() { return device_singleton; }
      public:
#ifdef __CUDACC__
         Device() : _es(chai::ExecutionSpace::GPU) {}
#else
         Device() : _es(chai::ExecutionSpace::CPU) {}
#endif
         Device(chai::ExecutionSpace es) : _es(es) {
            Get()._es = es;
         }
         void SetBackend(chai::ExecutionSpace es) { Get()._es = es; }
         static inline chai::ExecutionSpace GetBackend() { return Get()._es; }
         ~Device() {
#ifdef __CUDACC__
            Get()._es = chai::ExecutionSpace::GPU;
#else
            Get()._es = chai::ExecutionSpace::CPU;
#endif
         }
   };

      Device Device::device_singleton;
      

   /// The forall kernel body wrapper. It should be noted that one
   /// limitation of this wrapper is that the lambda captures can
   /// only capture functions / variables that are publically available
   /// if this is called within a class object.
   template <const int NUMTHREADS, typename DBODY, typename HBODY>
   inline void PROB_ForallWrap(const int st,
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
         case(chai::ExecutionSpace::GPU): {
            printf("Running on GPU...\n");
            RAJA::forall<RAJA::cuda_exec<NUMTHREADS>>(RAJA::RangeSegment(st, end), d_body);
            break;
         }
   #endif
   #ifdef RAJA_ENABLE_OPENMP
         case(chai::ExecutionSpace::NONE): {
            RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(st, end), h_body);
            break;
         }
   #endif
#endif
         case(chai::ExecutionSpace::CPU):
         default: {
            // Moved from a for loop to raja forall so that the chai ManagedArray
            // would automatically move the memory over
            RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(st, end), h_body);
            break;
         }
      } // End of switch
   } // end of forall wrap

class memoryManager2 {
   public:
      memoryManager2() :
      _complete(false),
      _rm(umpire::ResourceManager::getInstance())
      {
         _host_allocator = _rm.getAllocator("HOST");
   #ifdef __CUDACC__
         // Do we want to make this pinned memory instead?
         _device_allocator = _rm.makeAllocator<umpire::strategy::DynamicPool>
                            ("DEVICE_pool", _rm.getAllocator("DEVICE"));
   #endif
      }
     
      /** Changes the internal host allocator to be one that
       *  corresponds with the integer id provided. This method
       *  should be preferably called before the class is initialized
       *  as complete.
       *  This host allocator should hopefully not be a pooled memory allocator
       *  due to performance reasons.  
       */
      __host__
      void setHostAllocator(int id)
      {
         if(_rm.getAllocator(id).getPlatform() == umpire::Platform::host) {
            _host_allocator = _rm.getAllocator(id);
         } else {
            printf("memoryManager::setHostAllocator. The supplied id should be associated with a host allocator");
         }
      }
      /** Changes the internal device allocator to be one that
       *  corresponds with the integer id provided. This method
       *  should be preferably called before the class is initialized
       *  as complete.
       *  This device allocator should hopefully be a pooled memory allocator
       *  due to performance reasons.
       */
      __host__
      void setDeviceAllocator(int id)
      {
   #ifdef __CUDACC__
         // We don't want to disassociate our default device allocator from
         // Umpire just in case it still has memory associated with it floating around.
         if(_rm.getAllocator(id).getPlatform() == umpire::Platform::cuda) {
            _device_allocator = _rm.getAllocator(id);
         } else {
            printf("memoryManager::setDeviceAllocator. The supplied id should be associated with a device allocator");
         }
   #endif
      }
      /// Tells the class that it is now considered completely initialized
      __host__
      void complete() { _complete = true; }
      /// Returns a boolean for whether or not the class is complete
      __host__
      bool getComplete() { return _complete; }


      template<typename T>
      __host__
      inline
      chai::ManagedArray<T> allocManagedArray(std::size_t size=0)
      {
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
         chai::ExecutionSpace::CPU);

         return array;
      }

      template<typename T>
      __host__
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
         chai::ExecutionSpace::CPU);

         return array;

      }

      virtual ~memoryManager2(){}
  private:
      bool _complete = false;
#ifdef HAVE_UMPIRE
      umpire::Allocator _host_allocator;
#ifdef __CUDACC__
      umpire::Allocator _device_allocator;
#endif
      umpire::ResourceManager& _rm;
#endif
};

__host__ __device__
void test_fcn(chai::ManagedArray<double> data, int i)
{
   data[i] = 1.0;
   return;
}

class testCase
{
   public:
      chai::ManagedArray<double> data_public;
      // This is the only way I've found to work...

      testCase(int nBatch) 
      {
#ifdef __CUDA_ARCH__
#else
         init(nBatch);
#endif
      }
      ~testCase()
      {
#ifdef __CUDA_ARCH__
#else
         data_public.free();
         data_private.free();
#endif
      }

      void init(int nBatch)
      {
         memoryManager2 mm;

         auto test = mm.allocManagedArray<double>(nBatch);

         // Now this should work...
         printf("Running a simple test of things...");
         PROB_FORALL(i, 0, nBatch, {
            test[i] = 1.0;
         });

         test.free();

         // neither of these methods work
         // You just get that error message
         // I've also tried making this a pointer as well...
         data_public = mm.allocManagedArray<double>(nBatch); 
         data_private = mm.allocManagedArray<double>(nBatch);
         printf("\nin init for testCase class\n");
         auto &data_public_ref = data_public;
         auto &data_private_ref = data_private;
         PROB_FORALL(i, 0, nBatch, {
            // test_fcn(data_public, i);
            data_public[i] = 1.0;
            data_private[i] = 1.0;
         });
         printf("it worked \n");
      }

   private:
      chai::ManagedArray<double> data_private;
};

int main()
{
    testCase test(5000);
    return 0;
}
