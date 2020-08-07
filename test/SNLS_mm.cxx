#include <iostream>
#include <cstdlib>

using namespace std;

#include "../src/SNLS_memory_manager.h"
#include "../src/SNLS_device_forall.h"

int main(int argc, char ** argv) {

  snls::memoryManager& mm = snls::memoryManager::getInstance();

  std::cout << "We seem to have a snls::memoryManager object." << std::endl;

  mm.complete();

  std::cout << "Our snls::memoryManager object is now complete." << std::endl;
  
  constexpr std::size_t SIZE = 1024;
  double* data = mm.alloc<double>(SIZE);

  SNLS_FORALL(i, 0, SIZE, {
    data[i] = 1.0;
    });
#ifdef HAVE_UMPIRE
  if(!mm.isUmpirePointer(data)) {
    std::cerr << "This should definitely be an Umpire pointer..." << std::endl;
    exit(1);
  }
#endif
  double* cdata = mm.allocHost<double>(SIZE);
#ifdef HAVE_UMPIRE
  mm.copy(data, cdata, SIZE*sizeof(double));
  for(int i = 0; i < SIZE; i++) {
    if(!cdata[i] == 1.0) {
      std::cerr << "Copied data at " << i << " was not copied over correctly." << std::endl;
      exit(1);
    }
  }
#endif
#ifdef __CUDACC__
  mm.copyDevice2Host(data, cdata, SIZE*sizeof(double));
#else
  mm.copyHost2Host(data, cdata, SIZE*sizeof(double));
#endif
  for(int i = 0; i < SIZE; i++) {
    if(!cdata[i] == 1.0) {
#ifdef __CUDACC__
      std::cerr << "Copied D2H data at " << i << " was not copied over correctly." << std::endl;
#else
      std::cerr << "Copied H2H data at " << i << " was not copied over correctly." << std::endl;
#endif
      exit(1);
    }
  }

  if(mm.isUmpirePointer(data)) {
    std::cout << "Ptr is an umpire pointer" << std::endl;
  }

  mm.dealloc<double>(data);
  mm.dealloc<double>(cdata);

  exit(0);
}
