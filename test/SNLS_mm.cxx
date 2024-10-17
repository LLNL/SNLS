#include <iostream>
#include <cstdlib>

using namespace std;

#include "SNLS_config.h"

#if defined(SNLS_RAJA_PORT_SUITE)

#include "SNLS_memory_manager.h"
#include "SNLS_device_forall.h"

int main(int argc, char ** argv) {

  snls::memoryManager& mm = snls::memoryManager::getInstance();

  std::cout << "We seem to have a snls::memoryManager object." << std::endl;

  mm.complete();

  std::cout << "Our snls::memoryManager object is now complete." << std::endl;
  
  constexpr std::size_t SIZE = 1024;
   auto data = mm.allocManagedArray<double>(SIZE);

  SNLS_FORALL(i, 0, SIZE, {
    data[i] = 1.0;
  });

  std::cout << "Our data object has been set to 1." << std::endl;

  data.free();

  exit(0);
}

#endif
