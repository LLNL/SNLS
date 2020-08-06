/**************************************************************************
 Module:  SNLS_device_forall
 Purpose: Provides an abstraction layer over various different execution
          backends to allow for an easy way to write code for either the
          CPU, OpenMP, or the GPU using a single piece of code.
 ***************************************************************************/

#include "SNLS_device_forall.h"

namespace snls {
   // ExecutionStrategy is set initially to the CPU
   // value even if this is never set.
   Device Device::device_singleton;
}