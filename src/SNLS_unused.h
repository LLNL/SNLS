#pragma once
#include "SNLS_gpu_portability.h"

#ifdef UNUSED
#elif defined(__GNUC__)
# define UNUSED(x) UNUSED_ ## x __attribute__((unused))
#else
# define UNUSED(x) x
#endif

#if defined(__snls_gpu_active__)
#define UNUSED_GPU(x) x
#else
#if defined(UNUSED)
# define UNUSED_GPU(x) UNUSED(x)
#elif defined(__GNUC__)
# define UNUSED_GPU(x) UNUSED_ ## x __attribute__((unused))
#else
# define UNUSED_GPU(x) x
#endif
#endif