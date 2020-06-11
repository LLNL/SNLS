#ifndef SNLS_port_h__
#define SNLS_port_h__

#if SNLS_HAVE_MSLIB

#include "MS_port.h"
#include "MS_Log.h"
#ifdef __cuda_host_only__
#define SNLS_FAIL(loc,str) MS_Fail(loc,str);
#endif

#include "MS_math.h"

#else
/* SNLS_HAVE_MSLIB */

/*
 * used to do "typedef double double", but have switched to just using double
 */

#ifdef __cuda_host_only__
#include <stdio.h>
#include <exception>
#include <stdexcept>
#define SNLS_FAIL(loc,str) throw std::runtime_error(std::string("at ") + std::string(loc) + std::string(" failure : ") + std::string(str)) ;
#else
#define SNLS_FAIL(loc,str) printf("ERROR : SNLS failure in %s : %s\n",loc,str) ;
#endif

#if defined(_WIN32) && __INTEL_COMPILER
#include <mathimf.h>
#else
#include <math.h>
#endif


#endif
/* SNLS_HAVE_MSLIB */

#endif
/* SNLS_port_h__ */
