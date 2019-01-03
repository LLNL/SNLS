#ifndef SNLS_port_h__
#define SNLS_port_h__

#if SNLS_HAVE_MSLIB
#include "MS_port.h"
#else

/* The interface uses 'real8' as a type */
typedef double real8 ;

#endif
/* SNLS_HAVE_MSLIB */

#endif
/* SNLS_port_h__ */
