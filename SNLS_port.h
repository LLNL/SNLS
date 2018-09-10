#ifndef SNLS_port_h__
#define SNLS_port_h__

#if HAVE_MSLIB
#include "MS_port.h"
#else

/* The interface uses 'real8' as a type */
typedef double real8 ;

#endif
/* HAVE_MSLIB */

#endif
/* SNLS_port_h__ */
