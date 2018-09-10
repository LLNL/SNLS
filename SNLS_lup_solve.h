#pragma once

#ifndef __SNLS_LUP_SOLVE_H
#define __SNLS_LUP_SOLVE_H

#include "SNLS_port.h"

extern __snls_hdev__ void   SNLS_LUP_Fix_Columns (real8 **a ,         int n, real8 tol);
extern __snls_hdev__ int    SNLS_LUP_Decompose   (real8 **a , int *p, int n, real8 tol);
extern __snls_hdev__ int    SNLS_LUP_Solve       (real8 **a , int *p, real8 *x, real8 *b, int n);
extern __snls_hdev__ int    SNLS_LUP_Solve       (real8  *a ,         real8 *x, real8 *b, int n, real8 tol);
extern __snls_hdev__ void   SNLS_LUP_Invert      (real8 **ai, real8 **a, int *p, int n);
extern __snls_hdev__ real8  SNLS_LUP_Determinant (real8 **a , int *p, int n);

#endif  // __SNLS_LUP_SOLVE_H
