#include <stdio.h>
#include <math.h>

#include "SNLS_cuda_portability.h"

#if HAVE_MSLIB
#include "MS_port.h"
#endif

#include "SNLS_lup_solve.h"

//-----------------------------------------------------------------------------------------------
// SNLS_lup_solve
//
// This module was created to provide a simple, local solution for solving the stiffness
// matrices within SNLS. It was created to be used for both execution on host and device/gpu
// architectures and was incorporated into SNLS as a placeholder until a more robust solver 
// is available.
//
// -Brett Wayne-  3-Nov-2017
//-----------------------------------------------------------------------------------------------


#define SNLS_GPU_LUP_N 10

// SNLS_LUP_Fix_Columns()
//
// One of the idiosyncrasies of the MS stiffness matrices is that one or more column may 
// contain all zeros (or functionally zeros). LUP will fail if the pivot is too small. 
// This routine will check and replace zero columns with 1.0 on the diagonal.
//-----------------------------------------------------------------------------------------------

__snls_hdev__ 
void SNLS_LUP_Fix_Columns
(
   real8 **a  ,  ///< source matrix (NxN)
   int     n  ,  ///< size of the source matrix (NxN)
   real8   tol   ///< error tolerance for degeneracy test
)
{
   if (a && (n>0))
   {
      for(int i=0; (i<n); ++i)
      {
         real8 cmax = 0.0;
         for(int j=0; (j<n); ++j)
         {
            real8 absa=fabs(a[j][i]);
            cmax = ( (absa>cmax) ? absa : cmax );
         }

         if (cmax<tol)
         {
            for(int j=0; (j<n); ++j) { a[j][i]=0.0; }
            a[i][i]=1.0;
         }
      }
   }
}

// SNLS_LUP_Decompose()
//
// Performs an in-place, LU decomposition of a matrix.
//
// On exit, the source matrix is deomposed into two matrices :  L-E and U as 
// A=(L-E)+U such that P*A=L*U.  Each element of of the pivot vector contains the 
// column index where the permutation matrix has "1". The last element of the pivot 
// vector, P[N]=S+N, where S is the number of row exchanges needed for determinant 
// computation, det(P)=(-1)^S
//-----------------------------------------------------------------------------------------------

__snls_hdev__
int SNLS_LUP_Decompose
(
   real8 **a  ,  ///< source matrix (NxN)
   int    *p  ,  ///< pivot vector  (N+1)
   int     n  ,  ///< size of the source matrix (NxN)
   real8   tol   ///< error tolerance for degeneracy test
)
{
   if (a && p && (n>0) )
   {
      for(int i=0; (i<=n); ++i) { p[i]=i; }       // initialize the pivot vector
   
      for(int i=0; (i< n); ++i)
      {
         // locate max of remaining rows to pivot...

         int   imax = i;
         real8 absa = 0.0;
         real8 maxa = 0.0;
   
         for(int k=i; (k<n); ++k)
            if((absa=fabs(a[k][i]))>maxa){ maxa=absa; imax=k; }

         // if the pivot row/value is less than the provided tolerance, give up.
         // We are essentially trying to avoid a division by zero with the pivot value.
   
         if (maxa<tol) 
         {
            printf("%s::%s() ln=%d - error - pivot value too small (pivot=%0.2le < max=%0.2le)\n", __FILE__, __func__, __LINE__, maxa, tol);
            return(-1);
         } 

         if (imax!=i)  // do we need to pivot?
         {
            { int    tmp=p[i]; p[i]=p[imax]; p[imax]=tmp; } // (swap pivot)
            { real8 *tmp=a[i]; a[i]=a[imax]; a[imax]=tmp; } // (swap rows )
   
            p[n]++; // update total pivot count (for determinant)
         }
   
         for(int j=(i+1); (j<n); ++j)
         {
            a[j][i] /= a[i][i];
   
            for(int k=(i+1); (k<n); ++k)
               a[j][k] -= (a[j][i]*a[i][k]);
         }
      }
   }

   return(0);
}

// SNLS_LUP_Solve()
//
// Will solve the system Ax=b using a matrix that was decomposed using SNLS_LUP_Decompose().
// Inputs are the source decomposed matrix, pivot vector, and the right-hand-side (RHS)
// vector (b). The result is left in x.
//-----------------------------------------------------------------------------------------------

__snls_hdev__
int SNLS_LUP_Solve
(
   real8 **a,  ///< source matrix      (NxN, LUP decomposed)
   int    *p,  ///< pivot vector       (N, or N+1)
   real8  *x,  ///< computed solution  (N)
   real8  *b,  ///< rhs vector         (N)
   int     n   ///< size of the system (NxN)
)
{
   if (a && p && x && b && (n>0))
   {
      for(int i=0; (i<n); ++i)
      {
         x[i] = b[p[i]];

         for(int k=0; (k<i); ++k)
            x[i] -= a[i][k]*x[k];
      }

      for(int i=(n-1); (i>=0); i--)
      {
         for(int k=(i+1); (k<n); ++k)
            x[i] -= a[i][k]*x[k];

         x[i] = x[i]/a[i][i];
      }
   }

   return(0);
}

// SNLS_LUP_Solve()
//
// Alternate version.  Given a source matrix, performs an LU decomposition on that matrix and
// solves the system.  The original matrix is modified.
//
// Note - invoking new/malloc from a CUDA device function is horrifically inefficient. 
// The LU solver uses a traditional c++ 2D array where pointers to the matrix rows are
// used to perform the matrix lookup and assignment  mtx[row][col] where mtx[row] points
// to a dense vector of values.
//
// For small systems (e.g. N<10), I maintain local arrays on the stack rather than 
// allocate locals via malloc. This avoids hammering the local heap for small matrices.
//
// Note - this LUP solver was failing for some of the MS stiffness matrices because
// the source matrices had zero (or near zero) columns. This routine checks and fixes
// those columns before attempting to solve the system.
//-----------------------------------------------------------------------------------------------

__snls_hdev__ 
int SNLS_LUP_Solve 
(
   real8  *a  ,  ///< NxN source matrix, dense, row-major, modified on output
   real8  *x  ,  ///< computed solution vector  (N)
   real8  *b  ,  ///< rhs vector                (N)
   int     n  ,  ///< size of the source matrix (NxN)
   real8   tol   ///< error tolerance for degeneracy test
)
{
   int     err = 0;                    // default error return
   real8  *rows_lcl[SNLS_GPU_LUP_N  ];   // local row pointers (for small systems)
   int     piv_lcl [SNLS_GPU_LUP_N+1];   // local pivot vector (for small systems)

   real8 **mtx = (real8 **) ( (n<SNLS_GPU_LUP_N) ? rows_lcl : malloc((n  )*sizeof(real8 *)) );
   int    *piv = (int    *) ( (n<SNLS_GPU_LUP_N) ? piv_lcl  : malloc((n+1)*sizeof(int    )) );

   if (mtx) 
   { for (int i=0,k=0; (i<n); ++i, k+=n) mtx[i]=(a+k); }   // (init matrix row-pointers)

   if (mtx && piv && x && b && (n>0))
   {
                err = ::SNLS_LUP_Decompose(mtx,piv,n,tol);    // mtx = LU(mtx)
      if (!err) err = ::SNLS_LUP_Solve    (mtx,piv,x,b,n);    // solve for x 
   }

   if (mtx && (mtx!=rows_lcl) ) { free(mtx); }
   if (piv && (piv!=piv_lcl ) ) { free(piv); }

   return(err);
}

// SNLS_LUP_Invert()
//
// Will compute the inverse of a matrix that was decomposed using SNLS_LUP_Decompose().
// Inputs are the source decomposed matrix and pivot vector. Note that only the 
// first N elements of the pivot vector are used. Also note that the source 
// and destination matrices must be different.
//-----------------------------------------------------------------------------------------------

__snls_hdev__
void SNLS_LUP_Invert
(
   real8 **ai,   ///< inverse matrix     (NxN, result)
   real8 **a ,   ///< source matrix      (NxN, LUP decomposed)
   int    *p ,   ///< pivot vector       (N)
   int     n     ///< size of the matrix (NxN)
)
{
   if (ai && a && (ai!=a) && p && (n>0))
   {
      for(int j=0; (j<n); ++j)
      {
         for(int i=0; (i<n); ++i)
         {
            ai[i][j] = ( (p[i]==j) ? 1.0 : 0.0 );

            for(int k=0; (k<i); ++k)
               ai[i][j] -= a[i][k]*ai[k][j];
         }

         for(int i=(n-1); (i>=0); i--)
         {
            for(int k=(i+1); (k<n); ++k)
               ai[i][j] -= a[i][k]*ai[k][j];

            ai[i][j] = ai[i][j]/a[i][i];
         }
      }
   }
}

// SNLS_LUP_Determinant()
//
// Will return the determinant of a matrix that was decomposed using SNLS_LUP_Decompose().
// Inputs are the decomposed matrix and pivot vector.  Note that the pivot vector is of 
// length N+1 (not N) where the final entry of the pivot vector contains the number of 
// row exchanges that occurred in the decomposition.
// 
// (note that this routine is mainly provided for debugging the LUP solver).
//-----------------------------------------------------------------------------------------------

__snls_hdev__
real8 SNLS_LUP_Determinant
(
   real8 **a,  ///< source matrix      (NxN, LUP decomposed)
   int    *p,  ///< pivot vector       (N+1)
   int     n   ///< size of the matrix (N)
)
{
   real8 det=0.0;

   if (a && p && (n>0))
   {
      det = a[0][0];

      for(int i=1; (i<n); ++i)
         det *= a[i][i];

      det = ( ((p[n]-n)%2==0) ? det : -det );
   }

   return(det);
}

