// -*-c++-*-
#ifndef __SNLS_LUP_SOLVE_H
#define __SNLS_LUP_SOLVE_H

#include <stdio.h>
#include <math.h>

#include "SNLS_gpu_portability.h"
#include "SNLS_port.h"

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

// SNLS_LUP_Fix_Columns()
//
// One of the idiosyncrasies of the MS stiffness matrices is that one or more column may 
// contain all zeros (or functionally zeros). LUP will fail if the pivot is too small. 
// This routine will check and replace zero columns with 1.0 on the diagonal.
//-----------------------------------------------------------------------------------------------

template<int n> // size of the system (NxN)
__snls_hdev__ 
void SNLS_LUP_Fix_Columns
(
   double ** const a  ,  ///< source matrix (NxN)
   double          tol   ///< error tolerance for degeneracy test
)
{
   if (a && (n>0))
   {
      for(int i=0; (i<n); ++i)
      {
         double cmax = 0.0;
         for(int j=0; (j<n); ++j)
         {
            double absa=fabs(a[j][i]);
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

template<int n> // size of the system (NxN)
__snls_hdev__
int SNLS_LUP_Decompose
(
   double** const a  ,  ///< source matrix (NxN)
   int* const     p  ,  ///< pivot vector  (N+1)
   double         tol   ///< error tolerance for degeneracy test
)
{
   if (a && p && (n>0) )
   {
      for(int i=0; (i<=n); ++i) { p[i]=i; }       // initialize the pivot vector
   
      for(int i=0; (i< n); ++i)
      {
         // locate max of remaining rows to pivot...

         int   imax = i;
         double absa = 0.0;
         double maxa = 0.0;
   
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
            { double *tmp=a[i]; a[i]=a[imax]; a[imax]=tmp; } // (swap rows )
   
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

template<int n> // size of the system (NxN)
__snls_hdev__
int SNLS_LUP_Solve
(
         double** const a,  ///< source matrix      (NxN, LUP decomposed)
         int*     const p,  ///< pivot vector       (N, or N+1)
         double*  const x,  ///< computed solution  (N)
   const double*  const b   ///< rhs vector         (N)
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

   return(0); // make return code always-positive, so that can add them
}

// SNLS_LUP_Solve()
//
// Alternate version.  Given a source matrix, performs an LU decomposition on that matrix and
// solves the system.  The original matrix is modified.
//-----------------------------------------------------------------------------------------------

template<int n> // size of the system (NxN)
__snls_hdev__ 
int SNLS_LUP_Solve 
(
         double* const a  ,        ///< NxN source matrix, dense, row-major, modified on output
         double* const x  ,        ///< computed solution vector  (N)
   const double* const b  ,        ///< rhs vector                (N)
         double        tol=1e-50   ///< error tolerance for degeneracy test
)
{
   int      err = 0  ;   // default error return
   double  *mtx[n  ] ;   // local row pointers
   int      piv[n+1] ;   // local pivot vector

   { for (int i=0,k=0; (i<n); ++i, k+=n) mtx[i]=(a+k); }   // (init matrix row-pointers)

   if (x && b && (n>0))
   {
                err = ::SNLS_LUP_Decompose<n>(mtx,piv,tol);    // mtx = LU(mtx)
      if (!err) err = ::SNLS_LUP_Solve<n>    (mtx,piv,x,b);    // solve for x 
   }

   return(err);
}

// multi-right-hand-side version of SNLS_LUP_SolveX
// NOTES :
//	() xb is stored with entries in a given RHS indexing fastest
// 	() on entry xb is rhs vectors, on exit it is solutions
template<int n> // size of the system (NxN)
__snls_hdev__ 
int SNLS_LUP_SolveX
(
   double* const a    ,     ///< NxN source matrix, dense, row-major, modified on output
   double* const xb   ,     ///< rhs and solution vectors  (nRHS x N)
   int           nRHS ,
   double        tol=1e-50  ///< error tolerance for degeneracy test
)
{
   int      err = 0;    // default error return
   double  *mtx[n  ];   // local row pointers 
   double   wrk[n  ];   // local workspace 
   int      piv[n+1];   // local pivot vector 

   { for (int i=0,k=0; (i<n); ++i, k+=n) mtx[i]=(a+k); }   // (init matrix row-pointers)

   if (xb && (n>0))
   {
      err = ::SNLS_LUP_Decompose<n>(mtx,piv,tol);    // mtx = LU(mtx)
      if (!err) {
         for (int iRHS=0; iRHS<nRHS; ++iRHS) {
            double* xThis = &(xb[iRHS*n]);
            for (int iX=0; iX<n; ++iX) {
               wrk[iX] = xThis[iX];
            }
            err += ::SNLS_LUP_Solve<n>(mtx,piv,xThis,wrk);
         }
      }
   }

   return(err);
}

#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)
#include "RAJA/RAJA.hpp"
#include "SNLS_unused.h"

/**
 * @file
 * @brief Cooperative (team-of-threads) LU solve overloads.
 *
 * Everything above this point in this file is the original, always-serial
 * LU solve: one thread does an entire factorization/solve alone. This
 * block adds new, additively-overloaded entry points -- distinguished by
 * extra trailing `tid`/`nthreads` parameters -- that instead let a *team*
 * of `nthreads` cooperating threads factor/solve one matrix together
 * (MAGMA-style), reached through snls::forall_team(). None of the
 * existing signatures above change; these are pure additions.
 *
 * **The caller-provided-scratch pattern**: every buffer these overloads
 * touch (`a`, and the pivot vector) is allocated and populated by the
 * *caller*, not by these functions. This is not a style preference -- it
 * is required for correctness. `RAJA_TEAM_SHARED` (`__shared__` on
 * device) memory is scoped to the *physical CUDA/HIP block*, not to any
 * one logical "team" within it. When `snls::forall_team` packs several
 * independent teams into one block (its `itemsPerBlock` > 1 case), a
 * buffer declared *inside* one of these functions would be silently
 * shared -- and corrupted -- by every co-resident team, since there is no
 * hardware notion of "shared memory private to my team." Only the caller
 * knows which `teamBase`-indexed slot of a larger, block-wide buffer
 * belongs to its own team (e.g. `&Jshared[teamBase][0][0]`); by the time
 * that resolved pointer reaches these functions, it is already safe to
 * treat as if it belonged to this team alone.
 *
 * **teamSync() is block-wide, not team-wide.** RAJA::LaunchContext::
 * teamSync() compiles to `__syncthreads()`, which unconditionally
 * synchronizes every thread in the physical block, not just the
 * `nthreads` cooperating on one matrix. Every function below is written
 * so that every thread in a team calls it the same fixed number of times
 * regardless of data (masking gates *work*, never a teamSync() call
 * itself -- see the SNLS_LUP_Decompose contract note below); a caller
 * that invokes one of these functions a *data-dependent* number of times
 * per thread (e.g. inside its own per-thread loop) would reintroduce
 * exactly the divergent-barrier-count hazard this design exists to avoid.
 * SNLS_LUP_SolveX's per-right-hand-side loop below is a worked example of
 * how to stay safe under that constraint (it deliberately calls the
 * plain serial substitution, which has no teamSync() of its own, once per
 * right-hand side rather than a cooperative overload that does).
 *
 * @see snls::forall_team and snls::TeamActivityConsensus in
 *      SNLS_device_forall.h for the surrounding machinery these overloads
 *      are meant to be used from.
 */

/**
 * @brief Cooperative in-place LU decomposition: `nthreads` threads factor
 * one matrix together instead of one thread doing it alone.
 *
 * Mirrors the plain SNLS_LUP_Decompose() above algorithmically -- same
 * partial-pivoting, right-looking elimination, same pivot vector
 * convention -- but the elimination step (the O(n^3) part) is split across
 * `nthreads` threads by row, and every step is separated by a
 * RAJA::LaunchContext::teamSync() so the team agrees on the matrix's state
 * before the next thread's read of it.
 *
 * **Division of labor**: the pivot search and row swap at each step are
 * done by `tid==0` alone (O(n) total across the whole decomposition,
 * negligible next to the O(n^3) elimination -- not worth parallelizing).
 * The elimination itself parallelizes cleanly because, at a fixed pivot
 * step `i`, every row `j>i`'s update depends only on the
 * already-finalized pivot row `i`, never on any other row being updated
 * concurrently -- so `for (int j = i+1+tid; j<n; j+=nthreads)` is safe
 * with no further synchronization needed until the next pivot step.
 *
 * **On a genuinely singular matrix** (as opposed to a caller's
 * deliberately masked-off/dummy-seeded point, which is expected to look
 * well-conditioned and never trigger this): every thread in the team
 * independently recomputes `fabs(a[i*n+i]) < tol` against the same,
 * already-synchronized pivot value, so every thread reaches the identical
 * conclusion without needing any additional shared "did we fail" flag.
 * Once true, later steps skip their elimination work (nothing left to
 * safely divide by) but the loop itself still runs its full `n`
 * iterations and both teamSync() calls per iteration, for every thread,
 * unconditionally -- exactly the masked-work-not-masked-control-flow
 * contract snls::forall_team's body itself must uphold, applied one level
 * deeper, inside this function's own loop. No diagnostic is printed here
 * (unlike the plain serial version above): a masked-off point's
 * caller-seeded dummy identity is expected to never trigger this path,
 * and a real failure is reported through the return value instead, for
 * the caller to act on (see the masking contract on snls::forall_team).
 *
 * @tparam n the size of the system (n x n).
 *
 * @param[in,out] a        n*n, row-major, team-shared scratch, already
 *                         populated by the caller (masked-off points
 *                         seeded with a well-conditioned dummy, e.g. the
 *                         identity, using the same tid-strided pattern
 *                         this function uses internally -- see the
 *                         caller-provided-scratch note above). Decomposed
 *                         in place on return.
 * @param[out]    p        n+1, team-shared scratch, owned by the caller
 *                         the same way `a` is. Initialized by this
 *                         function; holds the row-permutation and, in
 *                         `p[n]`, the total swap count (for determinant
 *                         sign, matching the plain serial version's
 *                         convention, even though SNLS_LUP_Determinant
 *                         itself is untouched by this work).
 * @param[in]     tol      error tolerance for the degeneracy test.
 * @param[in]     tid      this thread's role within its own team,
 *                         0..nthreads-1.
 * @param[in]     nthreads how many threads, total, are cooperating on this
 *                         one decomposition.
 *
 * @return 0 on success, or a negative value if a genuinely singular pivot
 *         (not a masked-off dummy) was encountered -- identical to every
 *         thread in the team, computed independently rather than
 *         broadcast, per the note above.
 *
 * @note This function's own teamSync() count is a fixed `2 + 2*n` calls,
 *       identical for every thread regardless of `tid` or whether a
 *       singular pivot was hit -- safe to call exactly once per team, but
 *       NOT safe to call a data-dependent number of times per thread (see
 *       the file-level note on teamSync() being block-wide).
 */
template<int n>
__snls_hdev__
int SNLS_LUP_Decompose
(
   double* const a       ,
   int*    const p       ,
   double        tol     ,
   int           tid     ,
   int           nthreads
)
{
   // The caller has already written `a`'s rows into this team-shared
   // scratch (masked/dummy-seeded as needed) before calling us -- this
   // teamSync() is what makes that write visible to every thread before
   // anyone reads `a` for the first pivot search.
   RAJA::LaunchContext{}.teamSync();

   if (tid == 0) {
      for (int i = 0; i <= n; ++i) { p[i] = i; }
   }
   RAJA::LaunchContext{}.teamSync();

   bool singular = false;
   for (int i = 0; i < n; ++i) {
      if (tid == 0) {
         int imax = i;
         double maxa = 0.0;
         for (int k = i; k < n; ++k) {
            double absa = fabs(a[k*n + i]);
            if (absa > maxa) { maxa = absa; imax = k; }
         }
         if (imax != i) {
            for (int c = 0; c < n; ++c) {
               double tmp    = a[i*n + c];
               a[i*n + c]    = a[imax*n + c];
               a[imax*n + c] = tmp;
            }
            { int tmp = p[i]; p[i] = p[imax]; p[imax] = tmp; }
            p[n]++; // total pivot count, for determinant sign
         }
      }
      RAJA::LaunchContext{}.teamSync();

      // Every thread reads the same, already-synchronized pivot value, so
      // every thread reaches the identical verdict independently -- no
      // extra shared state needed to broadcast it (see the doc comment
      // above).
      if (!singular && fabs(a[i*n + i]) < tol) {
         singular = true;
      }
      if (!singular) {
         for (int j = i + 1 + tid; j < n; j += nthreads) {
            double factor = a[j*n + i] / a[i*n + i];
            a[j*n + i] = factor;
            for (int k = i + 1; k < n; ++k) {
               a[j*n + k] -= factor * a[i*n + k];
            }
         }
      }
      RAJA::LaunchContext{}.teamSync();
   }

   return singular ? -1 : 0;
}

/**
 * @brief Cooperative forward/back substitution, given a matrix already
 * decomposed by the cooperative SNLS_LUP_Decompose() above.
 *
 * Deliberately kept single-threaded (`tid==0` only), unlike elimination.
 * `x[i]` has a strict sequential dependency on every `x[k]` for `k<i`
 * (forward) or `k>i` (backward) -- there is no row-independence to
 * exploit the way elimination has at a fixed pivot step, so there is
 * nothing here for other threads to do concurrently. It would be
 * possible to extract some parallelism from the inner dot product of each
 * individual `x[i]`'s computation, but that would trade elimination's
 * O(n) teamSync() calls guarding O(n^3) work for O(n) *new* teamSync()
 * calls guarding only O(n) *new* work (a far worse ratio) -- not worth it
 * for the small systems SNLS targets, where substitution (O(n^2)) is
 * already asymptotically cheap next to elimination (O(n^3)).
 *
 * @tparam n the size of the system (n x n).
 *
 * @param[in]  a        n*n, row-major, team-shared scratch, already
 *                       decomposed (by SNLS_LUP_Decompose() above).
 * @param[in]  p        n (or n+1), team-shared pivot vector from
 *                       SNLS_LUP_Decompose().
 * @param[out] x        n, the computed solution. Written by `tid==0`
 *                       only -- the caller must gate any further use of
 *                       `x` by threads other than `tid==0` behind its own
 *                       teamSync(), exactly as it must already gate any
 *                       other team-cooperative write.
 * @param[in]  b        n, the right-hand-side vector.
 * @param[in]  tid      this thread's role within its own team,
 *                       0..nthreads-1.
 * @param[in]  nthreads how many threads are cooperating on this call (not
 *                       otherwise used, since substitution itself is not
 *                       parallelized -- accepted for a uniform calling
 *                       convention with the other cooperative overloads).
 *
 * @return 0, always -- forward/back substitution against an
 *         already-decomposed matrix cannot itself fail (matching the
 *         plain serial version above, which returns 0 unconditionally
 *         too, precisely so multiple calls' return values can be summed).
 *
 * @note Exactly one teamSync() call, at entry -- safe to call once per
 *       team; like SNLS_LUP_Decompose, NOT safe to call a data-dependent
 *       number of times per thread.
 */
template<int n>
__snls_hdev__
int SNLS_LUP_Solve
(
         double* const a       ,
         int*    const p       ,
         double* const x       ,
   const double* const b       ,
         int           tid     ,
         int           UNUSED(nthreads)
)
{
   RAJA::LaunchContext{}.teamSync(); // after decompose, before substitution

   if (tid == 0) {
      for (int i = 0; i < n; ++i) {
         x[i] = b[p[i]];
         for (int k = 0; k < i; ++k) { x[i] -= a[i*n + k]*x[k]; }
      }
      for (int i = n - 1; i >= 0; --i) {
         for (int k = i + 1; k < n; ++k) { x[i] -= a[i*n + k]*x[k]; }
         x[i] = x[i] / a[i*n + i];
      }
   }

   return 0;
}

/**
 * @brief Cooperative composite solve: decomposes `a` in place, then
 * solves, using a team of `nthreads` cooperating threads -- the
 * cooperative counterpart of the plain composite SNLS_LUP_Solve() above,
 * and the one both SNLSTrDlDenseG and SNLSTrDlDenseG_Batch actually call.
 *
 * @tparam n the size of the system (n x n).
 *
 * @param[in,out] a        n*n, row-major, team-shared scratch. Modified
 *                         (decomposed) in place, exactly like the plain
 *                         composite overload above.
 * @param[out]    piv      n+1, team-shared pivot-vector scratch. Unlike
 *                         the plain composite overload above (which owns
 *                         its pivot vector as a private local array), this
 *                         cooperative overload cannot safely allocate its
 *                         own: any buffer it declared would be
 *                         block-shared, not team-private, and so would be
 *                         corrupted by every other team co-resident in
 *                         the same block (see the file-level note above).
 *                         The caller must supply one, sized/indexed the
 *                         same way it already sizes/indexes `a`.
 * @param[out]    x        n, the computed solution.
 * @param[in]     b        n, the right-hand-side vector.
 * @param[in]     tid      this thread's role within its own team,
 *                         0..nthreads-1.
 * @param[in]     nthreads how many threads are cooperating on this solve.
 * @param[in]     tol      error tolerance for the degeneracy test.
 *
 * @return 0 on success, negative on a genuinely singular matrix (see
 *         SNLS_LUP_Decompose() above).
 *
 * @note `tol` has no default here, unlike the plain composite overload
 *       above -- deliberately, so that a call with exactly the 2-array
 *       cooperative SNLS_LUP_Solve() overload's argument count
 *       (`a, piv/p, x, b, tid, nthreads`) can never be ambiguous between
 *       the two overloads. Adding `piv` to this composite overload (a
 *       correctness requirement, per the file-level note above) would
 *       otherwise make its argument list, with `tol` defaulted, a prefix
 *       of the 2-array overload's -- and C++ treats that as ambiguous at
 *       the exact argument count where both would apply, not as
 *       preferring the more specific match.
 */
template<int n>
__snls_hdev__
int SNLS_LUP_Solve
(
         double* const a       ,
         int*    const piv     ,
         double* const x       ,
   const double* const b       ,
         int           tid     ,
         int           nthreads,
         double        tol
)
{
   int err = SNLS_LUP_Decompose<n>(a, piv, tol, tid, nthreads);
   if (!err) { err = SNLS_LUP_Solve<n>(a, piv, x, b, tid, nthreads); }
   return err;
}

/**
 * @brief Cooperative multi-right-hand-side solve: decomposes `a` once,
 * cooperatively, then solves for `nRHS` right-hand sides -- the
 * cooperative counterpart of the plain SNLS_LUP_SolveX() above. Not
 * currently called by either SNLS solver, but added alongside the other
 * cooperative overloads as a real, standalone use case.
 *
 * **Why the right-hand sides are split across threads, not the
 * substitution algorithm itself**: unlike one right-hand side's
 * substitution (strictly sequential within itself -- see the cooperative
 * 2-array SNLS_LUP_Solve() above), solving `A*x_1=b_1, ..., A*x_k=b_k`
 * against the same already-factored `A` has no dependency *across*
 * right-hand sides at all. So this distributes whole right-hand-side
 * vectors across the team's threads, each doing its own complete,
 * ordinary serial substitution for its own disjoint subset -- not by
 * calling the cooperative 2-array SNLS_LUP_Solve() overload above (which
 * contains its own teamSync(), and would therefore be called a
 * data-dependent number of times per thread whenever `nRHS` is not a
 * multiple of `nthreads` -- precisely the block-wide-barrier-count hazard
 * described at the top of this file), but the plain, always-serial,
 * teamSync()-free row-pointer SNLS_LUP_Solve() from above this cooperative
 * block, which carries no such restriction and needs no per-call
 * synchronization at all.
 *
 * @tparam n the size of the system (n x n).
 *
 * @param[in,out] a        n*n, row-major, team-shared scratch. Decomposed
 *                         in place.
 * @param[out]    piv      n+1, team-shared pivot-vector scratch,
 *                         caller-provided for the same reason as the
 *                         composite SNLS_LUP_Solve() above.
 * @param[in,out] xb       nRHS*n, right-hand-side vectors on entry,
 *                         solution vectors on exit, one right-hand side's
 *                         indexing fastest.
 * @param[in]     nRHS     how many right-hand sides to solve for.
 * @param[in]     tid      this thread's role within its own team,
 *                         0..nthreads-1.
 * @param[in]     nthreads how many threads are cooperating -- both
 *                         smaller and larger than `nRHS` are handled
 *                         correctly, exactly like the `NTEAM`-vs-matrix-
 *                         dimension striding elsewhere in this design.
 * @param[in]     tol      error tolerance for the degeneracy test.
 *
 * @return 0 on success, negative on a genuinely singular matrix -- from
 *         SNLS_LUP_Decompose() alone; the per-right-hand-side substitution
 *         step cannot itself fail (see the 2-array SNLS_LUP_Solve() above),
 *         so it contributes nothing to this return value.
 */
template<int n>
__snls_hdev__
int SNLS_LUP_SolveX
(
   double* const a       ,
   int*    const piv     ,
   double* const xb      ,
   int           nRHS    ,
   int           tid     ,
   int           nthreads,
   double        tol
)
{
   int err = SNLS_LUP_Decompose<n>(a, piv, tol, tid, nthreads);
   RAJA::LaunchContext{}.teamSync(); // after decompose, before substitution

   if (!err) {
      double* mtx[n];
      { for (int i = 0, k = 0; i < n; ++i, k += n) { mtx[i] = a + k; } }

      for (int r = tid; r < nRHS; r += nthreads) {
         double* xThis = &xb[r*n];
         double  wrk[n];
         for (int iX = 0; iX < n; ++iX) { wrk[iX] = xThis[iX]; }
         ::SNLS_LUP_Solve<n>(mtx, piv, xThis, wrk);
      }
   }

   return err;
}

#endif // SNLS_RAJA_PORT_SUITE || SNLS_RAJA_ONLY

// SNLS_LUP_Invert()
//
// Will compute the inverse of a matrix that was decomposed using SNLS_LUP_Decompose().
// Inputs are the source decomposed matrix and pivot vector. Note that only the 
// first N elements of the pivot vector are used. Also note that the source 
// and destination matrices must be different.
//-----------------------------------------------------------------------------------------------

template<int n> // size of the system (NxN)
__snls_hdev__
void SNLS_LUP_Invert
(
   double** const ai,   ///< inverse matrix     (NxN, result)
   double** const a ,   ///< source matrix      (NxN, LUP decomposed)
   int*     const p     ///< pivot vector       (N)
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

template<int n> // size of the system (NxN)
__snls_hdev__
double SNLS_LUP_Determinant
(
   double** const a,  ///< source matrix      (NxN, LUP decomposed)
   int*     const p   ///< pivot vector       (N+1)
)
{
   double det=0.0;

   if (a && p && (n>0))
   {
      det = a[0][0];

      for(int i=1; (i<n); ++i)
         det *= a[i][i];

      det = ( ((p[n]-n)%2==0) ? det : -det );
   }

   return(det);
}

#endif  // __SNLS_LUP_SOLVE_H
