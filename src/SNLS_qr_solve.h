#pragma once

#include "SNLS_base.h"
#include "SNLS_linalg.h"

#include <stdlib.h>
#include <iostream>
#ifdef __snls_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

namespace snls {

// Need to add an actual QR solver in here at some point. It currently contains just QR factorization functions

// Algorithm taken from Matrix Computation 4th ed. by GH Golub and CF Van Loan
// pg. 236 alg. 5.1.1
__snls_hdev__
inline
void householderVec(double& beta,
                    double* const nu,
                    const double *const x,
                    const int length) 
{
    double sigma = 0.0;
    for (int i = 1; i < length; i++) {
        sigma += x[i] * x[i];
        nu[i] = x[i];
    }

    nu[0] = 1.0;

    if (sigma == 0.0 && x[0] >= 0.0) {
        beta = 0.0;
    }
    else if( sigma == 0.0 && x[0] < 0.0) {
        beta = 0.0;
    }
    else {
        const double mu = sqrt(x[0] * x[0] + sigma);
        if (x[0] >= 0.0) {
            nu[0] = x[0] - mu;
        }
        else {
            nu[0] = -sigma / (x[0] + mu);
        }
        // Sometimes nu[0] is pratically zero which can cause issues when inverting it.
        // Therefore, we want to multiply it by some small number.
        // Here we choose double machine precision. However, we might need to improve this
        // in the future and copy the scaling scheme in LAPACK
        if (fabs(nu[0]) < snls::dbl_epsmch) {
            nu[0] = snls::dbl_epsmch;
        }
        beta = (2.0 * nu[0] * nu[0]) / (sigma + nu[0] * nu[0]);
        const double inu0 = 1.0 / nu[0];
        nu[0] = 1.0;
        for (int i = 1; i < length; i++) {
            nu[i] *= inu0;
        }
    }
}

// Algorithm taken from Matrix Computation 4th ed. by GH Golub and CF Van Loan
// pg. 238 alg 5.1.5 with the slight modification that we use the beta values
// saved off earlier.
template<int m>
__snls_hdev__
inline
void householderQMat(double *const Q,
                     const double *const matFac, 
                     const double *const betav)
{
    // We need two working arrays for the
    // nu^T Q product and nu.
    double nu[m] = { 0.0 };
    double nuTQ[m] = { 0.0 };
    for (int i = 0; i < m * m; i++) {
        Q[i] = 0.0;
    }
    // Initialize Q to be an identity matrix
    for (int i = 0; i < m; i++) {
        Q[SNLS_NN_INDX(i, i, m)] = 1.0;
    }

    for (int i = m - 1; i > -1; i--){
        for (int j = 0; j < (m - i); j++) {
            nu[j] = matFac[SNLS_NN_INDX(j + i, i, m)];
        }
        nu[0] = 1.0;
        const double beta = betav[i];

        // nuTQ will contain the product nu.T * Q[i:m, i:m]
        // It has dimensions of 1xn
        // We need to zero it first
        for (int k = 0; k < (m - i); k++) {
            nuTQ[k] = 0;
        }
        // nu.T * Q[i:m, i:m]
        for (int j = 0; j < (m - i); j++) {
            for (int k = 0; k < (m - i); k++) {
                // prod_nu_Q_k = nu_j q_jk
                nuTQ[k] += nu[j] * Q[SNLS_NN_INDX(i + j, i + k, m)];
            }
        }
        // Now Q[i:m, i:m]  -= (betav[i] * nu) \otimes nuTQ
        // Q_jk -= betav[i] * nu_j * nuTQ_k
        for (int j = 0; j < (m - i); j++) {
            for (int k = 0; k < (m - i); k++) {
                Q[SNLS_NN_INDX(i + j, i + k, m)] -= beta * nu[j] * nuTQ[k]; 
            }
        }
    }
}

// Algorithm taken from Matrix Computation 4th ed. by GH Golub and CF Van Loan
// pg. 249
// Here we return the householder R matrix in the input matrix
// and Q within the supplied Q matrix. Q should have dimensions
// mxm and R should have dimensions of mxn where m >= n.
// For our use cases in SNLS m will always equal n.
// We'll also supply 3 working arrays of length n.
template<int m, int n>
__snls_hdev__
inline
void householderQR(double *const matrix,
                   double *const Q,
                   double *const work_arr1,
                   double *const work_arr2,
                   double *const work_arr3)
{
    for (int i = 0; i < n; i++) {
        // Work arrary 1 is the beta values
        // work array 2 are the nu vector which have length m - i
        // The householder vector is
        for (int j = 0; j < (m - i); j++) {
            work_arr3[j] = matrix[SNLS_NM_INDX(i + j, i, n, m)];
        }
        householderVec(work_arr1[i], work_arr2, work_arr3, (m - i));
        // work array 3 will contain the product nu.T * matrix[i:m, i:n]
        // It has dimensions of 1xn
        // We need to zero it first
        for (int k = 0; k < (n - i); k++) {
            work_arr3[k] = 0;
        }
        for (int j = 0; j < (m - i); j++) {
            for (int k = 0; k < (n - i); k++) {
                // prod_nu_matrix_k = nu_j matrix_jk
                work_arr3[k] += work_arr2[j] * matrix[SNLS_NM_INDX(i + j, i + k, n, m)];
            }
        }
        // Now matrix[i:m, i:n]  -= (beta[i] * nu) \otimes work_arr3
        // matrix_jk -= beta[i] * nu_j * work_arr3_k
        for (int j = 0; j < (m - i); j++) {
            for (int k = 0; k < (n - i); k++) {
                matrix[SNLS_NM_INDX(i + j, i + k, n, m)] -= work_arr1[i] * work_arr2[j] * work_arr3[k]; 
            }
        }
        if (i < m) {
            // Save off our nu[1:(m - i)] values to the lower triangle parts of the matrix
            // This won't run for the last element in the array as nu is a single value and 
            // has the trivial 1 value.
            for (int j = 1; j < (m - i); j++) {
                matrix[SNLS_NM_INDX(i + j, i, n, m)] = work_arr2[j];
            }
        }
    }
    // Now back out the Q array. Although, we could maybe do this inline with the above
    // if we thought about this a bit more.
    householderQMat<m>(Q, matrix, work_arr1);
}

__snls_hdev__
inline
void makeGivens(double* const givens,
                const double p,
                const double q) 
{
    if (q == 0.0) {
        givens[0] = (p < 0.0) ? -1.0 : 1.0;
        givens[1] = 0.0;
    }
    else if ( p == 0.0) {
        givens[0] = 0.0;
        givens[1] = (q < 0.0) ? 1.0 : -1.0;
    }
    else if (fabs(p) > fabs(q)) {
        const double t = q / p;
        const double fac = (p < 0) ? -1.0 : 1.0;
        const double u = fac * sqrt(1.0 + t * t);
        givens[0] = 1.0 / u;
        givens[1] = -t * givens[0];
    }
    else {
        const double t = p / q;
        const double fac = (q < 0) ? -1.0 : 1.0;
        const double u = fac * sqrt(1.0 + t * t);
        givens[1] = -1.0 / u;
        givens[0] = -t * givens[1];
    }
}

}