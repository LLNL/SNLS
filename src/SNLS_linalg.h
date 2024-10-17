#pragma once

#include <math.h>
#include "SNLS_base.h"

#ifdef __snls_host_only__
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#endif

namespace snls {

const double dbl_epsmch = 2.220446049250313e-16;

/// All matrices are assummed to be ordered in row-major order
namespace linalg {

/// Dot product of two vectors
/// v1 and v2 both have lengths ndim
template <int ndim>
__snls_hdev__
inline double dotProd(const double* const v1,
                      const double* const v2)
{
    double prod = 0.0;
    for (int iN = 0; iN < ndim; ++iN) {
        prod += v1[iN] * v2[iN];
    }
    return prod;
}

// Might want a stable norm eventually
/// Takes the L2 norm of a vector
/// where vec has length ndim
template<int ndim>
__snls_hdev__
inline double norm(const double* const vec)
{
    return sqrt(dotProd<ndim>(vec, vec));
}

/// Takes the norm of the columns of a matrix
/// where the matrix has dimensions nxm
/// The values are stored in norm_vec which is of length m
template<int ndim, int mdim>
__snls_hdev__
inline void normColumn(const double* const matrix, 
                       double* const norm_vec)
{

    // Initialize this to have the squared values of the first row
    for (int iM = 0; iM < mdim; iM++) {
        norm_vec[iM] = matrix[iM] * matrix[iM];
    }

    // Accumulate the results across all remaining rows
    for (int iN = 1; iN < ndim; iN++) {
        for (int jM = 0; jM < mdim; jM++) {
            norm_vec[jM] += (matrix[iN * mdim + jM] * matrix[iN * mdim + jM]);
        }
    }

    // Calculate the norm for each column
    for (int iM = 0; iM < mdim; iM++) {
        norm_vec[iM] = sqrt(norm_vec[iM]);
    }

}

/// Outer product of two vectors
/// over writes value in supplied matrix
/// mat = vec1 \otimes vec2
/// vec1 has length ndim
/// vec2 has length mdim
/// mat has dimensions ndim x mdim
template <int ndim, int mdim>
__snls_hdev__
inline void outerProd(const double* const vec1,
                      const double* const vec2,
                      double* const mat)
{
    for (int iN = 0; iN < ndim; iN++)
    {
        for (int jM = 0; jM < mdim; jM++)
        {
            mat[iN * mdim + jM] = vec1[iN] * vec2[jM];
        }
    }
}

/// Adds a scaled outer product to supplied matrix
/// mat += \alpha * vec1 \otimes vec2
/// vec1 has length ndim
/// vec2 has length mdim
/// alpha has default value of 1
/// mat has dimensions ndim x mdim
template <int ndim, int mdim>
__snls_hdev__
inline void outerProdAddScale(const double* const vec1,
                            const double* const vec2,
                            double* const mat,
                            const double alpha = 1.0)
{
    for (int iN = 0; iN < ndim; iN++)
    {
        for (int jM = 0; jM < mdim; jM++)
        {
            mat[iN * mdim + jM] += alpha * vec1[iN] * vec2[jM];
        }
    }
}

/// Matrix vector product
/// M has dimensions ndim x mdim
/// a has dimensions mdim
/// p has dimensions ndim
template <int ndim, int mdim>
__snls_hdev__
inline void matVecMult(const double* const M,
                        const double* const a,
                        double* const p)
{
    for (int iN = 0; iN < ndim; ++iN) {
        p[iN] = 0.0;
        for (int jM = 0; jM < mdim; ++jM) {
            p[iN] += M[iN * mdim + jM] * a[jM];
        }
    }
}

/// Matrix transpose vector product
/// M has dimensions ndim x mdim
/// a has dimensions ndim
/// p has dimensions mdim
template <int ndim, int mdim>
__snls_hdev__
inline void matTVecMult(const double* const M,
                        const double* const a,
                        double* const p)
{
    for (int iM = 0; iM < mdim; ++iM) {
        p[iM] = 0.0;
        for (int jN = 0; jN < ndim; ++jN) {
            p[iM] += M[jN * ndim + iM] * a[jN];
        }
    }

}

/// Upper triangle matrix vector product
/// M is an upper triangle matrix 
/// (values below the diagonal are assummed zero)
/// M has dimensions ndim x mdim
/// a has dimensions mdim
/// p has dimensions ndim
/// ndim <= mdim
template<int ndim, int mdim>
__snls_hdev__
inline void matUTriVecMult(const double* const M,
                           const double* const a,
                           double* const p)
{
    // 
    for (int iN = 0; iN < ndim; ++iN) {
        p[iN] = 0.0;
        for (int jM = iN; jM < mdim; ++jM) {
            p[iN] += M[iN * mdim + jM] * a[jM];
        }
    }
}

/// Upper triangle matrix transpose vector product
/// M is an upper triangle matrix 
/// (values below the diagonal are assummed zero)
/// M has dimensions ndim x mdim
/// a has dimensions mdim
/// p has dimensions ndim
/// ndim <= mdim
template<int ndim, int mdim>
__snls_hdev__
inline void matUTriTVecMult(const double* const M,
                           const double* const a,
                           double* const p)
{
    // 
    for (int iN = 0; iN < ndim; ++iN) {
        p[iN] = 0.0;
        // M_ji * a_j = p_i
        // Only go down to diagonal
        for (int jM = 0; jM <= iN; ++jM) {
            p[iN] += M[jM * mdim + iN] * a[jM];
        }
    }
}

/// Matrix-matrix multiplication
/// mat1 has dimensions ldim x ndim
/// mat2 has dimensions ndim x mdim
/// prod has dimensions ldim x mdim
/// This function will either accumulate the values of
/// the multiplication on the product,
/// or it will zero out the product ahead of time
/// depedening on the run time flag.
template <int ldim, int ndim, int mdim, bool zero_out>
__snls_hdev__
inline void matMult(const double* const mat1,
                    const double* const mat2,
                    double* const prod)
{
    if (zero_out)
    {
        for (int iLM = 0; iLM < (ldim * mdim); iLM++) { prod[iLM] = 0.0; }
    }
    //prod_ik = mat1_ij * mat2_jk
    for (int iL = 0; iL < ldim; iL++)
    {
        for (int jN = 0; jN < ndim; jN++)
        {
            for (int kM = 0; kM < mdim; kM++)
            {
                prod[iL * mdim + kM] += mat1[iL * ndim + jN] * mat2[jN * mdim + kM];
            }
        }
    }
}

/// Matrix transpose-matrix multiplication
/// mat1 has dimensions ndim x ldim
/// mat2 has dimensions ndim x mdim
/// prod has dimensions ldim x mdim
/// This function will either accumulate the values of
/// the multiplication on the product,
/// or it will zero out the product ahead of time
/// depedening on the run time flag.
template <int ldim, int ndim, int mdim, bool zero_out>
__snls_hdev__
inline void matTMult(const double* const mat1,
                     const double* const mat2,
                     double* const prod)
{
    if (zero_out)
    {
        for (int iLM = 0; iLM < (ldim * mdim); iLM++) { prod[iLM] = 0.0; }
    }
    //prod_ik = mat1_ji * mat2_jk
    for (int iL = 0; iL < ldim; iL++)
    {
        for (int jN = 0; jN < ndim; jN++)
        {
            for (int kM = 0; kM < mdim; kM++)
            {
                prod[iL * mdim + kM] += mat1[jN * ldim + iL] * mat2[jN * mdim + kM];
            }
        }
    }
}

/// Matrix-matrix transpose multiplication
/// mat1 has dimensions ldim x ndim
/// mat2 has dimensions mdim x ndim
/// prod has dimensions ldim x mdim
/// This function will either accumulate the values of
/// the multiplication on the product,
/// or it will zero out the product ahead of time
/// depedening on the run time flag.
template <int ldim, int ndim, int mdim, bool zero_out>
__snls_hdev__
inline void matMultT(const double* const mat1,
                     const double* const mat2,
                     double* const prod)
{
    if (zero_out)
    {
        for (int iLM = 0; iLM < (ldim * mdim); iLM++) { prod[iLM] = 0.0; }
    }
    //prod_ik = mat1_ij * mat2_kj
    for (int iL = 0; iL < ldim; iL++)
    {
        for (int jN = 0; jN < ndim; jN++)
        {
            for (int kM = 0; kM < mdim; kM++)
            {
                prod[iL * mdim + kM] += mat1[iL * ndim + jN] * mat2[kM * ndim + jN];
            }
        }
    }
}

/// Performs the triple product operation
/// needed to rotate a matrix of ndim x ndim
/// by a rotatation matrix
/// The product matrix is zeroed out in this operation
/// Transpose does the operation:
/// prod_il = rot_ji mat_jk rot_kl
/// prod = rot^t * mat * rot
/// Non-transpose operation does:
/// prod_il = rot_ij mat_jk rot_lk
/// prod = rot * mat * rot^T
template<int ndim, bool transpose>
__snls_hdev__
inline void rotMatrix(const double* const mat,
                    const double* const rot,
                    double* const prod)
{
    // zero things out first
    for (int iNN = 0; iNN < ndim * ndim; iNN++)
    {
        prod[iNN] = 0.0;
    }
    // Now for matrix multiplication
    for (int iN = 0; iN < ndim; iN++)
    {
        for (int jN = 0; jN < ndim; jN++)
        {
            for (int kN = 0; kN < ndim; kN++)
            {
                for (int lN = 0; lN < ndim; lN++)
                {
                    // This is rot_ji mat_jk rot_kl
                    if (transpose)
                    {
                    prod[iN * ndim + lN] += rot[jN * ndim + iN] *
                                            mat[jN * ndim + kN] *
                                            rot[kN * ndim + lN];
                    }
                    // This is rot_ij mat_jk rot_lk
                    else
                    {
                    prod[iN * ndim + lN] += rot[iN * ndim + jN] *
                                            mat[jN * ndim + kN] *
                                            rot[lN * ndim + kN];
                    }
                }
            }
        }
    }
}

#ifdef __snls_host_only__
template<int n>
inline void
printVec(const double* const y,
        std::ostream & oss = std::cout) {
    for (int iX = 0; iX<n; ++iX) {
        oss << std::setw(21) << std::setprecision(14) << y[iX] << " ";
    }
    oss << std::endl;
}

template<int n>
inline void
printMat(const double* const A,
        std::ostream & oss = std::cout) {
    for (int iX = 0; iX<n; ++iX) {
        for (int jX = 0; jX<n; ++jX) {
            oss << std::setw(21) << std::setprecision(14) << A[SNLS_NN_INDX(iX, jX, n)] << " ";
        }
        oss << std::endl;
    }
}
#endif

} // End of linalg namespace
} // End of snls namespace