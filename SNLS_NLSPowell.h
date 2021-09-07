#pragma once

#include "SNLS_base.h"
#include "SNLS_qr_solve.h"
#include "SNLS_linalg.h"

#include <stdlib.h>
#include <iostream>
#ifdef __cuda_host_only__
#include <string>
#include <sstream>
#include <iomanip>
#endif

namespace snls {

/** Helper templates to ensure compliant CRJ implementations */
template<typename CRJ, typename = void>
struct nls_has_valid_computeRJ : std::false_type { static constexpr bool value = false;};

template<typename CRJ>
struct nls_has_valid_computeRJ <
   CRJ,typename std::enable_if<
       std::is_same<
           decltype(std::declval<CRJ>().computeRJ(std::declval<double* const>(), std::declval<double* const>(),std::declval<const double *>())),
           bool  
       >::value
       ,
       void
   >::type
>: std::true_type { static constexpr bool value = true;};

template<typename CRJ, typename = void>
struct nls_has_ndim : std::false_type { static constexpr bool value = false;};

template<typename CRJ>
struct nls_has_ndim <
   CRJ,typename std::enable_if<
       std::is_same<
           decltype(CRJ::nDimSys),
           const int  
       >::value
       ,
       void
   >::type
>: std::true_type { static constexpr bool value = true;};

// Hybrid Jacobian method from MINPACK which is a modified Powell method
// that makes use of the dogleg method and a rank-1 update of the jacobian
// using QR factorizations for a dense general Jacobian matrix
//
// CRJ should :
// 	have member function
// 		     __snls_hdev__ bool computeRJ( double* const r, double* const J, const double* const x ) ;
// 		computeRJ function returns true for successful evaluation
// 		TODO ... J becomes a RAJA::View ?
//	have trait nDimSys
//
template< class CRJ, bool scaleDiag = false >
class SNLSHybrdDenseG 
{
    public:
    static_assert(nls_has_valid_computeRJ<CRJ>::value, "The CRJ implementation in SNLSHybrdDenseG needs to implement bool computeRJ( double* const r, double* const J, const double* const x )");
    static_assert(nls_has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");

    // constructor
    __snls_hdev__ SNLSHybrdDenseG(CRJ &crj) :
                m_crj(crj),
                m_fevals(0), m_nIters(0), m_jevals(0), 
                m_maxfev(1000), m_outputLevel(0),
                m_delta(1e8), m_res(1e20),
                m_os(nullptr),
                m_status(unConverged)
                {
                };
    // destructor
    __snls_hdev__ ~SNLSHybrdDenseG()
    {
#ifdef __cuda_host_only__
        if ( m_outputLevel > 1 && m_os != nullptr ) {
            *m_os << "Function and Jacobian evaluations: " << m_fevals << " " << m_jevals << std::endl;
        }
#endif
    };
                
    __snls_hdev__ 
    int getNDim() const { return m_nDim; }
    __snls_hdev__ 
    int getNFEvals() const { return m_fevals; }
    __snls_hdev__ 
    int getNJEvals() const { return m_jevals; }
    __snls_hdev__ 
    double getDelta() const { return m_delta; }
    __snls_hdev__ 
    double getRes() const { return m_res; }

    // setX can be used to set the initial guess
    __snls_hdev__ 
    inline 
    void setX( const double* const x ) 
    {
        for (int iX = 0; iX < m_nDim; ++iX) {
            m_x[iX] = x[iX] ;
        }
    };

    __snls_hdev__ 
    inline 
    void getX( double* const x ) const 
    {
        for (int iX = 0; iX < m_nDim; ++iX) {
            x[iX] = m_x[iX] ;
        }
    };   

    /**
     * Must call setupSolver before calling solve
    */
    __snls_hdev__ 
    inline
    void setupSolver(int maxIter,
                     double x_tolerance,
                     double f_tolerance,
                     double factor = 100.0,
                     int outputLevel = 0) 
    {
        m_status = unConverged;
        m_fevals = 0;
        m_jevals = 0;
        m_factor = factor;

        m_maxIter = maxIter;
        m_maxfev = maxIter;
        m_x_tolerance = x_tolerance;
        m_f_tolerance = f_tolerance;

        this->setOutputlevel( outputLevel );

        m_complete = true;

    }

    __snls_hdev__ 
    void setOutputlevel( int outputLevel ) {
        m_outputLevel = outputLevel;
        m_os          = nullptr;
        //
#ifdef __cuda_host_only__
        if ( m_outputLevel > 0 ) {
            m_os = &(std::cout);
        }
#endif
    }

    // solve returns status
    //
    // on exit, _res is consistent with _x
    __snls_hdev__ 
    inline
    SNLSStatus_t solve() {
    
        if ( !m_complete) {
        SNLS_FAIL("solve", "Setup not called beforehand!") ;
        }

        double residual[m_nDim];
        double Jacobian[m_nXnDim];

        // Working arrays
        double Qmat[m_nXnDim];
        double work_arr1[m_nDim];
        double work_arr2[m_nDim];
        double work_arr3[m_nDim];
        double qtf[m_nDim];
        // Only use diag if we need it
        double diag[m_nDim];

        if (!scaleDiag) {
            for (int i = 0; i < m_nDim; i++) {
                diag[i] = 1.0;
            }
        }

        m_status = solveInit(residual);
        if (m_status != unConverged) {
            return m_status;
        }

        while (m_status == unConverged && m_nIters < m_maxIter && m_fevals < m_maxIter) {
            solveStep(residual, Jacobian, Qmat, diag, qtf, work_arr1, work_arr2, work_arr3);
        }

        return m_status;
    }

    // convenience wrapper, for the current _x
    __snls_hdev__ 
    inline bool computeRJ(double* const r,
                          double* const J ) 
    {
        bool retval = this->m_crj.computeRJ(r, J, m_x);
        // Add the finite difference portion of things back in here at a later point        
        return retval;  
    }

    private:

    __snls_hdev__
    inline
    SNLSStatus_t solveInit(double* const residual)
    {
        m_jevals = 0;
        m_fevals = 1; 
        m_status = snls::unConverged;
        // at m_x
        bool rjSuccess = this->computeRJ(residual, nullptr);
        if ( !(rjSuccess) ) {
            m_status = initEvalFailure;
            return m_status;
        }
        
        
        m_res = snls::linalg::norm<m_nDim>(residual);
        
        // initialize iteration counter and monitors
        m_nIters = 1;
        m_ncsuc = 0;
        m_ncfail = 0;
        m_nslow1 = 0;
        m_nslow2 = 0;
        return m_status;
    }

    __snls_hdev__
    inline
    SNLSStatus_t solveStep(double* const residual,
                           double* const Jacobian,
                           double* const Q,
                           double* const diag,
                           double* const qtf,
                           double* const work_arr1,
                           double* const work_arr2,
                           double* const work_arr3)
    {
        const double epsmch = snls::dbl_epsmch;
        bool jeval = true;
        
        // at m_x
        // We currently don't check on the success of this...
        // and we should at this back in at some point
        // However in the mean time, this is to quite the compiler
        // bool rjSuccess = 
        this->computeRJ(residual, Jacobian);
        // Need to think about what to do if this fails as things
        // are handled differently from the TrDL solver.
        // if ( !(rjSuccess) ) {
        //     m_status = initEvalFailure;
        //     return m_status;
        // }
        m_jevals += 1;


        snls::linalg::normColumn<m_nDim, m_nDim>(Jacobian, work_arr2);

        if(m_nIters == 1) {
            if (scaleDiag) {
                for (int i = 0; i < m_nDim; i++) {
                    if (work_arr2[i] != 0.0) {
                        diag[i] = work_arr2[i];
                    } else {
                        diag[i] = 1.0;
                    }
                }
            }

            m_xnorm = snls::linalg::norm<m_nDim>(m_x);
            m_delta = m_factor * m_xnorm;
            if (m_delta == 0.0) {
                m_delta = m_factor; 
            }
        }

        if (scaleDiag) {
            for (int i = 0; i < m_nDim; i++) {
                diag[i] = (diag[i] > work_arr2[i]) ? diag[i] : work_arr2[i];
            }
        }

        // Jacobian is our R matrix and Q
        snls::houseHolderQR<m_nDim, m_nDim>(Jacobian, Q, work_arr1, work_arr2, work_arr3);

        snls::linalg::matTVecMult<m_nDim, m_nDim>(Q, residual, qtf);

        // std::cout << "R" << std::endl;
        // snls::linalg::printMat<m_nDim>(Jacobian);
        // std::cout << "Q" << std::endl;
        // snls::linalg::printMat<m_nDim>(Q);
        // std::cout << "qtf" << std::endl;
        // snls::linalg::printVec<m_nDim>(qtf);
        // std::cout << "diag" << std::endl;
        // snls::linalg::printVec<m_nDim>(diag);

        for (int i = 0; i < m_nDim; i++) {
            work_arr1[i] = m_x[i];
        }

        for (;;) {
            // Determine the direction p
            this->dogleg(work_arr1, Jacobian, diag, qtf, m_delta);
            // Store the direction of p and x + p
            double pnorm = 0.0;
            for (int i = 0; i < m_nDim; i++) {
                work_arr1[i] *= -1.0; 
                // If our solution isn't satisfactory later on we'll
                // reject the solution and calculate the new value of things
                m_x[i] += work_arr1[i];
                pnorm += (diag[i] * work_arr1[i]) * (diag[i] * work_arr1[i]);
            }
            pnorm = sqrt(pnorm);

            // std::cout << "work_arr1" << std::endl;
            // snls::linalg::printVec<m_nDim>(work_arr1);
            // std::cout << "m_x" << std::endl;
            // snls::linalg::printVec<m_nDim>(m_x);


            // On the first iteration adjust the initial step bound
            if (m_nIters == 1) {
                m_delta = fmin(m_delta, pnorm);
            }

            // Evaluate the function at x + p and calculate its norm
            // Previously iterations kept x and residual seperated by
            // making use of working arrays...
            // This is a waste of memory as we can just use residual and x arrays.
            // Also, we currently don't check on the success of this...
            // and we should at this back in at some point
            // However in the mean time, this is to quite the compiler
            // bool rjSuccess = 
            this->computeRJ(residual, nullptr);
            // Need to think about what to do if this fails as things
            // are handled differently from the TrDL solver.
            // if ( !(rjSuccess) ) {
            //     m_status = initEvalFailure;
            //     return m_status;
            // }
            m_fevals += 1;

            const double res1 = snls::linalg::norm<m_nDim>(residual);

            // Compute the scaled actual reduction
            double actual_reduction = -1.0;
            // Compute 2nd power
            if (res1 < m_res) {
                const double tmp = res1 / m_res;
                actual_reduction = 1.0 - (tmp * tmp); 
            }

            // Compute the predicted reduction
            snls::linalg::matUTriVecMult<m_nDim, m_nDim>(Jacobian, work_arr1, work_arr3);
            double sum = 0.0;
            for (int i = 0; i < m_nDim; i++) {
                // work_arr3 = R \Delta x + Q^T F
                // where F here is our residual vector
                work_arr3[i] += qtf[i];
                sum += (work_arr3[i] * work_arr3[i]);
            }
            sum = sqrt(sum);

            double predicted_reduction = 0.0;

            // Compute 2nd power
            if (sum < m_res) {
                sum = sum / m_res;
                predicted_reduction = 1.0 - sum * sum;
            }

            // Compute the ratio of the actual to the predicted reduction
            const double ratio = (predicted_reduction > 0.0) ? (actual_reduction / predicted_reduction) : 0.0;

            // Update the step bound
            if (ratio < 0.1) {
                m_ncsuc = 0;
                m_ncfail += 1;
                m_delta *= 0.5;
            }
            else {
                m_ncfail = 0;
                m_ncsuc += 1;
                if (ratio > 0.5 || m_ncsuc > 1) {
                    m_delta = fmax(m_delta, 2.0 * pnorm);
                }
                if (fabs(ratio - 1.0) <= 0.1) {
                    m_delta = 2.0 * pnorm;
                }
            }

            // Test for successful iteration
            if (ratio >= 1e-4) {
                // Successful iteration
                // Update norms
                double sum = 0.0;
                for (int i = 0; i < m_nDim; i++) {
                    sum += (diag[i] * m_x[i]) * (diag[i] * m_x[i]);
                }
                m_xnorm = sqrt(sum);
                m_res = res1;
                m_nIters += 1;
            }
            else {
                // Iteration failed
                // revert x back to old values
                for (int i = 0; i < m_nDim; i++) {
                    m_x[i] -= work_arr1[i];
                }
            }

            // Determine the progress of the iteration
            m_nslow1 = (actual_reduction >= 0.001) ? 0 : (m_nslow1 + 1);
            if (jeval) {
                m_nslow2 += 1;
            }
            if (actual_reduction >= 0.1) {
                m_nslow2 = 0;
            }
            // Test for convergence
            if ((m_delta <= (m_x_tolerance * m_xnorm)) || (m_res < m_f_tolerance)) {
                m_status = converged;
#ifdef __cuda_host_only__
                if ( m_os != nullptr ) {
                    *m_os << "converged" << std::endl ;
                }
#endif
                return m_status;
            }

            // Tests for termination and stringent tolerances
            if (m_fevals >= m_maxfev) {
                m_status = unConvergedMaxIter;
                return m_status;
            }
            if (0.1 * fmax(0.1 * m_delta, pnorm) <= epsmch * m_xnorm) {
                m_status = deltaFailure;
                return m_status;
            }
            if (m_nslow2 == 5) {
                m_status = slowJacobian;
                return m_status;
            }
            if (m_nslow1 == 10) {
                m_status = slowConvergence;
                return m_status;
            }

            // Criterion for recalculating jacobian
            if (m_ncfail == 2) {
                // leave inner loop and go for the next outer loop iteration
                return m_status;
            }

            // calculate the rank one modification to the jacobian
            // and update qtf if necessary
            {
                const double ipnorm = 1.0 / pnorm;
                for (int i = 0; i < m_nDim; i++) {
                    work_arr1[i] = diag[i] * diag[i] * work_arr1[i] * ipnorm;
                }
                snls::linalg::matTVecMult<m_nDim, m_nDim>(Q, residual, work_arr2);
                if (ratio >= 1e-4) {
                    for (int i = 0; i < m_nDim; i++) {
                        qtf[i] = work_arr2[i];
                    }
                }
                for (int i = 0; i < m_nDim; i++) {
                    work_arr2[i] = (work_arr2[i] - work_arr3[i]) * ipnorm;
                }
            }

            // compute the qr factorization of the updated jacobian
            bool singular = false;
            this->rank1_updt(Jacobian, Q, qtf, work_arr3, singular, work_arr1, work_arr2);
            if (singular) {
                m_status = algFailure;
                return m_status;
            }
            jeval = false;
        } // End of while loop
        // Return the unconverged result
        return m_status;
    }

    // Based on Eigen NonlinearSolver dogleg code
    // which is based on the old MINPACK version of things
    __snls_hdev__
    inline
    void dogleg(double* const x,
                const double* const qrfac,
                const double* const diag, 
                const double* const qtb, 
                const double delta)
    {
        const double epsmch = snls::dbl_epsmch;
        double work_arr1[m_nDim] = { 0.0 };
        double work_arr2[m_nDim] = { 0.0 };
        const double ndim1 = m_nDim - 1;
        // Calculating the gauss-newton direction
        for (int i = ndim1; i > -1; i-- ) {
            double temp = qrfac[SNLS_NN_INDX(i, i, m_nDim)];
            if (temp == 0.0) {
                double max = 0.0;
                for (int j = 0; j < i+1; j++) {
                    if (qrfac[SNLS_NN_INDX(j, i, m_nDim)] > max) {
                        max = qrfac[SNLS_NN_INDX(j, i, m_nDim)];
                    }
                }
                temp = (max == 0.0) ? epsmch : max * epsmch;
            }
            double sum = 0.0;
            for (int j = i + 1; j < m_nDim; j++) {
                sum += qrfac[SNLS_NN_INDX(i, j, m_nDim)] * x[j];
            }
            x[i] = (qtb[i] - sum) / temp;
        }

        // Check to see if the gauss-newton direction is acceptable
        // Scale x by diag and then take L2 norm
        double sum = 0.0;
        for (int i = 0; i < m_nDim; i++) {
            sum += (diag[i] * x[i]) * (diag[i] * x[i]);
        }
        const double qnorm = sqrt(sum);

        if (qnorm <= delta) {
            return;
        }
        
        // The gauss-newton direction is not acceptable
        // so calculate the scaled gradient direction
        for (int i = 0; i < m_nDim; i++) {
            for (int j = i; j < m_nDim; j++) {
                work_arr2[j] += qrfac[SNLS_NN_INDX(i, j, m_nDim)] * qtb[i]; 
            }
            work_arr2[i] /= diag[i];
        }

        // Norm of the scaled gradient and check for special case in 
        // which the scaled gradient is zero.  
        const double gnorm = snls::linalg::norm<m_nDim>(work_arr2);
        double sgnorm = 0.0;
        double alpha = delta / qnorm;
        
        if (gnorm != 0.0) {
            // Find the point along the scaled gradient
            // from which the quadratic is minimized.
            for (int i = 0; i < m_nDim; i++) {
                work_arr2[i] /= (diag[i] * gnorm);
            }

            for (int i = 0; i < m_nDim; i++) {
                double sum = 0.0;
                for (int j = i; j < m_nDim; j++) {
                    sum += qrfac[SNLS_NN_INDX(i, j, m_nDim)] * work_arr2[j];
                }
                work_arr1[i] = sum;
            }

            const double itemp = 1.0 / snls::linalg::norm<m_nDim>(work_arr1);
            sgnorm = gnorm * itemp * itemp;
            alpha = 0.0;
            // Test whether the scaled gradient direction is acceptable
            if (sgnorm < delta) {
                // The scaled gradient direction is not acceptable
                // Now , calculate the point along the dogleg
                // at which the quadratic is minimized.

                const double bnorm = snls::linalg::norm<m_nDim>(qtb);
                const double iqnorm = 1.0 / qnorm;
                const double idelta = 1.0 / delta;
                double temp = (bnorm / gnorm) * (bnorm * iqnorm) * (sgnorm * idelta);
                // Computing 2nd power
                const double d1 = (sgnorm * idelta);
                const double d2 = temp - delta * iqnorm;
                const double d3 = delta * iqnorm;
                const double d4 = sgnorm * idelta;

                temp = temp - delta * iqnorm * (d1 * d1) + sqrt(d2 * d2 + (1.0 - d3 * d3) * (1.0 - d4 * d4));
                alpha = delta * iqnorm * (1.0 - d1 * d1) / temp;
            }
        }

        // Form appropriate convex combination of the gauss-newton 
        // direction and the scaled gradient direction.
        const double temp = (1.0 - alpha) * fmin(sgnorm, delta);
        for (int i = 0; i < m_nDim; i++) {
            x[i] = temp * work_arr2[i] + alpha * x[i];
        }
        return;
    }

    // So, the original algorithms kept the rank1_updt and the updates to Q and QtF separate. We're
    // going to combine them into one step to completely eliminate the need to save off the givens
    // rotation between steps.
    __snls_hdev__
    inline
    void rank1_updt(double* const smat, 
                    double* const qmat, 
                    double* const qtf, 
                    double* const w, 
                    bool& sing, 
                    const double* const u, 
                    const double* const v) 
    {
        const int ndim1 = m_nDim - 1;
        double givens[2];
        double vn = v[ndim1];
        
        // Move the nontrivial part of the last column of s into w
        w[ndim1] = smat[SNLS_NN_INDX(ndim1, ndim1, m_nDim)];

        // Rotate the vector v into a multiple of the n-th unit vector in
        // such a way that a spike is introduced into w
        for (int i = ndim1 - 1; i > -1; i--) {
            w[i] = 0;
            if (v[i] != 0.0) {
                // Determine a givens rotation which eliminates the information
                // necessary to recover the givens rotation
                makeGivens(givens, -vn, v[i]);
                
                vn = givens[1] * v[i] + givens[0] * vn;

                // Apply the transformation to s and extend the spike in w
                for (int j = i; j < m_nDim; j++) {
                    const double temp = givens[0] * smat[SNLS_NN_INDX(i, j, m_nDim)] - givens[1] * w[j];
                    w[j] = givens[1] * smat[SNLS_NN_INDX(i, j, m_nDim)] + givens[0] * w[j];
                    smat[SNLS_NN_INDX(i, j, m_nDim)] = temp;
                }
            }
            else {
                givens[0] = 1.0;
                givens[1] = 0.0; 
            }
            // 1st updates of Q and qtf
            for (int j = 0; j < m_nDim; j++) {
                const double temp = givens[0] * qmat[SNLS_NN_INDX(j, i, m_nDim)] - givens[1] * qmat[SNLS_NN_INDX(j, ndim1, m_nDim)];
                qmat[SNLS_NN_INDX(j, ndim1, m_nDim)] = givens[1] * qmat[SNLS_NN_INDX(j, i, m_nDim)] + givens[0] * qmat[SNLS_NN_INDX(j, ndim1, m_nDim)];
                qmat[SNLS_NN_INDX(j, i, m_nDim)] = temp;
            }

            {
                const double temp = givens[0] * qtf[i] - givens[1] * qtf[ndim1];
                qtf[ndim1] = givens[1] * qtf[i] + givens[0] * qtf[ndim1];
                qtf[i] = temp;
            } 

        }

        // Add the spike from the rank1 update to w
        for (int i = 0; i < m_nDim; i++) {
            w[i] += vn * u[i];
        }

        // Eliminate the spike
        sing = false;
        for (int i = 0; i < ndim1; i++) {
            if (w[i] != 0.0) {
                // Determine a givens rotation which eliminates the i-th element of the spike
                makeGivens(givens, -smat[SNLS_NN_INDX(i, i, m_nDim)], w[i]);
                // Apply the transformation to s and reduce the spike in w
                for (int j = i; j < m_nDim; j++) {
                    const double temp = givens[0] * smat[SNLS_NN_INDX(i, j, m_nDim)] + givens[1] * w[j];
                    w[j] = -givens[1] * smat[SNLS_NN_INDX(i, j, m_nDim)] + givens[0] * w[j];
                    smat[SNLS_NN_INDX(i, j, m_nDim)] = temp;
                }
            }
            else {
                givens[0] = 1.0;
                givens[1] = 0.0;
            }
            // Test for zero diagonal in output
            if (smat[SNLS_NN_INDX(i, i, m_nDim)] == 0.0) {
                sing = true;
            }

            // 2nd update of Q and qtf
            for (int j = 0; j < m_nDim; j++) {
                const double temp = givens[0] * qmat[SNLS_NN_INDX(j, i, m_nDim)] + givens[1] * qmat[SNLS_NN_INDX(j, ndim1, m_nDim)];
                qmat[SNLS_NN_INDX(j, ndim1, m_nDim)] = -givens[1] * qmat[SNLS_NN_INDX(j, i, m_nDim)] + givens[0] * qmat[SNLS_NN_INDX(j, ndim1, m_nDim)];
                qmat[SNLS_NN_INDX(j, i, m_nDim)] = temp;
            }

            {
                const double temp = givens[0] * qtf[i] + givens[1] * qtf[ndim1];
                qtf[ndim1] = -givens[1] * qtf[i] + givens[0] * qtf[ndim1];
                qtf[i] = temp;
            } 
        }

        // Move w back into the last column of the output s
        smat[SNLS_NN_INDX(ndim1, ndim1, m_nDim)] = w[ndim1];
    }

    // Class member variables
    public:
    static const int m_nDim = CRJ::nDimSys;
    CRJ & m_crj ;
    double m_x[m_nDim];

    private:
    static const int m_nXnDim = m_nDim * m_nDim;
    int m_fevals, m_nIters, m_jevals;
    int m_maxfev,  m_outputLevel, m_maxIter;
    int m_ncsuc, m_ncfail, m_nslow1, m_nslow2;
    double m_delta, m_factor;
    double m_x_tolerance, m_f_tolerance;
    double m_res;
    double m_xnorm;
    bool m_complete = false;

#ifdef __cuda_host_only__
    std::ostream* m_os ;
#else
    char* m_os ; // do not use
#endif

    SNLSStatus_t  m_status;

};

}