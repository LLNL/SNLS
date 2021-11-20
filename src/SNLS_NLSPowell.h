#pragma once

#include "SNLS_config.h"
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

// A hybrid trust region type solver, dogleg approximation
// for dense general Jacobian matrix that makes use of a rank-1 update of the jacobian
// using QR factorization
// Method is inspired by SNLS current trust region dogleg solver, Powell's original hybrid method for
// nonlinear equations, and MINPACK's modified version of it.
//
// CRJ should :
// 	have member function
// 		     __snls_hdev__ bool computeRJ( double* const r, double* const J, const double* const x ) ;
// 		computeRJ function returns true for successful evaluation
// 		TODO ... J becomes a RAJA::View ?
//	have trait nDimSys
//
template< class CRJ>
class SNLSHybrdTrDLDenseG 
{
    public:
    static_assert(snls::has_valid_computeRJ<CRJ>::value, "The CRJ implementation in SNLSHybrdTrDLDenseG needs to implement bool computeRJ( double* const r, double* const J, const double* const x )");
    static_assert(snls::has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");

    // constructor
    __snls_hdev__ SNLSHybrdTrDLDenseG(CRJ &crj) :
                m_crj(crj),
                m_fevals(0), m_nIters(0), m_jevals(0), 
                m_maxfev(1000), m_outputLevel(0),
                m_delta(1e8), m_res(1e20),
                m_os(nullptr),
                m_status(unConverged)
                {
                };
    // destructor
    __snls_hdev__ ~SNLSHybrdTrDLDenseG()
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
                     double tolerance,
                     TrDeltaControl * deltaControl,
                     int outputLevel = 0) 
    {
        m_status = unConverged;
        m_fevals = 0;
        m_jevals = 0

        m_maxIter = maxIter;
        m_maxfev = maxIter;
        m_tolerance = tolerance;
        m_deltaControl = deltaControl;

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
        // Do initial solver checks
        m_status = solveInit(residual);
        if (m_status != unConverged) {
            return m_status;
        }

        // Run our solver until it converges or fails
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
                           double* const grad,
                           double* const nrStep,
                           double* const delx)
    {
        const double epsmch = snls::dbl_epsmch;
        bool jeval = true;

        bool rjSuccess = this->computeRJ(residual, Jacobian);
        // If this fails our solver is in trouble and needs to die.
        if ( !(rjSuccess) ) {
            m_status = evalFailure;
            return m_status;
        }
        m_jevals += 1;

        // Jacobian is our R matrix and Q
        // could re-use nrstep, grad, and delx given if we're already here than we need to reset our solver
        // so these arrays can be used as scratch arrays.
        snls::houseHolderQR<m_nDim, m_nDim>(Jacobian, Q, work_arr1, work_arr2, work_arr3);
        // Nothing crazy here as qtf = Q^T * residual
        snls::linalg::matTVecMult<m_nDim, m_nDim>(Q, residual, qtf);

        // we're essentially starting over here so we can reset these values
        bool reject_prev = false;
        double Jg_2 = 0.0;
        double res_0 = m_res;

        for (;;) {

            // This is done outside this step so that these operations can be done with varying solve
            // techniques such as LU/QR or etc...
            if(!reject_prev) {
               // So the LU solve does things in-place which causes issues when calculating the grad term...
               // So, we need to pull this out and perform this operation first
               // R^T * Q^T * f
               // Need a version of the below that can do the transpose...
               // work_arr1 = grad
               snls::linalg::matUTriTVecMult<m_nDim, m_nDim>(Jacobian, qtf, grad);
               {
                  double ntemp[m_nDim];
                  snls::linalg::matUTriVecMult<m_nDim, m_nDim>(Jacobian, grad, ntemp);
                  Jg_2 = snls::linalg::dotProd<m_nDim>(ntemp, ntemp);
               }
               // R x = Q^T f solve
               this->computeNewtonStep( Jacobian, qtf, nrStep);
            }
            //
            double pred_resid;
            bool use_nr = false;

            // If step was rejected nrStep will be the same value and so we can just recalculate it here
            const double nr_norm = snls::linalg::norm<_nDim>(nrStep);

            // computes the updated delta x, predicated residual error, and whether or not NR method was used.
            snls::dogleg<_nDim>(m_delta, res_0, nr_norm, Jg_2, grad, nrStep,
                                delx, m_x, pred_resid, use_nr, m_os);
            reject_prev = false;

            //
            bool rjSuccess = this->computeRJ(residual, nullptr) ; // at _x
            // Could also potentially include the reject previous portion in here as well
            // if we want to keep this similar to the batch version of things
            snls::updateDelta<_nDim>(m_deltaControl, residual, res_0, pred_resid, nr_norm, m_tolerance, use_nr, rjSuccess,
                                    m_delta, m_res, m_rhoLast, reject_prev, m_status, m_os);
            // This new check is required due to moving all the delta update stuff into its own function to share features between codes
            if(_status != SNLSStatus_t::unConverged) { return; }

            if ( reject_prev ) {
#ifdef __cuda_host_only__
               if ( m_os != nullptr ) {
                  *m_os << "rejecting solution" << std::endl ;
               }
#endif
               m_res = res_0 ;
               this->reject( delx ) ;
            }

            // Look at a relative reduction in residual to see if convergence is slow
            const double actual_reduction = 1.0 - m_res / res_0;

            // Delta has been updated from a bounds already
            // Check to see if we need to recalculate jacobian
            m_ncfail = (m_rhoLast < 0.1) ? (m_ncfail + 1) : 0;
            if (m_ncfail == 2) {
                // leave inner loop and go for the next outer loop iteration
                return m_status;
            }

            // Determine the progress of the iteration
            m_nslow1 = (actual_reduction >= 0.001) ? 0 : (m_nslow1 + 1);
            m_nslow2 = (jeval && (actual_reduction < 0.1)) ? (m_nslow2 + 1) : 0;

            // Tests for termination and stringent tolerances
            if (m_nslow2 == 5) {
                m_status = slowJacobian;
                return m_status;
            }
            if (m_nslow1 == 10) {
                m_status = slowConvergence;
                return m_status;
            }
            // Only calculate this if solution wasn't rejected
            if( !reject_prev ) 
            {
                // Here we can use delx, nrStep, and grad as working arrays as we're just going
                // to rewrite them in a second...
                // nrStep = (R * delx + Q^T * f_i)
                snls::linalg::matUTriVecMult<m_nDim, m_nDim>(Jacobian, delx, nrStep);
                double sum = 0.0;
                for (int i = 0; i < m_nDim; i++) {
                    // work_arr3 = R \Delta x + Q^T F
                    // where F here is our residual vector
                    nrStep[i] += qtf[i];
                }
                // calculate the rank one modification to the jacobian
                // and update qtf if necessary
                {
                    // delx = delx / ||delx||_L2
                    const double idxnorm = 1.0 / snls::linalg::norm<m_nDim>(delx);
                    for (int i = 0; i < m_nDim; i++) {
                        delx[i] = delx[i] * idxnorm;
                    }
                    snls::linalg::matTVecMult<m_nDim, m_nDim>(Q, residual, grad);
                    if (ratio >= 1e-4) {
                        for (int i = 0; i < m_nDim; i++) {
                            qtf[i] = grad[i];
                        }
                    }
                    // grad = (Q^T * f_{i+1} - Q^T * f_i - R * delx)
                    for (int i = 0; i < m_nDim; i++) {
                        grad[i] = (grad[i] - nrStep[i]) * idxnorm;
                    }
                }

                // compute the qr factorization of the updated jacobian
                bool singular = false;
                this->rank1_update(Jacobian, Q, qtf, nrStep, singular, delx, grad);
                if (singular) {
                    m_status = algFailure;
                    return m_status;
                }
            // 
            }
            res_0 = m_res;
            m_fevals += 1;
            jeval = false;
        } // End of while loop
        // Return the unconverged result
        return m_status;
    }

    // So, the original algorithms kept the rank1_updt and the updates to Q and QtF separate. We're
    // going to combine them into one step to completely eliminate the need to save off the givens
    // rotation between steps.
    __snls_hdev__
    inline
    void rank1_update(double* const rmat, 
                     double* const qmat, 
                     double* const qtf, 
                     double* const resid_vec, // (R * \Delta x + Q^T * f_i)
                     bool& sing, 
                     const double* const delx_normalized, // delta x / || delta_x||_L2
                     const double* const del_resid_vec)  // (Q^T * f_{i+1} - Q^T * f_i - R * \Delta x)
    {
        const int ndim1 = m_nDim - 1;
        double givens[2];
        double del_resid_vec_n = del_resid_vec[ndim1];
        
        // Move the nontrivial part of the last column of R into (R * delx + Q^T * f_i)
        resid_vec[ndim1] = rmat[SNLS_NN_INDX(ndim1, ndim1, m_nDim)];

        // Rotate the vector (Q^T * f_{i+1} - Q^T * f_i - R * \Delta x) into a multiple of the n-th unit vector in
        // such a way that a spike is introduced into (R * \Delta x + Q^T * f_i)
        for (int i = ndim1 - 1; i > -1; i--) {
            resid_vec[i] = 0;
            if (del_resid_vec[i] != 0.0) {
                // Determine a givens rotation which eliminates the information
                // necessary to recover the givens rotation
                makeGivens(givens, -del_resid_vec_n, del_resid_vec[i]);
                
                del_resid_vec_n = givens[1] * del_resid_vec[i] + givens[0] * del_resid_vec_n;

                // Apply the transformation to s and extend the spike in w
                for (int j = i; j < m_nDim; j++) {
                    const double temp = givens[0] * rmat[SNLS_NN_INDX(i, j, m_nDim)] - givens[1] * resid_vec[j];
                    resid_vec[j] = givens[1] * rmat[SNLS_NN_INDX(i, j, m_nDim)] + givens[0] * resid_vec[j];
                    rmat[SNLS_NN_INDX(i, j, m_nDim)] = temp;
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
            resid_vec[i] += del_resid_vec_n * delx_normalized[i];
        }

        // Eliminate the spike
        sing = false;
        for (int i = 0; i < ndim1; i++) {
            if (resid_vec[i] != 0.0) {
                // Determine a givens rotation which eliminates the i-th element of the spike
                makeGivens(givens, -rmat[SNLS_NN_INDX(i, i, m_nDim)], resid_vec[i]);
                // Apply the transformation to s and reduce the spike in w
                for (int j = i; j < m_nDim; j++) {
                    const double temp = givens[0] * rmat[SNLS_NN_INDX(i, j, m_nDim)] + givens[1] * resid_vec[j];
                    resid_vec[j] = -givens[1] * rmat[SNLS_NN_INDX(i, j, m_nDim)] + givens[0] * resid_vec[j];
                    rmat[SNLS_NN_INDX(i, j, m_nDim)] = temp;
                }
            }
            else {
                givens[0] = 1.0;
                givens[1] = 0.0;
            }
            // Test for zero diagonal in output
            if (rmat[SNLS_NN_INDX(i, i, m_nDim)] == 0.0) {
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
        rmat[SNLS_NN_INDX(ndim1, ndim1, m_nDim)] = resid_vec[ndim1];
    }

    // Class member variables
    public:
    static const int m_nDim = CRJ::nDimSys;
    CRJ & m_crj ;
    double m_x[m_nDim];

    private:
    static const int m_nXnDim = m_nDim * m_nDim;
    TrDeltaControl* m_deltaControl;
    int m_fevals, m_nIters, m_jevals;
    int m_maxfev,  m_outputLevel, m_maxIter;
    int m_ncsuc, m_ncfail, m_nslow1, m_nslow2;
    double m_delta, m_rhoLast;
    double m_tolerance;
    double m_res;
    bool m_complete = false;

#ifdef __cuda_host_only__
    std::ostream* m_os ;
#else
    char* m_os ; // do not use
#endif

    SNLSStatus_t  m_status;

};

}
