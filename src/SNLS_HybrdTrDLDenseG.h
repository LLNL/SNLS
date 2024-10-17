#pragma once

#include "SNLS_config.h"
#include "SNLS_base.h"
#include "SNLS_qr_solve.h"
#include "SNLS_linalg.h"
#include "SNLS_TrDelta.h"
#include "SNLS_kernels.h"

#include <stdlib.h>
#include <iostream>
#ifdef __snls_host_only__
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
// Powell's original hybrid method can be found at:
// M. J. D. Powell, "A hybrid method for nonlinear equations", in Numerical methods for nonlinear algebraic equations, 
// Philip Rabinowitz, editor, chapter 6, pages 87-114, Gordon and Breach Science Publishers, New York, 1970.
// MINPACK's user guide is found at https://doi.org/10.2172/6997568
//
// CRJ should :
// 	have member function
// 		     __snls_hdev__ bool computeRJ( double* const r, double* const J, const double* const x ) ;
// 		computeRJ function returns true for successful evaluation
// 		TODO ... J becomes a RAJA::View ?
//	have trait nDimSys
//
// Might want to look into having this templated on a class that controls the delta as well as the trust region can at times decrease the step control a bit
// too fast which can lead the solver to fail.
template< typename CRJ, int nDimSys = CRJ::nDimSys>
class SNLSHybrdTrDLDenseG 
{
    public:
    static_assert(has_valid_computeRJ<CRJ>::value || has_valid_computeRJ_lambda<CRJ>::value, "The CRJ implementation in SNLSHybrdTrDLDenseG needs to implement bool computeRJ( double* const r, double* const J, const double* const x ) or be a lambda function that takes in the same arguments");
    // static_assert(snls::has_ndim<CRJ>::value, "The CRJ Implementation must define the const int 'nDimSys' to represent the number of dimensions");

    // constructor
    __snls_hdev__ SNLSHybrdTrDLDenseG(CRJ &crj) :
                m_crj(crj),
                m_fevals(0), m_nIters(0), m_jevals(0), 
                m_maxfev(1000), m_outputLevel(0),
                m_delta(1e8), m_res(1e20),
                m_os(nullptr),
                m_status(unConverged)
                {
                }
    // destructor
    __snls_hdev__ ~SNLSHybrdTrDLDenseG()
    {
#ifdef __snls_host_only__
        if ( m_outputLevel > 1 && m_os != nullptr ) {
            *m_os << "Function and Jacobian evaluations: " << m_fevals << " " << m_jevals << std::endl;
        }
#endif
    }
                
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
    __snls_hdev__
    SNLSStatus_t getStatus() const { return m_status; }


    // setX can be used to set the initial guess
    __snls_hdev__ 
    inline 
    void setX( const double* const x ) 
    {
        for (int iX = 0; iX < m_nDim; ++iX) {
            m_x[iX] = x[iX] ;
        }
    }

    __snls_hdev__ 
    inline 
    void getX( double* const x ) const 
    {
        for (int iX = 0; iX < m_nDim; ++iX) {
            x[iX] = m_x[iX] ;
        }
    }

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
        m_jevals = 0;

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
#ifdef __snls_host_only__
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
            solveStep(residual, Jacobian, Qmat, qtf, work_arr1, work_arr2, work_arr3);
        }

        return m_status;
    }

    // convenience wrapper, for the current _x
    __snls_hdev__ 
    inline bool computeRJ(double* const r,
                          double* const J ) 
    {
        m_fevals += 1;
        m_jevals += ((J == nullptr) ? 0 : 1);
        bool retval;
        if constexpr(has_valid_computeRJ<CRJ>::value) {
            retval = this->m_crj.computeRJ(r, J, m_x);
        } else {
            retval = this->m_crj(r, J, m_x);
        }
#ifdef SNLS_DEBUG
#ifdef __snls_host_only__
         if ( m_outputLevel > 2 && m_os != nullptr && J != nullptr) {
            // do finite differencing
            // assume system is scaled such that perturbation size can be standard

            double r_base[m_nDim]; 
            for ( int jX = 0; jX < m_nDim ; ++jX ) {
               r_base[jX] = r[jX] ;
            }
            
            const double pert_val     = 1.0e-7 ;
            const double pert_val_inv = 1.0/pert_val ;
            
            double J_FD[m_nXnDim] ;
            
            for ( int iX = 0; iX < m_nDim ; ++iX ) {
               double r_pert[m_nDim];
               double x_pert[m_nDim];
               for ( int jX = 0; jX < m_nDim ; ++jX ) {
                  x_pert[jX] = m_x[jX] ;
               }
               x_pert[iX] = x_pert[iX] + pert_val ;
               bool retvalThis;
               if constexpr(has_valid_computeRJ<CRJ>::value) {
                    retvalThis = this->m_crj.computeRJ(r_pert, nullptr, x_pert);
               } else {
                    retvalThis = this->m_crj(r_pert, nullptr, x_pert);
               }
               if ( !retvalThis ) {
                  SNLS_FAIL(__func__, "Problem while finite-differencing");
               }
               for ( int iR = 0; iR < m_nDim ; iR++ ) {
                  J_FD[SNLS_NN_INDX(iR,iX,m_nDim)] = pert_val_inv * ( r_pert[iR] - r_base[iR] ) ;
               }
            }
            
            *m_os << "J_an = " << std::endl ; snls::linalg::printMat<m_nDim>( J,    *m_os ) ;
            *m_os << "J_fd = " << std::endl ; snls::linalg::printMat<m_nDim>( J_FD, *m_os ) ;

            // put things back the way they were ;
            if constexpr(has_valid_computeRJ<CRJ>::value) {
                retval = this->m_crj.computeRJ(r, J, m_x);
            } else {
                retval = this->m_crj(r, J, m_x);
            }
            
         } // _os != nullptr
#endif
#endif        
        return retval;  
    }

    private:

    __snls_hdev__
    inline
    SNLSStatus_t solveInit(double* const residual)
    {
        m_status = snls::unConverged;
        m_fevals = 0;
        m_jevals = 0;
        // at m_x
        bool rjSuccess = this->computeRJ(residual, nullptr);
        if ( !(rjSuccess) ) {
            m_status = initEvalFailure;
            return m_status;
        }
        
        
        m_res = snls::linalg::norm<m_nDim>(residual);

#ifdef __snls_host_only__
         if (m_os) { *m_os << "res = " << m_res << std::endl ; }
#endif

        m_delta = m_deltaControl->getDeltaInit();
        
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
                           double* const qtf,
                           double* const grad,
                           double* const nrStep,
                           double* const delx)
    {
        bool jeval = true;

        bool rjSuccess = this->computeRJ(residual, Jacobian);
        // If this fails our solver is in trouble and needs to die.
        if ( !(rjSuccess) ) {
            m_status = evalFailure;
            return m_status;
        }

        // Jacobian is our R matrix and Q
        // could re-use nrstep, grad, and delx given if we're already here than we need to reset our solver
        // so these arrays can be used as scratch arrays.
        snls::householderQR<m_nDim, m_nDim>(Jacobian, Q, grad, nrStep, delx);
        // Nothing crazy here as qtf = Q^T * residual
        snls::linalg::matTVecMult<m_nDim, m_nDim>(Q, residual, qtf);

        // we're essentially starting over here so we can reset these values
        bool reject_prev = false;
        double Jg_2 = 0.0;
        // m_res is set initially in solveInit and later on in snls::updateDelta, so it's always set
        double res_0 = m_res;

        for (;;) {

            // This is done outside this step so that these operations can be done with varying solve
            // techniques such as LU/QR or etc...
            if(!reject_prev) {
                // So the LU solve does things in-place which causes issues when calculating the grad term...
                // So, we need to pull this out and perform this operation first
                // R^T * Q^T * f
                snls::linalg::matUTriTVecMult<m_nDim, m_nDim>(Jacobian, qtf, grad);
                {
                    double ntemp[m_nDim];
                    snls::linalg::matUTriVecMult<m_nDim, m_nDim>(Jacobian, grad, ntemp);
                    Jg_2 = snls::linalg::dotProd<m_nDim>(ntemp, ntemp);
                }
                // R x = Q^T f solve
                // If R is signular we fail out of the solve with an error on the CPU
                // On the GPU, the fail just prints out and  doesn't abort anything so
                // we return this signal notifying us of the failure which can then be passed
                // onto the other libraries / application codes using SNLS.
                const bool success = this->computeNewtonStep( Jacobian, qtf, nrStep);
                if (!success) {
                    m_status = SNLSStatus_t::linearSolveFailure;
                    return m_status;
                }
            }
            //
            double pred_resid;
            bool use_nr = false;

            // If the step was rejected nrStep will be the same value as previously, and so we can just recalculate nr_norm here.
            const double nr_norm = snls::linalg::norm<m_nDim>(nrStep);

            // computes the updated delta x, predicated residual error, and whether or not NR method was used.
            snls::dogleg<m_nDim>(m_delta, res_0, nr_norm, Jg_2, grad, nrStep,
                                delx, m_x, pred_resid, use_nr, m_os);
            reject_prev = false;
            //
            bool rjSuccess = this->computeRJ(residual, nullptr) ; // at _x
            snls::updateDelta<m_nDim>(m_deltaControl, residual, res_0, pred_resid, nr_norm, m_tolerance, use_nr, rjSuccess,
                                     m_delta, m_res, m_rhoLast, reject_prev, m_status, m_os);
            if(m_status != SNLSStatus_t::unConverged) { return m_status; }

            if ( reject_prev ) {
#ifdef __snls_host_only__
               if ( m_os != nullptr ) {
                  *m_os << "rejecting solution" << std::endl ;
               }
#endif
               m_res = res_0 ;
               this->reject( delx );
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
                    for (int i = 0; i < m_nDim; i++) {
                        qtf[i] = grad[i];
                    }
                    // grad = (Q^T * f_{i+1} - Q^T * f_i - R * delx)
                    for (int i = 0; i < m_nDim; i++) {
                        grad[i] = (grad[i] - nrStep[i]) * idxnorm;
                    }
                }

                // compute the qr factorization of the updated jacobian
                bool singular = false;
                this->rank1_update(delx, grad, Jacobian, Q, qtf, nrStep, singular);
                if (singular) {
                    m_status = algFailure;
                    return m_status;
                }
            // 
            }
            res_0 = m_res;
            jeval = false;
        } // End of while loop
        // Return the unconverged result
        return m_status;
    }

    // This just does the solve of
    // R x = Q^T f
    // where we're solving for x
    // Since R is upper diagonal this  is a trivial process.
    __snls_hdev__
    inline
    bool computeNewtonStep(const double* const rmat,
                           const double* const func,
                           double* const x,
                           const double tol = 1e-50)
    {
        const double ndim1 = m_nDim - 1;
        for (int i = ndim1; i > -1; i-- ) {
            double temp = rmat[SNLS_NN_INDX(i, i, m_nDim)];
            // If R has a tiny diagonal diagonal 
            if (fabs(temp) < tol) {
                // We could try something like the coding below to make sure this function always
                // succeeds, but just results in a large value in the solution vector
                // double max = 0.0;
                // for (int j = 0; j < i+1; j++) {
                //     if (rmat[SNLS_NN_INDX(j, i, m_nDim)] > max) {
                //         max = rmat[SNLS_NN_INDX(j, i, m_nDim)];
                //     }
                // }
                // temp = (max == 0.0) ? epsmch : max * snls::dbl_epsmch;
                // If R is signular we fail out of the solve with an error on the CPU
               // On the GPU, the fail just prints out and  doesn't abort anything so
               // we return this signal notifying us of the failure which can then be passed
               // onto the other libraries / application codes using SNLS.
                SNLS_WARN(__func__, "Diagonal term in R matrix was too small");
                return false;
            }
            double sum = 0.0;
            for (int j = i + 1; j < m_nDim; j++) {
                sum += rmat[SNLS_NN_INDX(i, j, m_nDim)] * x[j];
            }
            x[i] = (func[i] - sum) / temp;
        }
        for (int i = 0; i < m_nDim; i++) {
            x[i] *= -1.0;
        }
        return true;
    }

    // This performs a Broyden Rank-1 style update for Q, R and Q^T f
    // This version has origins in this paper:
    // Gill, Philip E., et al. "Methods for modifying matrix factorizations." Mathematics of computation 28.126 (1974): 505-535.
    // However, you can generally find it described in more approachable manners elsewhere on the internet
    // 
    // The Broyden update method is described in:
    // Broyden, Charles G. "A class of methods for solving nonlinear simultaneous equations." Mathematics of computation 19.92 (1965): 577-593.
    // Additional resources that might  be of interest are:
    // Chapter 8 of https://doi.org/10.1137/1.9781611971200.ch8
    // or the pseudo-algorithms / code for how to update things in
    // Appendix A of https://doi.org/10.1137/1.9781611971200.appa

    __snls_hdev__
    inline
    void rank1_update(const double* const delx_normalized, // delta x / || delta_x||_L2
                      const double* const del_resid_vec, // (Q^T * f_{i+1} - Q^T * f_i - R * \Delta x)
                      double* const rmat, 
                      double* const qmat, 
                      double* const qtf, 
                      double* const resid_vec, // (R * \Delta x + Q^T * f_i)
                      bool& sing
                     )  
    {
        const int ndim1 = m_nDim - 1;
        double givens[2];
        double del_resid_vec_n = del_resid_vec[ndim1];
        
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

                // Apply the transformation to R and extend the spike in (R * \Delta x + Q^T * f_i)
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
            // 1st updates of Q and Q^T f
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

        // Add the spike from the Rank-1 update to (R * \Delta x + Q^T * f_i)
        for (int i = 0; i < m_nDim; i++) {
            resid_vec[i] += del_resid_vec_n * delx_normalized[i];
        }

        // Eliminate the spike
        sing = false;
        for (int i = 0; i < ndim1; i++) {
            if (resid_vec[i] != 0.0) {
                // Determine a givens rotation which eliminates the i-th element of the spike
                makeGivens(givens, -rmat[SNLS_NN_INDX(i, i, m_nDim)], resid_vec[i]);
                // Apply the transformation to R and reduce the spike in (R * \Delta x + Q^T * f_i)
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

            // 2nd update of Q and Q^T F
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

        // Move (R * \Delta x + Q^T * f_i) back into the last column of the output R
        rmat[SNLS_NN_INDX(ndim1, ndim1, m_nDim)] = resid_vec[ndim1];
    }

    __snls_hdev__
    inline
    void reject( const double* const delx )
    {
        for (int i = 0; i < m_nDim; i++) {
            m_x[i] -= delx[i];
        }
    }

    // Class member variables
    public:
    static constexpr int m_nDim = nDimSys;
    static constexpr int _nDim = nDimSys;

    CRJ & m_crj ;
    double m_x[m_nDim];

    private:
    static constexpr int m_nXnDim = m_nDim * m_nDim;
    TrDeltaControl* m_deltaControl;
    int m_fevals, m_nIters, m_jevals;
    int m_maxfev,  m_outputLevel, m_maxIter;
    int m_ncsuc, m_ncfail, m_nslow1, m_nslow2;
    double m_delta, m_rhoLast;
    double m_tolerance;
    double m_res;
    bool m_complete = false;

#ifdef __snls_host_only__
    std::ostream* m_os ;
#else
    char* m_os ; // do not use
#endif

    SNLSStatus_t  m_status;

};

}
