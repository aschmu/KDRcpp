#ifndef _KDR_LS_H
#define _KDR_LS_H

#include "annealed_gradient_descent.h"
#include <RcppArmadillo.h>


typedef double (F::*FMemFn)(double s);  // memberfunction pointer typedef

double Fmin(double a, double b, F &obj, FMemFn f, double tol);
Rcpp::List kdr_linesearch_cpp(const arma::mat& X, 
                        const arma::mat& Ky,
                        const double& sz2, 
                        const arma::mat& B, 
                        const arma::mat& dB,
                        const double eta, 
                        const double eps, 
                        const double tol);

#endif