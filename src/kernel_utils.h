#ifndef _KERNEL_UTILS_H
#define _KERNEL_UTILS_H

#include <RcppArmadillo.h>


// Helper functions for kernel dimension reduction

arma::mat centerMatrix(const arma::mat& X);
arma::mat distSquared(arma::mat A, arma::mat B);
arma::mat RBFdot(const arma::mat& X, const arma::mat& Y, const double sigma);


#endif