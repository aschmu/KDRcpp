#include "kernel_utils.hpp"


// using namespace arma;

// [[Rcpp::export]]
arma::mat centerMatrix(const arma::mat& X) {
  const int n = X.n_rows;
  arma::mat out = X;
  arma::colvec colMean = arma::sum(out, 1) / n;
  out.each_row() -= arma::sum(out, 0) / n;
  out.each_col() -= colMean;
  out += arma::sum(colMean) / n;
  
  return out;
  
}

// [[Rcpp::export]]
arma::mat distSquared(arma::mat A, arma::mat B) {
  arma::colvec An =  arma::sum(square(A), 1);
  arma::colvec Bn =  arma::sum(square(B), 1);
  
  arma::mat C = -2 * (A * B.t());
  C.each_col() += An;
  C.each_row() += Bn.t();
  
  return C;
}

// [[Rcpp::export]]
arma::mat RBFdot(const arma::mat& X, const arma::mat& Y, const double sigma) {
  if (sigma <=0.0) {
    Rcpp::stop("sigma must be a positive numeric value");
  }
  const double gamma = 0.5/(sigma*sigma);
  arma::mat K = arma::exp(-gamma*distSquared(X, Y));
  
  return K;
}
