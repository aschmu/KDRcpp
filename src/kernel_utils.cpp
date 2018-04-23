#include "kernel_utils.h"


// using namespace arma;

//' Center a numeric matrix
//' 
//' @title Center a matrix
//' @param X an n x d matrix 
//' @return the n x d centred matrix
//' 
// [[Rcpp::export]]
arma::mat centerMatrix(const arma::mat& X) {
  const size_t n = X.n_rows;
  arma::mat out = X;
  arma::colvec colMean = arma::sum(out, 1) / n;
  out.each_row() -= arma::sum(out, 0) / n;
  out.each_col() -= colMean;
  out += arma::sum(colMean) / n;
  
  return out;
  
}

//' Squared distance matrix computation
//'
//' @title Squared distance matrix
//' @param A an n x d matrix 
//' @param B an n x d matrix 
//'
//' @return an n x n distance matrix
//' @export
// [[Rcpp::export]]
arma::mat distSquared(arma::mat A, arma::mat B) {
  arma::colvec An =  arma::sum(square(A), 1);
  arma::colvec Bn =  arma::sum(square(B), 1);
  
  arma::mat C = -2 * (A * B.t());
  C.each_col() += An;
  C.each_row() += Bn.t();
  
  return C;
}

//' Radial basis function kernel matrix
//'
//' @title RBF kernel matrix
//' @param X an n x d matrix or numeric vector
//' @param Y an n x e matrix or numeric vector
//' @param sigma the scale parameter for the rbf kernel
//'
//' @return an n x n rbf kernel matrix
//'
//' @details this should be slightly faster than using \code{\link[kernlab]{kernelMatrix}}
//' @export
//' @examples
//' x <- as.matrix(rnorm(5e2, 0, 1))
//' K <- RBFdot(x, x, .5)
// [[Rcpp::export]]
arma::mat RBFdot(const arma::mat& X, const arma::mat& Y, const double sigma) {
  if (sigma <=0.0) {
    Rcpp::stop("sigma must be a positive numeric value");
  }
  const double gamma = 0.5/(sigma*sigma);
  arma::mat K = arma::exp(-gamma*distSquared(X, Y));
  
  return K;
}
