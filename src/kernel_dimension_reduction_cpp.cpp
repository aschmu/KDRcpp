#include "kdr_linesearch.hpp"
#include "kernel_utils.hpp"
#include <Rcpp.h>
#include <RcppArmadillo.h>

using namespace Rcpp;

//' @title Kernel dimension reduction with rbf kernels
//'
//' @param X an n x d matrix of inputs
//' @param Y an n x 1 matrix of outputs
//' @param B a d x r matrix of initial effective directions estimate or an integer denoting the SDR subspace dimension
//' @param max_loop the number of annealing steps
//' @param sigmax0 the initial unannealed kernel for the \code{X} rbf kernel
//' @param sigmay0 the initial unannealed kernel for the \code{Y} rbf kernel
//' @param eps a positive regularization parameter
//' @param eta a positive numeric, the upper bound of golden ratio search
//' @param anl a positive numeric, the annealing parameter
//' @param verbose a boolean for detailed output
//' @param tol a stopping tolerance for gradient descent
//' @param init_deriv a boolean indicating whether to initialize the SDR matrix estimate with the derivative at \code{B}
//' @param disp a boolean, whether to display optimization progress (not implemented)
//' @details For more details, see:Fukumizu, K. Francis R. Bach and M. Jordan. Kernel dimension reduction in regression.
//' The Annals of Statistics. 37(4), pp.1871-1905 (2009)
//'
//' @return a d x r matrix with the estimated effective directions
//'
//' @examples
//' \dontrun{
//' data(wine)
//' p <- 2target reduced dimension
//' l <- 3number of classes in the wine data
//' m <- ncol(wine) - l
//' X <- as.matrix(wine[, 1:m])
//' Y <- as.matrix(wine[, (m+1):(m+l)])
//' Xs <- scale(X)
//' sx <- estim_sigma_median(Xs)
//' sy <- estim_sigma_median(Y)
//'
//' max_loop <- 50    number of iterations in kdr method
//' eps <- 0.0001   regularization parameter for matrix inversion	
//' eta <-10.0     range of the golden ratio search
//' anl	<- 4        maximum value for anealing
//' eta <- 10
//' verbose <- T    print the optimization process
//' disp <- 0
//' init_deriv <- F  1: initialization by derivative method. 0: random
//'
//' Gaussian kernels are used.  Deviation parameter are set by the median of
//' mutual distances.   In the aneaning, sigma chages to 2*median to
//' 0.5*median
//' sgx <- 0.5*sx
//' sgy <- sy  As Y is discrete, tuning is not necessary.
//'
//' cputime <- system.time(B <- kdr_trace_cpp(X = Xs, Y = Y, K = p, max_loop = max_loop, sigmax0 = sx*sqrt(p/m),
//'                                          sigmay0 = sy, eps = eps, eta = eta, anl = anl, verbose = verbose,
//'                                          tol = 1e-9))
//'
//' Z <- Xs%*%B
//'
//' }
//' @export
// [[Rcpp::export]]
arma::mat kdr_trace_cpp(arma::mat& X, arma::mat& Y, const unsigned int K, const int max_loop, const double sigmax0,
                  const double sigmay0, const double eps, const double eta, const double anl,
                  bool verbose = true, const double tol = 1e-9)
{
  const unsigned int n = X.n_rows;
  const unsigned int d = X.n_cols;
  
  if (n != Y.n_rows)
    stop("X and Y have incompatible dimensions ");
  
  if (K >= d)
    stop("Dimension of the effective subspace should be smaller than the dimension of arma::vector X !");
  arma::mat B = arma::randn(d, K);
  arma::mat u;
  arma::vec lambda;
  arma::mat v;
  arma::svd_econ(u, lambda, v, B, "left");
  
  B = u;
  // centered Gram matrix of Y
  const arma::mat Gy = RBFdot(Y, Y, sigmay0);
  arma::mat yy = distSquared(Y, Y);
  arma::mat Kyo = centerMatrix(Gy);
  Kyo += Kyo.t();
  Kyo *= 0.5;
  
  //intial objective function value
  arma::mat Z = X*B;
  arma::mat Gz = RBFdot(Z, Z, sigmax0);
  arma::mat Kz = centerMatrix(Gz);
  Kz += Kz.t();
  Kz *= 0.5;
  
  arma::mat mz = arma::inv_sympd(Kz + eps*n*arma::eye(n, n));//solve(Kz + eps*n*diag(n))
  double tr = arma::accu(Kyo % mz); //sum(Kyo*mz)
  
  if (verbose)
  {
    std::cout<<"[0] trace = "<<tr<<std::endl;
  }
  
  const double ssz2 = 2*sigmax0*sigmax0;
  const double ssy2 = 2*sigmay0*sigmay0;
  arma::mat Kzw, Kzi, Ky, Kyzi, dB, KziKyzi, Xa, Zb, tt, dKB;
  arma::mat uB, vB; //dB's left, right eigenarma::vectors not used
  arma::vec lambdaB; //dB's singular values
  double nm; //dB's spectral norm
  double sz2, sy2;
  for (int h=1; h <= max_loop; h++) {
    //annealing kernel bandwidths
    sz2 = ssz2+(anl-1)*ssz2*(max_loop-h)/(double) max_loop;
    sy2 = ssy2+(anl-1)*ssy2*(max_loop-h)/(double) max_loop;
    
    Z   = X*B;
    Kzw = RBFdot(Z, Z, std::sqrt(sz2/2.0));
    Kz  = centerMatrix(Kzw);
    Kzi = arma::inv_sympd(Kz + eps*n*arma::eye(n, n)); //chol2inv(chol(Kz + eps*n*diag(n)))solve(Kz + eps*n*diag(n))
    
    Ky = exp(-yy/sy2);
    Ky = centerMatrix(Ky);
    Kyzi = Ky*Kzi;
    
    dB = arma::zeros<arma::mat>(d, K);
    KziKyzi = Kzi*Kyzi;
    
    for (unsigned int a=0; a<d; a++)
    {
      Xa = repmat(X.col(a), 1, n);
      Xa -= Xa.t();
      for (unsigned int b=0; b < K; b++) 
      {
        Zb = repmat(Z.col(b), 1, n);
        tt = Xa % (Zb - Zb.t()) % Kzw;
        dKB = centerMatrix(tt);
        dB(a, b) =  arma::accu(KziKyzi % dKB.t());
      }
      
    }
    arma::svd(uB, lambdaB, vB, dB);
    nm = lambdaB.max();
    
    if (nm < tol)
      break;
    
    List res_linesearch = kdr_linesearch_cpp(X, Ky, sz2, B, dB/nm, eta, eps, 1e-4);
    B  = as<arma::mat>(res_linesearch["Bn"]);
    tr = res_linesearch["tr"];
    arma::svd_econ(u, lambda, v, B, "left");
    B = u;

    if (verbose)
    {
      Z  = X*B;
      Gz = RBFdot(Z, Z, sigmax0);
      Kz = centerMatrix(Gz);
      Kz += Kz.t();
      Kz *= 0.5;
      
      mz = arma::inv_sympd(Kz + eps*n*arma::eye(n,n));//chol2inv(chol(Kz + eps*n*diag(n)))solve(Kz + eps*n*diag(n))
      tr = arma::accu(Kyo % mz);
      
      std::cout << "[" << h << "] trace = " << tr << std::endl;
    }
    
  }
  return B;
  
}
