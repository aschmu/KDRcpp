#include "kdr_linesearch.hpp"
#include "kernel_utils.hpp"
#include <Rcpp.h>
#include <RcppArmadillo.h>

using namespace Rcpp;

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
  
  // #intial objective function value
  arma::mat Z = X*B;
  arma::mat Gz = RBFdot(Z, Z, sigmax0);
  arma::mat Kz = centerMatrix(Gz);
  Kz += Kz.t();
  Kz *= 0.5;
  // Kz = .5*(Kz + Kz.t());
  
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
    Kzi = arma::inv_sympd(Kz + eps*n*arma::eye(n, n)); //chol2inv(chol(Kz + eps*n*diag(n))) #solve(Kz + eps*n*diag(n))
    
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
    // nm = norm(dB, "2");
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
      
      mz = arma::inv_sympd(Kz + eps*n*arma::eye(n,n));//chol2inv(chol(Kz + eps*n*diag(n))) #solve(Kz + eps*n*diag(n))
      tr = arma::accu(Kyo % mz);
      
      std::cout << "[" << h << "] trace = " << tr << std::endl;
    }
    
  }
  return B;
  
}
