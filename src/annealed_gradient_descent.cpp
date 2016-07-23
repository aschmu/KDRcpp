#include "annealed_gradient_descent.hpp"
#include "kernel_utils.hpp"

using namespace Rcpp;

F::F(arma::mat Bi, arma::mat dBi, arma::mat Kyi, arma::mat Xi, int ni, double epsi, double sz2i, double etai) :
  B_(Bi), dB_(dBi), Ky_(Kyi), X_(Xi), n_(ni), eps_(epsi), sz2_(sz2i), eta_(etai) {}
  
double F::kdr1dim(double s) {
  arma::mat tmpB = B_ - s*dB_;
  arma::mat u;
  arma::vec lambda;
  arma::mat v;
  arma::svd_econ(u, lambda, v, tmpB, "left");
  tmpB = u;
  arma::mat Z = X_*tmpB;
  arma::mat Kz = RBFdot(Z, Z, std::sqrt(sz2_/2.0));
  arma::mat mZ = arma::inv_sympd(centerMatrix(Kz) + n_*eps_*arma::eye(n_,n_));
  double t = arma::accu(Ky_ % mZ);
  
  return t;

}
  
