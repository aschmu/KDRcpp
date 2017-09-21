#include "annealed_gradient_descent.h"
#include <limits>
#include <cmath>
#include <RcppArmadillo.h>

using namespace Rcpp;

typedef double (F::*FMemFn)(double s);  // memberfunction pointer typedef

/*Brent's box minim in 1D : this is a very minor
 modification to Steve Verill's C++ translation from a FORTRAN code
Copyright Steve Verill <steve@ws13.fpl.fs.fed.us>
*/
double Fmin(double a, double b, F &obj, FMemFn f, double tol)
{
  double c, d, e, eps, xm, p, q, r, tol1, t2, 
  u, v, w, fu, fv, fw, fx, x, tol3;
  c = .5*(3.0 - std::sqrt(5.0));
  d = 0.0;
  // 1.1102e-16 is machine precision
  eps = std::numeric_limits<double>::epsilon();
  tol1 = eps + 1.0;
  eps = std::sqrt(eps);
  v = a + c*(b-a);
  w = v;
  x = v;
  e = 0.0;
  fx = (obj.*f)(x);
  fv = fx;
  fw = fx;
  tol3 = tol/3.0;
  xm = .5*(a + b);
  tol1 = eps*std::fabs(x) + tol3;
  t2 = 2.0*tol1;
  // main loop
  while (std::fabs(x-xm) > (t2 - .5*(b-a))) {
    p = q = r = 0.0;
    if (std::fabs(e) > tol1) {
      // fit the parabola
      r = (x-w)*(fx-fv);
      q = (x-v)*(fx-fw);
      p = (x-v)*q - (x-w)*r;
      q = 2.0*(q-r);
      if (q > 0.0) {
        p = -p;
      } else {
        q = -q;
      }
      r = e;
      e = d;
      // brace below corresponds to statement 50
    }
    if ((std::fabs(p) < std::fabs(.5*q*r)) &&
        (p > q*(a-x)) &&
        (p < q*(b-x))) {
      // a parabolic interpolation step
      d = p/q;
      u = x+d;
      // f must not be evaluated too close to a or b
      if (((u-a) < t2) || ((b-u) < t2)) {
        d = tol1;
        if (x >= xm) d = -d;
      }
      // brace below corresponds to statement 60
    } else {
      // a golden-section step
      if (x < xm) {
        e = b-x;
      } else {
        e = a-x;
      }
      d = c*e;
    }
    // f must not be evaluated too close to x
    if (std::fabs(d) >= tol1) {
      u = x+d;
    } else {
      if (d > 0.0) {
        u = x + tol1;
      } else {
        u = x - tol1;
      }
    }
    fu = (obj.*f)(u);
    // Update a, b, v, w, and x
    if (fx <= fu) {
      if (u < x) {
        a = u;
      } else {
        b = u;
      }
      // brace below corresponds to statement 140
    }
    if (fu <= fx) {
      if (u < x) {
        b = x;
      } else {
        a = x;
      }
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
      xm = .5*(a + b);
      tol1 = eps*std::fabs(x) + tol3;
      t2 = 2.0*tol1;
      // brace below corresponds to statement 170
    } else {
      if ((fu <= fw) || (w == x)) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
        xm = .5*(a + b);
        tol1 = eps*std::fabs(x) + tol3;
        t2 = 2.0*tol1;
      } else if ((fu > fv) && (v != x) && (v != w)) {
        xm = .5*(a + b);
        tol1 = eps*std::fabs(x) + tol3;
        t2 = 2.0*tol1;
      } else {
        v = u;
        fv = fu;
        xm = .5*(a + b);
        tol1 = eps*std::fabs(x) + tol3;
        t2 = 2.0*tol1;
      }
    }
    // brace below corresponds to statement 190
  }
  return x;
}

//' KDR gradient descent with line search
//'
//' @param X an n x d matrix
//' @param Ky an n x n kernel Matrix associated with \code{Y}
//' @param sz2 the kernel variance
//' @param B a d x r matrix, the current SDR matrix estimate
//' @param dB a d x r matrix, the gradient at \code{B}
//' @param eta a positive numeric, the upper bound on the linesearch parameter
//' @param eps a positive regularization parameter
//' @param tol a stopping tolerance for the minimizer
//' @details The function implements a simple linesearch by minimizing a univariate 
//' function on [0, \code{eta}] using Brent's algorithm
//'
//' @return the step size parameter
// [[Rcpp::export]]
List kdr_linesearch_cpp(const arma::mat& X, 
                        const arma::mat& Ky,
                        const double& sz2, 
                        const arma::mat& B, 
                        const arma::mat& dB,
                        const double eta, 
                        const double eps, 
                        const double tol=1e-4) {
  
  const int n = X.n_rows;
  
  F f =  F(B, dB, Ky, X, n, eps, sz2);
  
  FMemFn fun = &F::kdr1dim;
  double sopt = Fmin(0.5, eta, f, fun, tol);
  double tr = f.kdr1dim(sopt);
  arma::mat Bn = B - sopt*dB;
  
  return List::create(Named("Bn")=Bn, Named("tr")=tr);
}
