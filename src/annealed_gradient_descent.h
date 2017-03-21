#ifndef _ANN_GRAD_DESC_H
#define _ANN_GRAD_DESC_H
#include <RcppArmadillo.h>

//small class that enables 1D objective function during linesearch + 
// a few other parameters used in gradient descent


class F {
private:
  arma::mat B_, dB_, Ky_, X_;
  int n_;
  double eps_, sz2_;
public:
  F(arma::mat Bi, arma::mat dBi, arma::mat Kyi, arma::mat Xi, 
    int ni, double epsi, double sz2i);
  double kdr1dim(double s);
  
}
;

#endif