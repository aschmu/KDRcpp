[![Travis-CI Build Status](https://travis-ci.org/aschmu/KDRcpp.svg?branch=master)](https://travis-ci.org/aschmu/KDRcpp)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/aschmu/KDRcpp?branch=master&svg=true)](https://ci.appveyor.com/project/aschmu/KDRcpp)


# Kernel Dimension Reduction with RcppArmadillo

KDRcpp is a R/C++ port of the [KDR Matlab implementation](http://www.ism.ac.jp/~fukumizu/software.html) by K. Fukumizu.

KDR is a supervised dimension reduction method that uses RKHS to express conditional independance and find the dimension reduction subspace. It involves solving a non-trivial optimization problem.

Currently only Gaussian kernels are supported.


