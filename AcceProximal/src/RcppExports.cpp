// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// AcceProximal
arma::mat AcceProximal(arma::mat X, double lambda, double ratio, double precision, double step, double delta);
RcppExport SEXP _AcceProximal_AcceProximal(SEXP XSEXP, SEXP lambdaSEXP, SEXP ratioSEXP, SEXP precisionSEXP, SEXP stepSEXP, SEXP deltaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type ratio(ratioSEXP);
    Rcpp::traits::input_parameter< double >::type precision(precisionSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< double >::type delta(deltaSEXP);
    rcpp_result_gen = Rcpp::wrap(AcceProximal(X, lambda, ratio, precision, step, delta));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_AcceProximal_AcceProximal", (DL_FUNC) &_AcceProximal_AcceProximal, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_AcceProximal(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
