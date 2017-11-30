#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
arma::mat MissingMatrix(arma::mat X,double ratio){
    int m,n;
    m=X.n_rows;
    n=X.n_cols;
    arma::mat Y;
    Y.randu(m,n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(Y(i,j)<=ratio){X(i,j)=0;}
        }
    }
    return X;
}

arma::mat SVToperator(arma::mat X,double lambda){
  arma::mat U,V,Ud,Vd,Xhat;arma::vec s,sd;
    arma::svd(U,s,V,X);
    arma::uvec index=find(s>1e-5);
  Ud=U.cols(index);
  Vd=V.cols(index);
  sd=s.elem(index);
  for(int i=0;i<sd.n_elem;i++){
    if(sd(i)>lambda){sd(i)=sd(i)-lambda;}
    else{sd(i)=0;}
  }
  Xhat=Ud*diagmat(sd)*trans(Vd);
  return Xhat;
}

arma::mat ProjectMatrix(arma::mat X,arma::mat Y){
  for(int i=0;i<Y.n_rows;i++){
    for(int j=0;j<Y.n_cols;j++){
      if(Y(i,j)==0){X(i,j)=0;}
    }
  }
  return X;
}

// [[Rcpp::export]]
arma::mat AcceProximal(arma::mat X,double lambda,double ratio,double precision,double step, double delta){
  arma::mat Xiter;
  arma::mat Xmiss;
  arma::mat Yiter;
  arma::mat Ziter;
  arma::mat Xtemp;
  int m,n;
  m=X.n_rows;
  n=X.n_cols;
  Xmiss=MissingMatrix(X,ratio);
  Ziter.randu(m,n);
  Xiter.randu(m,n);
  do{
    Yiter=Ziter+delta*ProjectMatrix((Xmiss-Ziter),Xmiss);
    Xtemp=Xiter;
    Xiter=SVToperator(Yiter,lambda*delta);
    Ziter=Xiter+step*(Xiter-Xtemp);
  } while (norm(Xtemp-Xiter)>precision);
  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      if(Xiter(i,j)>1){Xiter(i,j)=1;}
      if(Xiter(i,j)<0){Xiter(i,j)=0;}
    }
  }
  return Xiter;
}