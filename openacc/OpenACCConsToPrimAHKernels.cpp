#include <string>
#include <iostream>

#include "OpenACCConsToPrimAH.hpp"


#define SQR(X) (X)*(X)

template<typename T>
void OpenACCConsToPrimAH<T>::OpenACCNaiveConsToPrimAH(int dim){

  int i,j,k;
  #pragma acc data copy(cons_,prim_)
  {
  #pragma acc parallel loop
  for(k = ks_; k <= ke_; k++){
    for(j = js_; j <= je_; j++){
      for(i = is_; i <= ie_; i++){
      T& u_d  = cons_(IDN,k,j,i);
      T& u_m1 = cons_(IM1,k,j,i);
      T& u_m2 = cons_(IM2,k,j,i);
      T& u_m3 = cons_(IM3,k,j,i);
      T& u_e  = cons_(IEN,k,j,i);

      T& w_d  = prim_(IDN,k,j,i);
      T& w_vx = prim_(IVX,k,j,i);
      T& w_vy = prim_(IVY,k,j,i);
      T& w_vz = prim_(IVZ,k,j,i);
      T& w_p  = prim_(IPR,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > density_floor_) ?  u_d : density_floor_;
      w_d = u_d;

      T di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1_*(u_e - ke);

      // apply pressure floor, correct total energy
      u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
      w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

      }
    }
  }
  }
}

template void OpenACCConsToPrimAH<float>::OpenACCNaiveConsToPrimAH(int dim);
template void OpenACCConsToPrimAH<double>::OpenACCNaiveConsToPrimAH(int dim);

