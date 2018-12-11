#include <cmath>


#include "Matrix.hpp"

#define NHYDRO 5

//Indices for conserved variables
#define IDN 0
#define IM1 1
#define IM2 2
#define IM3 3
#define IEN 4

//Indices for primitive variables
#define IVX 1
#define IVY 2
#define IVZ 3
#define IPR 4

const double density_floor = 0.01;
const double pressure_floor = 0.01;
const double gm1 = 1.666666667;

//Initialize conserved variables
template <typename T>
void InitCons(Matrix<T>& cons,int ni, int nj,int nk){
  //Set conserved data to something reasonable
  for(int k = 0; k < nk; k++){
    for(int j = 0; j < nj; j++){
      for(int i = 0; i < ni; i++){
        T c = (i + ni*(j + nj*k) + 1.0)/(ni*nj*nk);
        cons(IDN,k,j,i) = c;
        cons(IM1,k,j,i) = sin(c);
        cons(IM2,k,j,i) = cos(c);
        cons(IM3,k,j,i) = tan(c);
        cons(IEN,k,j,i) = c*c+4.0;
      }
    }
  }

}


//Convert conserved to primitive variables
template <typename T>
void ConsToPrimAH(Matrix<T>& cons, Matrix<T>& prim,int ni, int nj,int nk,
    int is, int ie, int js, int je, int ks, int ke){

  for(int k = ks; k <= ke; k++){
    for(int j = js; j <= je; j++){
      for(int i = is; i <= ie; i++){
      T& u_d  = cons(IDN,k,j,i);
      T& u_m1 = cons(IM1,k,j,i);
      T& u_m2 = cons(IM2,k,j,i);
      T& u_m3 = cons(IM3,k,j,i);
      T& u_e  = cons(IEN,k,j,i);

      T& w_d  = prim(IDN,k,j,i);
      T& w_vx = prim(IVX,k,j,i);
      T& w_vy = prim(IVY,k,j,i);
      T& w_vz = prim(IVZ,k,j,i);
      T& w_p  = prim(IPR,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > density_floor) ?  u_d : density_floor;
      w_d = u_d;

      T di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1_*(u_e - ke);

      // apply pressure floor, correct total energy
      u_e = (w_p > pressure_floor) ?  u_e : ((pressure_floor/gm1) + ke);
      w_p = (w_p > pressure_floor) ?  w_p : pressure_floor;

      }
    }
  }
}
