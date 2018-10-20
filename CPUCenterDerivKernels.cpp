#include <string>
#include <iostream>

#include "CPUCenterDeriv.hpp"


template<typename T>
void CPUNaiveCenterDeriv3p_x(T *u, T *out, int nx, int ny, int nz){

  const int NG = 1;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
      for(i = NG; i < nx-NG; i++){
        out[i + nx * (j + nz*k) ] =    u[ (i-1) + nx * (j + nz*k) ]
                                    -2*u[ i + nx * (j + nz*k) ]
                                      *u[ (i+1) + nx * (j + nz*k) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveCenterDeriv3p_y(T *u, T *out, int nx, int ny, int nz){

  const int NG = 1;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
      for(i = NG; i < nx-NG; i++){
        out[i + nx * (j + nz*k) ] =    u[ i + nx * ( (j-1) + nz*k) ]
                                    -2*u[ i + nx * (j + nz*k) ]
                                      *u[ i + nx * ( (j+1) + nz*k) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveCenterDeriv3p_z(T *u, T *out, int nx, int ny, int nz){

  const int NG = 1;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
      for(i = NG; i < nx-NG; i++){
        out[i + nx * (j + nz*k) ] =    u[ i + nx * (j + nz*(k-1)) ]
                                    -2*u[ i + nx * (j + nz*k) ]
                                      *u[ i + nx * (j + nz*(k+1)) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveCenterDeriv7p_x(T *u, T *out, int nx, int ny, int nz){

  const int NG = 3;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
      for(i = NG; i < nx-NG; i++){
        out[i + nx * (j + nz*k) ] = -1./560.*u[ (i-4) + nx * (j + nz*k) ]
                                    +8./315.*u[ (i-3) + nx * (j + nz*k) ]
                                    -1./5.  *u[ (i-2) + nx * (j + nz*k) ]
                                    +8./5.  *u[ (i-1) + nx * (j + nz*k) ]
                                     -205./27.*u[ i + nx * (j + nz*k) ]
                                    +8./5.  *u[ (i+1) + nx * (j + nz*k) ]
                                    -1./5.  *u[ (i+2) + nx * (j + nz*k) ]
                                    +8./315.*u[ (i+3) + nx * (j + nz*k) ]
                                    -1./560.*u[ (i+4) + nx * (j + nz*k) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveCenterDeriv7p_y(T *u, T *out, int nx, int ny, int nz){

  const int NG = 3;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
      for(i = NG; i < nx-NG; i++){
        out[i + nx * (j + nz*k) ] = -1./560.*u[ i + nx * ( (j-4) + nz*k) ]
                                    +8./315.*u[ i + nx * ( (j-3) + nz*k) ]
                                    -1./5.  *u[ i + nx * ( (j-2) + nz*k) ]
                                    +8./5.  *u[ i + nx * ( (j-1) + nz*k) ]
                                     -205./27.*u[ i + nx * (j + nz*k) ]
                                    +8./5.  *u[ i + nx * ( (j+1) + nz*k) ]
                                    -1./5.  *u[ i + nx * ( (j+2) + nz*k) ]
                                    +8./315.*u[ i + nx * ( (j+3) + nz*k) ]
                                    -1./560.*u[ i + nx * ( (j+4) + nz*k) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveCenterDeriv7p_z(T *u, T *out, int nx, int ny, int nz){

  const int NG = 3;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
      for(i = NG; i < nx-NG; i++){
        out[i + nx * (j + nz*k) ] = -1./560.*u[ i + nx * (j + nz*(k-4)) ]
                                    +8./315.*u[ i + nx * (j + nz*(k-3)) ]
                                    -1./5.  *u[ i + nx * (j + nz*(k-2)) ]
                                    +8./5.  *u[ i + nx * (j + nz*(k-1)) ]
                                     -205./27.*u[ i + nx * (j + nz*k) ]
                                    +8./5.  *u[ i + nx * (j + nz*(k+1)) ]
                                    -1./5.  *u[ i + nx * (j + nz*(k+2)) ]
                                    +8./315.*u[ i + nx * (j + nz*(k+3)) ]
                                    -1./560.*u[ i + nx * (j + nz*(k+4)) ];
      }
    }
  }
}

//Naive triple for-loop
template<typename T>
void CPUCenterDeriv<T>::CPUNaiveCenterDeriv(int dim){
  switch (stencil_size_){
    case 3:
      switch(dim){
        case 0:
          CPUNaiveCenterDeriv3p_x(u_,u2_,nx_,ny_,nz_);
          break;
        case 1:
          CPUNaiveCenterDeriv3p_y(u_,u2_,nx_,ny_,nz_);
          break;
        case 2:
          CPUNaiveCenterDeriv3p_y(u_,u2_,nx_,ny_,nz_);
          break;
        case 3:
          CPUNaiveCenterDeriv3p_x(u_,u2_,nx_,ny_,nz_);
          CPUNaiveCenterDeriv3p_y(u_,u2_,nx_,ny_,nz_);
          CPUNaiveCenterDeriv3p_z(u_,u2_,nx_,ny_,nz_);
        default:
          std::stringstream ss;
          ss  << "dim '"<<dim<<"' unsupported!";
          throw ss.str();
          break;
      }
      break;
    case 7:
      switch(dim){
        case 0:
          CPUNaiveCenterDeriv7p_x(u_,u2_,nx_,ny_,nz_);
          break;
        case 1:
          CPUNaiveCenterDeriv7p_y(u_,u2_,nx_,ny_,nz_);
          break;
        case 2:
          CPUNaiveCenterDeriv7p_y(u_,u2_,nx_,ny_,nz_);
          break;
        case 3:
          CPUNaiveCenterDeriv7p_x(u_,u2_,nx_,ny_,nz_);
          CPUNaiveCenterDeriv7p_y(u_,u2_,nx_,ny_,nz_);
          CPUNaiveCenterDeriv7p_z(u_,u2_,nx_,ny_,nz_);
        default:
          std::stringstream ss;
          ss  << "dim '"<<dim<<"' unsupported!";
          throw ss.str();
          break;
      }
    default:
      std::stringstream ss;
      ss  << "stencil size '"<<dim<<"' unsupported!";
      throw ss.str();
      break;
  }

}
