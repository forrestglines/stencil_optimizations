#include <string>
#include <iostream>

#include "CPUCenterDeriv.hpp"


template<typename T>
void CPUNaiveOMPCenterDeriv3p_x(T *u, T *out, int ni, int nj, int nk){

  const int NG = 1;
  int i,j,k;
  #pragma omp parallel for collapse(3)
  for(k = NG; k < nk-NG; k++){
    for(j = NG; j < nj-NG; j++){
      for(i = NG; i < ni-NG; i++){
        out[i + ni * (j + nk*k) ] =    u[ (i-1) + ni * (j + nk*k) ]
                                    -2*u[ i + ni * (j + nk*k) ]
                                      *u[ (i+1) + ni * (j + nk*k) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveOMPCenterDeriv3p_y(T *u, T *out, int ni, int nj, int nk){

  const int NG = 1;
  int i,j,k;
  
  #pragma omp parallel for collapse(3)
  for(k = NG; k < nk-NG; k++){
    for(j = NG; j < nj-NG; j++){
      for(i = NG; i < ni-NG; i++){
        out[i + ni * (j + nk*k) ] =    u[ i + ni * ( (j-1) + nk*k) ]
                                    -2*u[ i + ni * (j + nk*k) ]
                                      *u[ i + ni * ( (j+1) + nk*k) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveOMPCenterDeriv3p_z(T *u, T *out, int ni, int nj, int nk){

  const int NG = 1;
  int i,j,k;

  #pragma omp parallel for collapse(3)
  for(k = NG; k < nk-NG; k++){
    for(j = NG; j < nj-NG; j++){
      for(i = NG; i < ni-NG; i++){
        out[i + ni * (j + nk*k) ] =    u[ i + ni * (j + nk*(k-1)) ]
                                    -2*u[ i + ni * (j + nk*k) ]
                                      *u[ i + ni * (j + nk*(k+1)) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveOMPCenterDeriv7p_x(T *u, T *out, int ni, int nj, int nk){

  const int NG = 3;
  int i,j,k;

  #pragma omp parallel for collapse(3)
  for(k = NG; k < nk-NG; k++){
    for(j = NG; j < nj-NG; j++){
      for(i = NG; i < ni-NG; i++){
        out[i + ni * (j + nk*k) ] = -1./560.*u[ (i-4) + ni * (j + nk*k) ]
                                    +8./315.*u[ (i-3) + ni * (j + nk*k) ]
                                    -1./5.  *u[ (i-2) + ni * (j + nk*k) ]
                                    +8./5.  *u[ (i-1) + ni * (j + nk*k) ]
                                     -205./27.*u[ i + ni * (j + nk*k) ]
                                    +8./5.  *u[ (i+1) + ni * (j + nk*k) ]
                                    -1./5.  *u[ (i+2) + ni * (j + nk*k) ]
                                    +8./315.*u[ (i+3) + ni * (j + nk*k) ]
                                    -1./560.*u[ (i+4) + ni * (j + nk*k) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveOMPCenterDeriv7p_y(T *u, T *out, int ni, int nj, int nk){

  const int NG = 3;
  int i,j,k;

  #pragma omp parallel for collapse(3)
  for(k = NG; k < nk-NG; k++){
    for(j = NG; j < nj-NG; j++){
      for(i = NG; i < ni-NG; i++){
        out[i + ni * (j + nk*k) ] = -1./560.*u[ i + ni * ( (j-4) + nk*k) ]
                                    +8./315.*u[ i + ni * ( (j-3) + nk*k) ]
                                    -1./5.  *u[ i + ni * ( (j-2) + nk*k) ]
                                    +8./5.  *u[ i + ni * ( (j-1) + nk*k) ]
                                     -205./27.*u[ i + ni * (j + nk*k) ]
                                    +8./5.  *u[ i + ni * ( (j+1) + nk*k) ]
                                    -1./5.  *u[ i + ni * ( (j+2) + nk*k) ]
                                    +8./315.*u[ i + ni * ( (j+3) + nk*k) ]
                                    -1./560.*u[ i + ni * ( (j+4) + nk*k) ];
      }
    }
  }
}

template<typename T>
void CPUNaiveOMPCenterDeriv7p_z(T *u, T *out, int ni, int nj, int nk){

  const int NG = 3;
  int i,j,k;

  #pragma omp parallel for collapse(3)
  for(k = NG; k < nk-NG; k++){
    for(j = NG; j < nj-NG; j++){
      for(i = NG; i < ni-NG; i++){
        out[i + ni * (j + nk*k) ] = -1./560.*u[ i + ni * (j + nk*(k-4)) ]
                                    +8./315.*u[ i + ni * (j + nk*(k-3)) ]
                                    -1./5.  *u[ i + ni * (j + nk*(k-2)) ]
                                    +8./5.  *u[ i + ni * (j + nk*(k-1)) ]
                                     -205./27.*u[ i + ni * (j + nk*k) ]
                                    +8./5.  *u[ i + ni * (j + nk*(k+1)) ]
                                    -1./5.  *u[ i + ni * (j + nk*(k+2)) ]
                                    +8./315.*u[ i + ni * (j + nk*(k+3)) ]
                                    -1./560.*u[ i + ni * (j + nk*(k+4)) ];
      }
    }
  }
}

//NaiveOMP triple for-loop
template<class T>
void CPUCenterDeriv<T>::CPUNaiveOMPCenterDeriv(int dim){
  switch (stencil_size_){
    case 3:
      switch(dim){
        case 0:
          CPUNaiveOMPCenterDeriv3p_x(u_,u2_,ni_,nj_,nk_);
          break;
        case 1:
          CPUNaiveOMPCenterDeriv3p_y(u_,u2_,ni_,nj_,nk_);
          break;
        case 2:
          CPUNaiveOMPCenterDeriv3p_y(u_,u2_,ni_,nj_,nk_);
          break;
        case 3:
          CPUNaiveOMPCenterDeriv3p_x(u_,u2_,ni_,nj_,nk_);
          CPUNaiveOMPCenterDeriv3p_y(u_,u2_,ni_,nj_,nk_);
          CPUNaiveOMPCenterDeriv3p_z(u_,u2_,ni_,nj_,nk_);
          break;
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
          CPUNaiveOMPCenterDeriv7p_x(u_,u2_,ni_,nj_,nk_);
          break;
        case 1:
          CPUNaiveOMPCenterDeriv7p_y(u_,u2_,ni_,nj_,nk_);
          break;
        case 2:
          CPUNaiveOMPCenterDeriv7p_y(u_,u2_,ni_,nj_,nk_);
          break;
        case 3:
          CPUNaiveOMPCenterDeriv7p_x(u_,u2_,ni_,nj_,nk_);
          CPUNaiveOMPCenterDeriv7p_y(u_,u2_,ni_,nj_,nk_);
          CPUNaiveOMPCenterDeriv7p_z(u_,u2_,ni_,nj_,nk_);
          break;
        default:
          std::stringstream ss;
          ss  << "dim '"<<dim<<"' unsupported!";
          throw ss.str();
          break;
      }
      break;
    default:
      std::stringstream ss;
      ss  << "stencil size '"<<stencil_size_<<"' unsupported!";
      throw ss.str();
      break;
  }

}



template void CPUCenterDeriv<float>::CPUNaiveOMPCenterDeriv(int dim);
template void CPUCenterDeriv<double>::CPUNaiveOMPCenterDeriv(int dim);
