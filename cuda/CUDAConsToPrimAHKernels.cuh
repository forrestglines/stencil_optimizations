#include <string>
#include <iostream>


#define SQR(X) (X)*(X)

__constant__ double c_density_floor_ = 0.01;
__constant__ double c_pressure_floor_ = 0.01;
__constant__ double c_gm1_ = 1.666666667;

template<typename T>
void CUDAConsToPrimAH<T>::MemcpyConstants(){
  cudaMemcpyToSymbol(c_density_floor_, &density_floor_, 
      sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_pressure_floor_, &pressure_floor_, 
      sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_gm1_, &gm1_, 
      sizeof(double), 0, cudaMemcpyHostToDevice);

}

/***************************************
  CUDA Kernel for conserved to primitive
  for adiabatic hydro, using my first
  naive approach with pencils spanning X
 ***************************************/
template<typename T>
__global__ void d_CUDANaiveConsToPrimAH(
    T* cons, T* prim, 
    const unsigned int ni, const unsigned int nj, const unsigned int nk,
    const unsigned int is, const unsigned int ie,
    const unsigned int js, const unsigned int je,
    const unsigned int ks, const unsigned int ke)
{

  unsigned int i = threadIdx.x + is;
  unsigned int j = blockIdx.x * blockDim.y + threadIdx.y + js;
  unsigned int k = blockIdx.y + ks;

  if( i <= ie && j <= je){
    T& u_d  = cons[i + ni*(j + nj*(k + nk*IDN))];
    T& u_m1 = cons[i + ni*(j + nj*(k + nk*IM1))];
    T& u_m2 = cons[i + ni*(j + nj*(k + nk*IM2))];
    T& u_m3 = cons[i + ni*(j + nj*(k + nk*IM3))];
    T& u_e  = cons[i + ni*(j + nj*(k + nk*IEN))];

    T& w_d  = prim[i + ni*(j + nj*(k + nk*IDN))];
    T& w_vx = prim[i + ni*(j + nj*(k + nk*IVX))];
    T& w_vy = prim[i + ni*(j + nj*(k + nk*IVY))];
    T& w_vz = prim[i + ni*(j + nj*(k + nk*IVZ))];
    T& w_p  = prim[i + ni*(j + nj*(k + nk*IPR))];

    // apply density floor, without changing momentum or energy
    u_d = (u_d > c_density_floor_) ?  u_d : c_density_floor_;
    w_d = u_d;

    T di = 1.0/u_d;
    w_vx = u_m1*di;
    w_vy = u_m2*di;
    w_vz = u_m3*di;

    T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    w_p = c_gm1_*(u_e - ke);

    // apply pressure floor, correct total energy
    u_e = (w_p > c_pressure_floor_) ?  u_e : ((c_pressure_floor_/c_gm1_) + ke);
    w_p = (w_p > c_pressure_floor_) ?  w_p : c_pressure_floor_;
  }
}

template<typename T>
void CUDAConsToPrimAH<T>::CUDANaiveConsToPrimAH(int dim){
  //Have pencils that span X, and as high as possible in y
  const dim3 blockDim(mi_, floor(max_block_size_/mi_),1);
  const dim3 gridDim(ceil(mj_/blockDim.y),mk_,1);

  d_CUDANaiveConsToPrimAH<<<gridDim,blockDim>>>(d_cons_,d_prim_,
      ni_,nj_,nk_,is_,ie_,js_,je_,ks_,ke_);
}



/***************************************
  CUDA Kernel for conserved to primitive
  for adiabatic hydro, using a naive 
  flattened 1D index. Intentionally a
  bit cumbersome, to maybe mimic Kokkos
 ***************************************/
template<typename T>
__global__ void d_CUDA1DConsToPrimAH(
    T* cons, T* prim, 
    const unsigned int ni, const unsigned int nj, const unsigned int nk,
    const unsigned mimjmk, const unsigned int mimj, const unsigned int mi,
    const unsigned int is, 
    const unsigned int js, 
    const unsigned int ks)
{

  const unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int k = idx / mimj;
  unsigned int j = (idx - k*mimj)/ mi;
  unsigned int i = idx - k*mimj - j*mi;

  k += ks;
  j += js;
  i += is;

  if( idx < mimjmk){
    T& u_d  = cons[i + ni*(j + nj*(k + nk*IDN))];
    T& u_m1 = cons[i + ni*(j + nj*(k + nk*IM1))];
    T& u_m2 = cons[i + ni*(j + nj*(k + nk*IM2))];
    T& u_m3 = cons[i + ni*(j + nj*(k + nk*IM3))];
    T& u_e  = cons[i + ni*(j + nj*(k + nk*IEN))];

    T& w_d  = prim[i + ni*(j + nj*(k + nk*IDN))];
    T& w_vx = prim[i + ni*(j + nj*(k + nk*IVX))];
    T& w_vy = prim[i + ni*(j + nj*(k + nk*IVY))];
    T& w_vz = prim[i + ni*(j + nj*(k + nk*IVZ))];
    T& w_p  = prim[i + ni*(j + nj*(k + nk*IPR))];

    // apply density floor, without changing momentum or energy
    u_d = (u_d > c_density_floor_) ?  u_d : c_density_floor_;
    w_d = u_d;

    T di = 1.0/u_d;
    w_vx = u_m1*di;
    w_vy = u_m2*di;
    w_vz = u_m3*di;

    T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    w_p = c_gm1_*(u_e - ke);

    // apply pressure floor, correct total energy
    u_e = (w_p > c_pressure_floor_) ?  u_e : ((c_pressure_floor_/c_gm1_) + ke);
    w_p = (w_p > c_pressure_floor_) ?  w_p : c_pressure_floor_;
  }
}

template<typename T>
void CUDAConsToPrimAH<T>::CUDA1DConsToPrimAH(int dim){
  //Have blocks as big as needed
  const unsigned int blockDim = max_block_size_;
  //And a grid x dimension that spans the domain
  const unsigned int gridDim = ceil(mi_*mj_*mk_/max_block_size_);

  d_CUDA1DConsToPrimAH<<<gridDim,blockDim>>>(
      d_cons_,d_prim_,
      ni_,nj_,nk_,mi_*mj_*mk_,mi_*mj_,mi_,is_,js_,ks_);
}
