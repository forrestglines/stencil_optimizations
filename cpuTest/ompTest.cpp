#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>


const int N_TESTS = 6;
typedef double Real;


const int SECOND_ORDER_FLO = 3;//3 floating point operations?
const int SECOND_ORDER_NG = 2;
void d2u_dx2_2(Real *u, Real *out, int nx, int ny, int nz){

  const int NG = SECOND_ORDER_NG;
  int i,j,k;

//#pragma omp parallel for collapse(2)
  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
#pragma omp simd
      for(i = NG; i < nx-NG; i++){
        out[i + nx * (j + nz*k) ] =    u[ (i-1) + nx * (j + nz*k) ]
                                    -2*u[ i + nx * (j + nz*k) ]
                                      *u[ (i+1) + nx * (j + nz*k) ];
      }
    }
  }
}

void d2u_dy2_2(Real *u, Real *out, int nx, int ny, int nz){

  const int NG = SECOND_ORDER_NG;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
#pragma omp simd
      for(i = NG; i < nx-NG; i++){
        out[i + nx * (j + nz*k) ] =    u[ i + nx * ( (j-1) + nz*k) ]
                                    -2*u[ i + nx * (j + nz*k) ]
                                      *u[ i + nx * ( (j+1) + nz*k) ];
      }
    }
  }
}

void d2u_dz2_2(Real *u, Real *out, int nx, int ny, int nz){

  const int NG = SECOND_ORDER_NG;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
#pragma omp simd
      for(i = NG; i < nx-NG; i++){
        out[i + nx * (j + nz*k) ] =    u[ i + nx * (j + nz*(k-1)) ]
                                    -2*u[ i + nx * (j + nz*k) ]
                                      *u[ i + nx * (j + nz*(k+1)) ];
      }
    }
  }
}

const int EIGTH_ORDER_FLO = 18;//18 floating point operations?
const int EIGTH_ORDER_NG = 4;
void d2u_dx2_8(Real *u, Real *out, int nx, int ny, int nz){

  const int NG = EIGTH_ORDER_NG;
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

void d2u_dy2_8(Real *u, Real *out, int nx, int ny, int nz){

  const int NG = EIGTH_ORDER_NG;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
#pragma omp simd
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

void d2u_dz2_8(Real *u, Real *out, int nx, int ny, int nz){

  const int NG = EIGTH_ORDER_NG;
  int i,j,k;

  for(k = NG; k < nz-NG; k++){
    for(j = NG; j < ny-NG; j++){
#pragma omp simd
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

int main(int argc, char** argv){

  int n = strtol(argv[1], NULL, 10);
  int nruns = 10000;
  if(argc > 2)
    nruns = strtol(argv[2],NULL,10);

  int nx = n, ny = n, nz = n;

  Real *u  = (Real*) malloc(sizeof(Real)*nx*ny*nz);
  Real *out= (Real*) malloc(sizeof(Real)*nx*ny*nz);

  int i,j,k;
  //Initialize u
  for( k =0; k < nz; k++){
    for( j =0; j < ny; j++){
      for( i =0; i < nx; i++){
        u[ i + nx * (j + nz*k)] = i*i*i + j*j*j + k*k*k; //d2u_dx = 3*i
        //u[ i + nx * (j + nz*k)] = exp(2*i) + exp(2*j) exp(2*k); //d2u_dx = 4*exp(2*i)
      }
    }
  }

  clock_t start,end;
  double test_times[N_TESTS];
  double test_flops[N_TESTS];
  int test_idx = 0;


  void (*functionPointers[])(Real*,Real*,int,int,int) = 
    {d2u_dx2_2,d2u_dy2_2,d2u_dz2_2,d2u_dx2_8,d2u_dy2_8,d2u_dz2_8};

  char test_names[N_TESTS][10] = 
    {"d2u_dx2_2","d2u_dy2_2","d2u_dz2_2","d2u_dx2_8","d2u_dy2_8","d2u_dz2_8"};

  int test_dims[] = {0,1,2,0,1,2};
  int test_ords[] = {2,2,2,8,8,8};

  FILE *file = fopen("ompTest.dat","w");
  fprintf(file,"# \tdim\tord\ttime\t\tflops\n");
 
  for(test_idx = 0; test_idx < N_TESTS; test_idx++){ 
    start = clock();
    for( i = 0; i < nruns; i++){
      functionPointers[test_idx](u,out,nx,ny,nz);
    }
    end = clock();

    test_times[test_idx] = ((end-start)/(double)CLOCKS_PER_SEC)/nruns;
    test_flops[test_idx] = SECOND_ORDER_FLO*pow(n-SECOND_ORDER_NG,3)/test_times[test_idx];

    printf("%s \t %.4e %.4e\n",
        test_names[test_idx],test_times[test_idx],test_flops[test_idx]);

    fprintf(file,"\t%d\t%d\t%.8e\t%.8e\n",
        test_dims[test_idx],test_ords[test_idx],
        test_times[test_idx],test_flops[test_idx]);
  }
  fclose(file);

  free(u);
  free(out);
}
