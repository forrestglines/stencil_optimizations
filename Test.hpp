#ifndef TEST_H_
#define TEST_H_

#include <ostream>

class Test{

  public:
    //Full size of the grid
    const unsigned int nx_,ny_,nz_;

    //Ghost points
    const unsigned int ng_;

    //Interior size of the grid
    const unsigned int mx_,my_,mz_;

    //Total size of the grid
    const unsigned int size_;

    //Flops per cell and arthimetic insentity
    //(In one dimension!)
    const unsigned int flops_per_cell_;
    const double arith_intenstity_;

    //Number of steps to run
    unsigned int nsteps_;

    Test(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int ng, 
         unsigned int nsteps,
         unsigned int flops_per_cell, double arith_intenstity):
          nx_(nx),ny_(ny),nz_(nz), ng_(ng),
          mx_(nx-2*ng),my_(ny-2*ng),mz_(nz-2*ng),size_(nx*ny*nz),
          nsteps_(nsteps),
          flops_per_cell_(flops_per_cell),arith_intenstity_(arith_intenstity)
        {};

    //Allocate Memory
    virtual int Malloc() = 0;

    //Start timing, extra prefetching, etc
    virtual void StartTest(int dim) = 0;

    //preStep things (borders, prefetching, copy to-from host/device memory)
    virtual void PreStep(int dim) = 0;

    //Perform a step
    virtual void Step(int dim) = 0;

    //preStep things (borders, prefetching, copy to-from host/device memory)
    virtual void PostStep(int dim) = 0;

    //End timing
    virtual void EndTest(int dim) = 0;

    //Get the seconds between the last start and end
    virtual double ElapsedTime() = 0;

    //Run the test, return seconds elapsed
    virtual double RunTest(int dim){

      StartTest(dim);
      for(int i = 0; i < nsteps_; i++){
        PreStep(dim);
        Step(dim);
        PostStep(dim);
      }
      EndTest(dim);

      return ElapsedTime();
    }

    //Free Memory
    virtual int Free() = 0;

    //Write the current data to a stream
    friend std::ostream& operator<<(std::ostream& os, Test& t);

  protected:
    virtual void Print(std::ostream& os) = 0;
};

inline std::ostream& operator<<(std::ostream& os, Test& t){
  t.Print(os);
  return os;
}

#endif //TEST_H_
