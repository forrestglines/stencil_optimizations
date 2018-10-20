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
    const double arith_intensity_;

    //Number of steps to run
    unsigned int nsteps_;

    Test(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int ng, 
         unsigned int nsteps,
         unsigned int flops_per_cell, double arith_intensity):
          nx_(nx),ny_(ny),nz_(nz), ng_(ng),
          mx_(nx-2*ng),my_(ny-2*ng),mz_(nz-2*ng),size_(nx*ny*nz),
          nsteps_(nsteps),
          flops_per_cell_(flops_per_cell),arith_intensity_(arith_intensity)
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

    virtual void PrintU(std::ostream& os) = 0;
    virtual void PrintTest(std::ostream& os){
      os << "nx_="<< nx_ <<"\t";
      os << "ny_="<< ny_ <<"\t";
      os << "nz_="<< nz_ <<"\t";
      os << "ng_="<< ng_ <<"\t";
      os << "mx_="<< mx_ <<"\t";
      os << "my_="<< my_ <<"\t";
      os << "mz_="<< mz_ <<"\t";
      os << "size_="<< size_ <<"\t";
      os << "flops_per_cell_="<< flops_per_cell_ <<"\t";
      os << "arith_intensity_="<< arith_intensity_ <<"\t";
      os << "nsteps_="<< nsteps_ <<"\t";
    }
};

inline std::ostream& operator<<(std::ostream& os, Test& t){
  t.PrintTest(os);
  return os;
}

#endif //TEST_H_
