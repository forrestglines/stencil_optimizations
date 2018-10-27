#ifndef TEST_H_
#define TEST_H_

#include <ostream>

class Test{

  public:

    //Total number of cells computed
    const unsigned int ncells_;

    //Flops per cell and arthimetic insentity
    //(In one dimension!)
    const unsigned int flops_per_cell_;
    const double arith_intensity_;

    //Number of steps to run
    unsigned int nsteps_;

    //Number of dims to try
    unsigned int ndims_;

    Test(unsigned int ncells_,
         unsigned int nsteps, unsigned int ndims,
         unsigned int flops_per_cell, double arith_intensity):
          ncells_(ncells_),
          nsteps_(nsteps), ndims_(ndims),
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

    virtual void TestAllDims(std::ostream& os){
      try{
        os << *this << std::endl;
        Malloc();

        for( int dim = 0; dim < ndims_; dim++){
          double time = RunTest(dim);
          os << time << std::endl;
        }
        Free();
      }
      catch( std::string e){
        os << "Caught Exception: \"" << e << "\"" << std::endl;
      }
    }

    //Free Memory
    virtual int Free() = 0;

    //Write the current data to a stream
    friend std::ostream& operator<<(std::ostream& os, Test& t);

    virtual void PrintU(std::ostream& os) = 0;
    virtual void PrintTest(std::ostream& os){
      os << "ncells_=" << ncells_ <<"\t";
      os << "flops_per_cell_=" << flops_per_cell_ <<"\t";
      os << "arith_intensity_=" << arith_intensity_ <<"\t";
      os << "nsteps_=" << nsteps_ <<"\t";
      os << "ndims_=" << ndims_ <<"\t";
    }
};

inline std::ostream& operator<<(std::ostream& os, Test& t){
  t.PrintTest(os);
  return os;
}


#endif //TEST_H_
