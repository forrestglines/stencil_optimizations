#include <vector>
#include <iostream>
#include <ostream>
#include <fstream>
#include <string>
using namespace std;

#include "../common/Test.hpp"
#include "CUDAConsToPrimAH.cuh"


int main(int argc, char** argv){
  if(argc != 2){
    cout<<"Expected a filename for csv output!"<<endl;
    return 0;
  }

  vector<Test*> tests;

  tests.push_back( new CUDAConsToPrimAH<float>(
        256,256,256,
        2,253,
        2,253,
        2,253,
        512,
        5,
        CUDAConsToPrimAH<float>::MemType::kMalloc,
        CUDAConsToPrimAH<float>::PreStepType::kNone,
        CUDAConsToPrimAH<float>::StepType::kNaive,
        CUDAConsToPrimAH<float>::PostStepType::kNone
        ) );
  tests.push_back( new CUDAConsToPrimAH<float>(
        256,256,256,
        2,253,
        2,253,
        2,253,
        512,
        5,
        CUDAConsToPrimAH<float>::MemType::kMalloc,
        CUDAConsToPrimAH<float>::PreStepType::kNone,
        CUDAConsToPrimAH<float>::StepType::k1D,
        CUDAConsToPrimAH<float>::PostStepType::kNone
        ) );
  tests.push_back( new CUDAConsToPrimAH<float>(
        256,256,256,
        2,253,
        2,253,
        2,253,
        512,
        5,
        CUDAConsToPrimAH<float>::MemType::kPinned,
        CUDAConsToPrimAH<float>::PreStepType::kNone,
        CUDAConsToPrimAH<float>::StepType::kNaive,
        CUDAConsToPrimAH<float>::PostStepType::kNone
        ) );
  tests.push_back( new CUDAConsToPrimAH<float>(
        256,256,256,
        2,253,
        2,253,
        2,253,
        512,
        5,
        CUDAConsToPrimAH<float>::MemType::kPinned,
        CUDAConsToPrimAH<float>::PreStepType::kNone,
        CUDAConsToPrimAH<float>::StepType::k1D,
        CUDAConsToPrimAH<float>::PostStepType::kNone
        ) );
  tests.push_back( new CUDAConsToPrimAH<float>(
        256,256,256,
        2,253,
        2,253,
        2,253,
        512,
        5,
        CUDAConsToPrimAH<float>::MemType::kUVM,
        CUDAConsToPrimAH<float>::PreStepType::kNone,
        CUDAConsToPrimAH<float>::StepType::kNaive,
        CUDAConsToPrimAH<float>::PostStepType::kNone
        ) );
  tests.push_back( new CUDAConsToPrimAH<float>(
        256,256,256,
        2,253,
        2,253,
        2,253,
        512,
        5,
        CUDAConsToPrimAH<float>::MemType::kUVM,
        CUDAConsToPrimAH<float>::PreStepType::kNone,
        CUDAConsToPrimAH<float>::StepType::k1D,
        CUDAConsToPrimAH<float>::PostStepType::kNone
        ) );

  //Open a file for csv output
  ofstream fout;
  fout.open(argv[1]);

  //First write the CSV Headers
  tests[0]->PrintTestCSVHeader(fout);
  fout <<endl;

  for( auto const& test: tests){

    //Run all variations of the test
    test->TestAllDims(cout);

    //Write parameters and timings to csv
    test->PrintTestCSV(fout);
    fout <<endl;

    //Done with this test, free it's memory
    delete test;
  }

  //Close the csv file
  fout.close();


  return 0;
}
