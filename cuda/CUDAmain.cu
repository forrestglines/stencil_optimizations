#include <vector>
#include <iostream>
#include <ostream>
#include <string>
using namespace std;

#include "../common/Test.hpp"
#include "CUDAConsToPrimAH.cuh"


int main(int argc, char** argv){
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

  for( auto const& test: tests){
    test->TestAllDims(cout);
    delete test;
  }


  return 0;
}
