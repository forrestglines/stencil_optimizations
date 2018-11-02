#include <vector>
#include <iostream>
#include <ostream>
#include <string>
using namespace std;

#include "Test.hpp"
#include "CPUCenterDeriv.hpp"
#include "CPUConsToPrimAH.hpp"
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

  tests.push_back( new CPUConsToPrimAH<float>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        CPUConsToPrimAH<float>::PreStepType::kNone,
        CPUConsToPrimAH<float>::StepType::kNaive,
        CPUConsToPrimAH<float>::PostStepType::kNone
        ) );
  tests.push_back( new CPUConsToPrimAH<float>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        CPUConsToPrimAH<float>::PreStepType::kNone,
        CPUConsToPrimAH<float>::StepType::kNaiveOMP,
        CPUConsToPrimAH<float>::PostStepType::kNone
        ) );
  tests.push_back( new CPUConsToPrimAH<float>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        CPUConsToPrimAH<float>::PreStepType::kNone,
        CPUConsToPrimAH<float>::StepType::kNaiveSIMD,
        CPUConsToPrimAH<float>::PostStepType::kNone
        ) );

  tests.push_back( new CPUCenterDeriv<float>(32,32,32,3,1000,
        CPUCenterDeriv<float>::PreStepType::kNone,
        CPUCenterDeriv<float>::StepType::kNaive,
        CPUCenterDeriv<float>::PostStepType::kNone
        ) );
  tests.push_back( new CPUCenterDeriv<float>(32,32,32,7,1000,
        CPUCenterDeriv<float>::PreStepType::kNone,
        CPUCenterDeriv<float>::StepType::kNaive,
        CPUCenterDeriv<float>::PostStepType::kNone
        ) );
  tests.push_back( new CPUCenterDeriv<float>(32,32,32,3,1000,
        CPUCenterDeriv<float>::PreStepType::kNone,
        CPUCenterDeriv<float>::StepType::kNaiveOMP,
        CPUCenterDeriv<float>::PostStepType::kNone
        ) );
  tests.push_back( new CPUCenterDeriv<float>(32,32,32,7,1000,
        CPUCenterDeriv<float>::PreStepType::kNone,
        CPUCenterDeriv<float>::StepType::kNaiveOMP,
        CPUCenterDeriv<float>::PostStepType::kNone
        ) );
  tests.push_back( new CPUCenterDeriv<float>(32,32,32,3,1000,
        CPUCenterDeriv<float>::PreStepType::kNone,
        CPUCenterDeriv<float>::StepType::kNaiveSIMD,
        CPUCenterDeriv<float>::PostStepType::kNone
        ) );
  tests.push_back( new CPUCenterDeriv<float>(32,32,32,7,1000,
        CPUCenterDeriv<float>::PreStepType::kNone,
        CPUCenterDeriv<float>::StepType::kNaiveSIMD,
        CPUCenterDeriv<float>::PostStepType::kNone
        ) );

  for( auto const& test: tests){
    test->TestAllDims(cout);
    delete test;
  }


  return 0;
}
