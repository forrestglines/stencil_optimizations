#include <vector>
#include <iostream>
#include <ostream>
#include <string>
using namespace std;

#include "Test.hpp"
#include "CPUCenterDeriv.hpp"


int main(int argc, char** argv){
  vector<Test*> tests;

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
