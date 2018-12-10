#include <vector>
#include <iostream>
#include <ostream>
#include <string>
using namespace std;

#include "../common/Test.hpp"
#include "OpenACCConsToPrimAH.hpp"


int main(int argc, char** argv){
  vector<Test*> tests;

  tests.push_back( new OpenACCConsToPrimAH<float>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        OpenACCConsToPrimAH<float>::PreStepType::kNone,
        OpenACCConsToPrimAH<float>::StepType::kNaive,
        OpenACCConsToPrimAH<float>::PostStepType::kNone
        ) );

  for( auto const& test: tests){
    test->TestAllDims(cout);
    delete test;
  }


  return 0;
}
