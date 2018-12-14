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

  //General options
  int dimensions[][3] = {{64,64,64},{256,256,256},{512,128,128}};
  //int dimensions[][3] = {{512,128,128}};
  int ghost_zones[] = {0,2};
  int max_block_sizes[] = {256,512,1024};
  int nsteps = 5;

  typedef  CUDAConsToPrimAH<float> FloatTest;
  FloatTest::MemType float_mem_types[] = {
    FloatTest::MemType::kMalloc,
    FloatTest::MemType::kPinned,
    FloatTest::MemType::kUVM,
  };
  FloatTest::StepType float_step_types[] = {
    FloatTest::StepType::kNaive,
    FloatTest::StepType::k1D,
  };
  typedef  CUDAConsToPrimAH<double> DoubleTest;
  DoubleTest::MemType double_mem_types[] = {
    DoubleTest::MemType::kMalloc,
    DoubleTest::MemType::kPinned,
    DoubleTest::MemType::kUVM,
  };
  DoubleTest::StepType double_step_types[] = {
    DoubleTest::StepType::kNaive,
    DoubleTest::StepType::k1D,
  };

  for( auto dimension : dimensions){
    for( auto ng : ghost_zones){
      for( auto max_block_size : max_block_sizes){
        if(max_block_size < dimension[0])
          continue;
        for(int mem_type_idx = 0; mem_type_idx < 3; mem_type_idx++){
          for(int step_type_idx = 0; step_type_idx < 2; step_type_idx++){
            tests.push_back( new CUDAConsToPrimAH<float>(
                  dimension[0],dimension[1],dimension[2], 
                  ng,dimension[0]-ng-1,ng,dimension[1]-ng-1,ng,dimension[2]-ng-1, 
                  max_block_size,
                  nsteps, 
                  float_mem_types[mem_type_idx],
                  CUDAConsToPrimAH<float>::PreStepType::kNone,
                  float_step_types[step_type_idx],
                  CUDAConsToPrimAH<float>::PostStepType::kNone
                  ) );
            tests.push_back( new CUDAConsToPrimAH<double>(
                  dimension[0],dimension[1],dimension[2], 
                  ng,dimension[0]-ng-1,ng,dimension[1]-ng-1,ng,dimension[2]-ng-1, 
                  max_block_size,
                  nsteps, 
                  double_mem_types[mem_type_idx],
                  CUDAConsToPrimAH<double>::PreStepType::kNone,
                  double_step_types[step_type_idx],
                  CUDAConsToPrimAH<double>::PostStepType::kNone
                  ) );
          }
        }
      }
    }
  }
  cout<< "Num tests: "<<tests.size()<<endl;

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
