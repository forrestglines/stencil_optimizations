#include <vector>
#include <iostream>
#include <ostream>
#include <fstream>
#include <string>
using namespace std;

#include "../common/Test.hpp"
#include "KokkosConsToPrimAH.hpp"


int main(int argc, char** argv){

  if(argc != 2){
    cout<<"Expected a filename for csv output!"<<endl;
    return 0;
  }

  Kokkos::initialize(argc,argv);
  {

  vector<Test*> tests;


  //General options
  //int dimensions[][3] = {{32,32,32},{64,64,64},{256,256,256},{512,128,128}};
  //int dimensions[][3] = {{256,256,256}};
  int dimensions[][3] = {{64,64,64},{256,256,256},{512,256,128}};
  int ghost_zones[] = {0,2};
  int nsteps = 5;

  FuncType func_types[] = {FuncType::kLambda,FuncType::kFunctor};

  //MDRange specific options
  int tilings[][7] = { {512,1,1}, {512,2,1}, {64,8,1}};//,{256,1,1},{256,2,1},{256,4,1},{128,4,1},{64,8,1}};

  //TVR/TTR specifc options
  int vector_lengths[] = {1,32,64};

  //ID counter for tests
  int id = 0;


  //Create the MDRange tests
  for( auto dimension : dimensions){
    for( auto ng : ghost_zones){
      for( auto tiling : tilings){
        for( auto func_type : func_types) {
#define TEST_MDRANGE_TYPE(T_,LAYOUT_,IT_,IT_OUTER_,IT_INNER_) \
        tests.push_back( new KokkosConsToPrimAH<T_,LAYOUT_,IT_,IT_OUTER_,IT_INNER_>( \
              dimension[0],dimension[1],dimension[2], \
              ng,dimension[0]-ng-1,ng,dimension[1]-ng-1,ng,dimension[2]-ng-1, \
              nsteps, \
              Kokkos::Array<IT_,3>({tiling[0],tiling[1],tiling[2]}), \
              PreStepType::kNone, \
              StepType::kMDRange, \
              func_type, \
              PostStepType::kNone, \
              id++) );
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left   ,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Right  ,Kokkos::Iterate::Default)
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Left   )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Right  )
      TEST_MDRANGE_TYPE(double,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Default,Kokkos::Iterate::Default)
        }
      }
    }

  }
  //Create 1DRange tests
  for( auto dimension : dimensions){
    for( auto ng : ghost_zones){
        for( auto func_type : func_types) {
#define TEST_1DRANGE_TYPE(T_,LAYOUT_,IT_) \
      tests.push_back( new KokkosConsToPrimAH<T_,LAYOUT_,IT_>( \
            dimension[0],dimension[1],dimension[2], \
            ng,dimension[0]-ng-1,ng,dimension[1]-ng-1,ng,dimension[2]-ng-1, \
            nsteps, \
            PreStepType::kNone, \
            StepType::k1DRange, \
            func_type, \
            PostStepType::kNone, \
            id++) );
      TEST_1DRANGE_TYPE(float ,Kokkos::LayoutLeft ,int64_t)
      TEST_1DRANGE_TYPE(float ,Kokkos::LayoutRight,int64_t)
      TEST_1DRANGE_TYPE(double,Kokkos::LayoutLeft ,int64_t)
      TEST_1DRANGE_TYPE(double,Kokkos::LayoutRight,int64_t)
      }
    }
  }

  //Create TVR/TTR tests
  for( auto dimension : dimensions){
    for( auto ng : ghost_zones){
      for( auto vector_length : vector_lengths){
        for( auto func_type : func_types) {
#define TEST_TVR_TTR_TYPE(T_,LAYOUT_,IT_) \
        tests.push_back( new KokkosConsToPrimAH<T_,LAYOUT_,IT_>( \
              dimension[0],dimension[1],dimension[2], \
              ng,dimension[0]-ng-1,ng,dimension[1]-ng-1,ng,dimension[2]-ng-1, \
              nsteps, \
              vector_length, \
              PreStepType::kNone, \
              StepType::kTVR, \
              func_type, \
              PostStepType::kNone, \
              id++) ); \
        tests.push_back( new KokkosConsToPrimAH<T_,LAYOUT_,IT_>( \
              dimension[0],dimension[1],dimension[2], \
              ng,dimension[0]-ng-1,ng,dimension[1]-ng-1,ng,dimension[2]-ng-1, \
              nsteps, \
              vector_length, \
              PreStepType::kNone, \
              StepType::kTTR, \
              func_type, \
              PostStepType::kNone, \
              id++) );
        TEST_TVR_TTR_TYPE(float ,Kokkos::LayoutLeft ,int64_t)
        TEST_TVR_TTR_TYPE(float ,Kokkos::LayoutRight,int64_t)
        TEST_TVR_TTR_TYPE(double,Kokkos::LayoutLeft ,int64_t)
        TEST_TVR_TTR_TYPE(double,Kokkos::LayoutRight,int64_t)
        }
      }
    }
  }

  std::cout<<"Number of tests : "<<tests.size()<<std::endl;


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

  }
  Kokkos::finalize();


  return 0;
}
