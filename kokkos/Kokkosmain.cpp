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
  }

  Kokkos::initialize(argc,argv);
  {

  vector<Test*> tests;

  int id = 0;
  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Left>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        Kokkos::Array<int64_t,3>({256,1,1}),
        KokkosConsToPrimAH<float,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Left>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Left>::StepType::kMDRange,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Left>::PostStepType::kNone,
        id++) );

  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutLeft,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Left>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        Kokkos::Array<int64_t,3>({256,1,1}),
        KokkosConsToPrimAH<float,Kokkos::LayoutLeft,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Left>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutLeft,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Left>::StepType::kMDRange,
        KokkosConsToPrimAH<float,Kokkos::LayoutLeft,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Left>::PostStepType::kNone,
        id++) );
  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutLeft,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Right>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        Kokkos::Array<int64_t,3>({256,1,1}),
        KokkosConsToPrimAH<float,Kokkos::LayoutLeft,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Right>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutLeft,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Right>::StepType::kMDRange,
        KokkosConsToPrimAH<float,Kokkos::LayoutLeft,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Right>::PostStepType::kNone,
        id++) );



  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Right>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        Kokkos::Array<int64_t,3>({256,1,1}),
        KokkosConsToPrimAH<float,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Right>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Right>::StepType::kMDRange,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight,int64_t,Kokkos::Iterate::Left,Kokkos::Iterate::Right>::PostStepType::kNone,
        id++) );

  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutLeft>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        KokkosConsToPrimAH<float,Kokkos::LayoutLeft>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutLeft>::StepType::k1DRange,
        KokkosConsToPrimAH<float,Kokkos::LayoutLeft>::PostStepType::kNone,
        id++) );
  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutRight>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::StepType::k1DRange,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PostStepType::kNone,
        id++) );

  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutRight>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        1,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::StepType::kTVR,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PostStepType::kNone,
        id++) );
  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutRight>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        32,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::StepType::kTVR,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PostStepType::kNone,
        id++) );
  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutRight>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        64,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::StepType::kTVR,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PostStepType::kNone,
        id++) );
  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutRight>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        1,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::StepType::kTTR,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PostStepType::kNone,
        id++) );
  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutRight>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        32,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::StepType::kTTR,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PostStepType::kNone,
        id++) );
  tests.push_back( new KokkosConsToPrimAH<float,Kokkos::LayoutRight>(
        256,256,256,
        2,253,2,253,2,253,
        5,
        64,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PreStepType::kNone,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::StepType::kTTR,
        KokkosConsToPrimAH<float,Kokkos::LayoutRight>::PostStepType::kNone,
        id++) );


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
