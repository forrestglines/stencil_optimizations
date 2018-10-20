#include <vector>
#include <iostream>
#include <ostream>
#include <string>
using namespace std;

#include "Test.hpp"
#include "CPUCenterDeriv.hpp"


template <class T>
class MyTest : public Test{

  T* data;
  MyTest():Test(1,1,1,1,1,1,1){}

  virtual int Malloc(){
    data = new T;
   return 0 ;
  }

  virtual void StartTest(int dim){}
  virtual void PreStep(int dim){}
  virtual void Step(int dim){}
  virtual void PostStep(int dim){}
  virtual void EndTest(int dim){}
  virtual double ElaspsedTime(){ return 0;}

  virtual double RunTest(int dim){ return 0;}

  virtual int Free(){return 0;}
  virtual void Print(ostream& os){}

};

int main(int argc, char** argv){
  MyTest<float> t();
  /*vector<Test*> tests;

  tests.push_back( new CPUCenterDeriv<int>(32,32,32,3,100,
        CPUCenterDeriv<int>::PreStepType::kNone,
        CPUCenterDeriv<int>::StepType::kNaive,
        CPUCenterDeriv<int>::PostStepType::kNone
        ) );


  for( auto const& test: tests){
    test->Malloc();

    for( int dim = 0; dim < 4; dim++){
      double time = test->RunTest(dim);
      cout <<time<<endl;
    }
    test->Free();
    delete test;
  }*/


  return 0;
}
