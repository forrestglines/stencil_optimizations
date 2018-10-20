#ifndef CPU_CENTER_DERIV_H_
#define CPU_CENTER_DERIV_H_

#include <chrono>
#include <ostream>
#include <string>
#include <sstream>

#include "Test.hpp"


//Class for tests on CPU for centered 2nd derivative
template <class T>
class CPUCenterDeriv : public Test{

  public:
    //Size of stencil
    const unsigned int stencil_size_;

    //PostStep,Step,PreStep Types
    enum class PreStepType: int{
      kNone,kBoundaries,kMPIBoundaries,
    };
    const PreStepType pre_step_type_;

    enum class StepType: int{
      kNaive,kNaiveOMP,kSIMD,
    };
    const StepType step_type_;

    enum class PostStepType: int{
      kNone,kBoundaries,kMPIBoundaries,
    };
    const PostStepType post_step_type_;

    //The data to differentiate
    T *u_,*u2_;
    
    //Timers 
    std::chrono::high_resolution_clock::time_point startTime,endTime;

    CPUCenterDeriv(unsigned int nx, unsigned int ny, unsigned int nz, 
                   unsigned int stencil_size, unsigned int nsteps,
                   PreStepType pre_step_type, 
                   StepType step_type, 
                   PostStepType post_step_type):
        Test(nx,ny,nz,(stencil_size-1)/2,
            nsteps,
            stencil_size*2-1, //flops_per_cell
            (stencil_size*2-1)/(sizeof(T)*stencil_size)),//arith_intensity
            stencil_size_(stencil_size),
            pre_step_type_(pre_step_type),step_type_(step_type),post_step_type_(post_step_type)
        {}



    //Allocate Memory
    virtual int Malloc(){
      u_  = new T[size_];
      u2_ = new T[size_];

      return 0;
    }

    //Start timing (from 0)
    virtual void StartTest(int dim){
      startTime = std::chrono::high_resolution_clock::now();
    }


    //preStep things (borders, prefetching, copy to-from host/device memory)
    virtual void PreStep(int dim){
      switch(pre_step_type_){ 
        case PreStepType::kNone: 
          break; 
        default:
          std::stringstream ss;
          ss  << "PostStepType '"
              << static_cast<typename std::underlying_type<T>::type>(post_step_type_)
              << "' unsupported!";
          throw ss.str();
          break;
      }
    }

    //Perform a step
    virtual void Step(int dim){
      switch(step_type_){
        case StepType::kNaive:
          CPUNaiveCenterDeriv(dim);
          break;
        case StepType::kNaiveOMP:
          CPUNaiveOMPCenterDeriv(dim);
          break;
        case StepType::kSIMD:
          CPUSIMDCenterDeriv(dim);
          break;
        default:
          std::stringstream ss;
          ss  << "PostStepType '"
              << static_cast<typename std::underlying_type<T>::type>(post_step_type_)
              << "' unsupported!";
          throw ss.str();
          break;
      }
    }

    //preStep things (borders, prefetching, copy to-from host/device memory)
    virtual void PostStep(int dim){
      switch(post_step_type_){ 
        case PostStepType::kNone: 
          break; 
        default:
          std::stringstream ss;
          ss  << "PostStepType '"
              << static_cast<typename std::underlying_type<T>::type>(post_step_type_)
              << "' unsupported!";
          throw ss.str();
          break;
      }
    }

    //End timing
    virtual void EndTest(int dim){
      endTime = std::chrono::high_resolution_clock::now();
    }

    //Get the seconds between the last start and end
    virtual double ElapsedTime(){
      return std::chrono::duration_cast<std::chrono::seconds>( 
          endTime - startTime ).count();
    }

    //Release memory
    virtual int Free(){
      delete[] u_;
      delete[] u2_;

      return 0;
    }

    //Different Kernel implementations
    
    //Naive triple for-loop
    void CPUNaiveCenterDeriv(int dim);

    //OMP naive triple for-loop
    void CPUNaiveOMPCenterDeriv(int dim){}

    //SIMD on inner for-loop
    void CPUSIMDCenterDeriv(int dim){}

  protected:
    virtual void Print(std::ostream& os){
      os<<"#"<< nx_ << " "<< ny_ << " " << nz_ << " "<<std::endl;

      os<<std::scientific;
      std::cout.precision(15);

      for(int k = 0; k < nz_; k++){
        for(int j = 0; j < ny_; j++){
          for(int i = 0; i < nx_; i++){
            os<< u_[i + nx_*(j + ny_*k)];
          }
          os<<std::endl;
        }
        os<<std::endl;//Double space for z breaks
      }
      os<<std::endl;//Triple space

      for(int k = 0; k < nz_; k++){
        for(int j = 0; j < ny_; j++){
          for(int i = 0; i < nx_; i++){
            os<< u2_[i + nx_*(j + ny_*k)];
          }
          os<<std::endl;
        }
        os<<std::endl;//Double space for z breaks
      }
    }


};

#endif //CPU_CENTER_DERIV_H_
