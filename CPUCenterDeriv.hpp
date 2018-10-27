#ifndef CPU_CENTER_DERIV_H_
#define CPU_CENTER_DERIV_H_

#include <chrono>
#include <ostream>
#include <string>
#include <sstream>

#include "Test.hpp"
#include "TypeName.hpp"


//Class for tests on CPU for centered 2nd derivative
template <class T>
class CPUCenterDeriv : public Test{

  public:
    //Size of the grid
    const unsigned int ni_,nj_,nk_,size_;

    //Size of stencil
    const unsigned int stencil_size_;

    //Number of Ghost zones
    const unsigned int ng_;

    //PostStep,Step,PreStep Types
    enum class PreStepType: int{
      kNone,kBoundaries,kMPIBoundaries,
    };
    const PreStepType pre_step_type_;

    std::string ToString(PreStepType type){
      switch(type){
      case PreStepType::kNone:
        return "kNone";
      case PreStepType::kBoundaries:
        return "kBoundaries";
      case PreStepType::kMPIBoundaries:
        return "kMPIBoundaries";
      }
      return "UNKNOWN";
    }

    enum class StepType: int{
      kNaive,kNaiveOMP,kNaiveSIMD,
    };
    const StepType step_type_;

    std::string ToString(StepType type){
      switch(type){
      case StepType::kNaive:
        return "kNaive";
      case StepType::kNaiveOMP:
        return "kNaiveOMP";
      case StepType::kNaiveSIMD:
        return "kNaiveSIMD";
      }
      return "UNKNOWN";
    }


    enum class PostStepType: int{
      kNone,kBoundaries,kMPIBoundaries,
    };
    const PostStepType post_step_type_;

    std::string ToString(PostStepType type){
      switch(type){
      case PostStepType::kNone:
        return "kNone";
      case PostStepType::kBoundaries:
        return "kBoundaries";
      case PostStepType::kMPIBoundaries:
        return "kMPIBoundaries";
      }
      return "UNKNOWN";
    }

    //The data to differentiate
    T *u_,*u2_;
    
    //Timers 
    std::chrono::high_resolution_clock::time_point startTime,endTime;

    CPUCenterDeriv(unsigned int ni, unsigned int nj, unsigned int nk, 
                   unsigned int stencil_size, unsigned int nsteps,
                   PreStepType pre_step_type, 
                   StepType step_type, 
                   PostStepType post_step_type):
        Test(  (ni - (stencil_size-1)/2)*
               (nj - (stencil_size-1)/2)*
               (nk - (stencil_size-1)/2),
              nsteps, 4,
              stencil_size*2-1, //flops_per_cell
              (stencil_size*2-1)/(sizeof(T)*stencil_size)),//arith_intensity
            ni_(ni),nj_(nj),nk_(nk), size_(ni*nj*nk),
            stencil_size_(stencil_size), ng_((stencil_size-1)/2),
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
          ss  << "PreStepType '"
              << ToString(pre_step_type_)
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
        case StepType::kNaiveSIMD:
          CPUNaiveSIMDCenterDeriv(dim);
          break;
        default:
          std::stringstream ss;
          ss  << "PostStepType '"
              << ToString(step_type_)
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
              << ToString(post_step_type_)
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
      return std::chrono::duration_cast<std::chrono::nanoseconds>( 
          endTime - startTime ).count()/1e9;
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
    void CPUNaiveOMPCenterDeriv(int dim);

    //NaiveSIMD on inner for-loop
    void CPUNaiveSIMDCenterDeriv(int dim);

    virtual void PrintTest(std::ostream& os){
      os <<"CPUCenterDeriv<"<< TypeName<T>() <<">";
      Test::PrintTest(os);
      os << "ni_=" << ni_ <<"\t";
      os << "nj_=" << nj_ <<"\t";
      os << "nk_=" << nk_ <<"\t";
      os << "size_=" << size_ <<"\t";
      os << "ng_=" << ng_ <<"\t";
      os << "stencil_size_" << stencil_size_<<"\t";
      os << "pre_step_type_=" << ToString(pre_step_type_)<<"\t";
      os << "step_type_=" << ToString(step_type_)<<"\t";
      os << "post_step_type_=" << ToString(post_step_type_)<<"\t";
    }

    virtual void PrintU(std::ostream& os){
      os<<"#"<< ni_ << " "<< nj_ << " " << nk_ << " "<<std::endl;

      os<<std::scientific;
      std::cout.precision(15);

      for(int k = 0; k < nk_; k++){
        for(int j = 0; j < nj_; j++){
          for(int i = 0; i < ni_; i++){
            os<< u_[i + ni_*(j + nj_*k)];
          }
          os<<std::endl;
        }
        os<<std::endl;//Double space for z breaks
      }
      os<<std::endl;//Triple space

      for(int k = 0; k < nk_; k++){
        for(int j = 0; j < nj_; j++){
          for(int i = 0; i < ni_; i++){
            os<< u2_[i + ni_*(j + nj_*k)];
          }
          os<<std::endl;
        }
        os<<std::endl;//Double space for z breaks
      }
    }


};

#endif //CPU_CENTER_DERIV_H_
