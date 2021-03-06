#ifndef CUDA_CENTER_DERIV_H_
#define CUDA_CENTER_DERIV_H_

#include <chrono>
#include <ostream>
#include <string>
#include <sstream>

#include "Test.hpp"
#include "TypeName.hpp"


//Class for tests on CUDA for centered 2nd derivative
template <class T>
class CUDACenterDeriv : public Test{

  public:
    //Size of stencil
    const unsigned int stencil_size_;

    //Type of memory to use
    enum class MemType: int{
      kMalloc,kPinned,kUVM
    };
    const MemType mem_type_;
    std::string ToString(MemType type){
      switch(type){
        case MemType::kMalloc:
          return "kMalloc";
        case MemType::kPinned:
          return "kPinned";
        case MemType::kUVM:
          return "kUVM";
      }
    }

    const PreStepType pre_step_type_;
    //PostStep,Step,PreStep Types
    enum class PreStepType: int{
      kNone,kBoundaries,kMPIBoundaries,kPrefetech,kAsyncMemcpy,
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
      case PreStepType::kPrefetch:
        return "kPrefetch";
      case PreStepType::kAsyncMemcpy:
        return "kAsyncMemcpy";
      }
      return "UNKNOWN";
    }

    enum class StepType: int{
      kNaive,kShared
    };
    const StepType step_type_;

    std::string ToString(StepType type){
      switch(type){
      case StepType::kNaive:
        return "kNaive";
      case StepType::kShared:
        return "kShared";
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

    //Data to differentiate (either on device or UVM)
    T *u_,*u2_;
    //Copy of memory space on the host
    T *host_u_,*host_u2_;
    
    //Timers 
    std::chrono::high_resolution_clock::time_point startTime,endTime;

    CUDACenterDeriv(unsigned int ni, unsigned int nj, unsigned int nk, 
                   unsigned int stencil_size, unsigned int nsteps,
                   MemType mem_type,
                   PreStepType pre_step_type, 
                   StepType step_type, 
                   PostStepType post_step_type):
        Test(ni,nj,nk,(stencil_size-1)/2,
            nsteps,
            stencil_size*2-1, //flops_per_cell
            (stencil_size*2-1)/(sizeof(T)*stencil_size)),//arith_intensity
            stencil_size_(stencil_size),
            mem_type_(mem_type),
            pre_step_type_(pre_step_type),
            step_type_(step_type),
            post_step_type_(post_step_type)
        {}



    //Allocate Memory
    virtual int Malloc(){

      switch( mem_type_){
        case MemType::kMalloc:
          //Allocate memory normally, with a device and host space
          host_u_ = new T[size_];
          host_u2_ = new T[size_];

          CheckCuda( cudaMalloc(&u_ ,size_*sizeof(T)) );
          CheckCuda( cudaMalloc(&u2_,size_*sizeof(T)) );
          break;
        case MemType::kPinned:
          //Use pinned memory for host data
          CheckCuda( cudaMallocHost( (void**)&host_u_, size_*sizeof(T) ) );
          CheckCuda( cudaMallocHost( (void**)&host_u2_, size_*sizeof(T) ) );

          CheckCuda( cudaMalloc(&u_ ,size_*sizeof(T)) );
          CheckCuda( cudaMalloc(&u2_,size_*sizeof(T)) );
          break;
        case MemType::
        default:
          std::stringstream ss;
          ss  << "MemType '"
              << ToString(mem_type_)
              << "' unsupported!";
          throw ss.str();
          break;
      }
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
          CUDANaiveCenterDeriv(dim);
          break;
        case StepType::kNaiveOMP:
          CUDANaiveOMPCenterDeriv(dim);
          break;
        case StepType::kNaiveSIMD:
          CUDANaiveSIMDCenterDeriv(dim);
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
    void CUDANaiveCenterDeriv(int dim);


    virtual void PrintTest(std::ostream& os){
      os <<"CUDACenterDeriv<"<< TypeName<T>() <<">";
      Test::PrintTest(os);
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
            os<< host_u_[i + ni_*(j + nj_*k)];
          }
          os<<std::endl;
        }
        os<<std::endl;//Double space for z breaks
      }
      os<<std::endl;//Triple space

      for(int k = 0; k < nk_; k++){
        for(int j = 0; j < nj_; j++){
          for(int i = 0; i < ni_; i++){
            os<< host_u2_[i + ni_*(j + nj_*k)];
          }
          os<<std::endl;
        }
        os<<std::endl;//Double space for z breaks
      }
    }


};

#endif //CUDA_CENTER_DERIV_H_
