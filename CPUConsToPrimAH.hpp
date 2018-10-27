#ifndef CPU_CONS_TO_PRIM_AH_H_
#define CPU_CONS_TO_PRIM_AH_H_

#include <chrono>
#include <ostream>
#include <string>
#include <sstream>

#include "Test.hpp"
#include "TypeName.hpp"


#define NHYDRO 5

//Indices for conserved variables
#define IDN 0
#define IM1 1
#define IM2 2
#define IM3 3
#define IEN 3

//Indices for primitive variables
#define IVX 1
#define IVY 2
#define IVZ 3
#define IPR 4

//Class for tests on CPU for centered 2nd derivative
template <class T>
class CPUConsToPrimAH : public Test{

  public:
    //Size of the grid
    const unsigned int nx_,ny_,nz_,size_;

    //Starting and ending indicies [start,end)
    const unsigned int is_,ie_,js_,je_,ks_,ke_;

    //Number of fluid variables
    const unsigned nvars_ = NHYDRO;

    //PostStep,Step,PreStep Types
    enum class PreStepType: int{
      kNone,
    };
    const PreStepType pre_step_type_;

    std::string ToString(PreStepType type){
      switch(type){
      case PreStepType::kNone:
        return "kNone";
      case PreStepType::kBoundaries:
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
      kNone,
    };
    const PostStepType post_step_type_;

    std::string ToString(PostStepType type){
      switch(type){
      case PostStepType::kNone:
        return "kNone";
      }
      return "UNKNOWN";
    }

    //The data to differentiate
    T *u_,*u2_;
    
    //Timers 
    std::chrono::high_resolution_clock::time_point startTime,endTime;

    CPUConsToPrimAH(unsigned int nx, unsigned int ny, unsigned int nz, 
                    unsigned int is, unsigned int ie,
                    unsigned int js, unsigned int je,
                    unsigned int ks, unsigned int ke,
                   unsigned int nsteps,
                   PreStepType pre_step_type, 
                   StepType step_type, 
                   PostStepType post_step_type):
        Test( (ie -is)*(je-js)*(ke-ks),
              nsteps, 1,
              0, //flops_per_cell
              0),//arith_intensity
            nx_(nx),ny_(ny),nz_(nz), size_(nx*ny*nz),
            is_(is),ie_(ie),
            js_(js),je_(je),
            ks_(ks),ke_(ke),
            pre_step_type_(pre_step_type),step_type_(step_type),post_step_type_(post_step_type)
        {}



    //Allocate Memory
    virtual int Malloc(){
      cons_  = new T[size_*nvars_];
      prim_ = new T[size_*nvars_];

      //Set conserved data to something reasonable
      for(knt k = 0; k < nz_; k++){
        for(jnt j = 0; j < ny_; j++){
          for(int i = 0; i < nx_; i++){
          }
        }
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
      delete[] cons_;
      delete[] prim_;

      return 0;
    }

    //Different Kernel implementations
    
    //Naive triple for-loop
    void CPUNaiveConsToPrimAH(int dim);

    //OMP naive triple for-loop
    void CPUNaiveOMPConsToPrimAH(int dim);

    //NaiveSIMD on inner for-loop
    void CPUNaiveSIMDCenterDeriv(int dim);

    virtual void PrintTest(std::ostream& os){
      os <<"CPUConsToPrimAH<"<< TypeName<T>() <<">";
      Test::PrintTest(os);
      os << "nx_=" << nx_ <<"\t";
      os << "ny_=" << ny_ <<"\t";
      os << "nz_=" << nz_ <<"\t";
      os << "size_=" << size_ <<"\t";
      os << "is_=" << is_ <<"\t";
      os << "ie_=" << ie_ <<"\t";
      os << "js_=" << js_ <<"\t";
      os << "je_=" << je_ <<"\t";
      os << "ks_=" << ks_ <<"\t";
      os << "ke_=" << ke_ <<"\t";
      os << "pre_step_type_=" << ToString(pre_step_type_)<<"\t";
      os << "step_type_=" << ToString(step_type_)<<"\t";
      os << "post_step_type_=" << ToString(post_step_type_)<<"\t";
    }

    virtual void PrintU(std::ostream& os){
      os<<"#"<< nx_ << " "<< ny_ << " " << nz_ << " "<<std::endl;

      os<<std::scientific;
      std::cout.precision(15);

      for(int k = 0; k < nz_; k++){
        for(int j = 0; j < ny_; j++){
          for(int i = 0; i < nx_; i++){
            os<< cons_[i + nx_*(j + ny_*k)];
          }
          os<<std::endl;
        }
        os<<std::endl;//Double space for z breaks
      }
      os<<std::endl;//Triple space

      for(int k = 0; k < nz_; k++){
        for(int j = 0; j < ny_; j++){
          for(int i = 0; i < nx_; i++){
            os<< prim_[i + nx_*(j + ny_*k)];
          }
          os<<std::endl;
        }
        os<<std::endl;//Double space for z breaks
      }
    }


};

#endif //CPU_CONS_TO_PRIM_AH_H_
