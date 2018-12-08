#ifndef OpenACC_CONS_TO_PRIM_AH_H_
#define OpenACC_CONS_TO_PRIM_AH_H_

#include <chrono>
#include <ostream>
#include <string>
#include <sstream>
#include <cmath>

#include "../common/Test.hpp"
#include "../common/TypeName.hpp"
#include "../common/Matrix.hpp"


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


//Class for tests on OpenACC for centered 2nd derivative
template <class T>
class OpenACCConsToPrimAH : public Test{

  public:
    //Size of the grid
    const unsigned int ni_,nj_,nk_,size_;

    //Starting and ending indicies [start,end]
    const unsigned int is_,ie_,js_,je_,ks_,ke_;

    //Interior size
    const unsigned int mi_,mj_,mk_;

    //Number of fluid variables
    const unsigned nvars_ = NHYDRO;

    const double density_floor_ = 0.01;
    const double pressure_floor_ = 0.01;
    const double gm1_ = 1.666666667;

    //PostStep,Step,PreStep Types
    enum class PreStepType: int{
      kNone,
    };
    const PreStepType pre_step_type_;

    std::string ToString(PreStepType type){
      switch(type){
      case PreStepType::kNone:
        return "kNone";
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

    //The Conserved and Primitive data
    Matrix<T> cons_,prim_;
    
    //Timers 
    std::chrono::high_resolution_clock::time_point startTime_,endTime_;

    OpenACCConsToPrimAH(unsigned int ni, unsigned int nj, unsigned int nk, 
                    unsigned int is, unsigned int ie,
                    unsigned int js, unsigned int je,
                    unsigned int ks, unsigned int ke,
                   unsigned int nsteps,
                   PreStepType pre_step_type, 
                   StepType step_type, 
                   PostStepType post_step_type):
        Test( (ie+1-is)*(je+1-js)*(ke+1-ks),
              nsteps, 1,
              0, //flops_per_cell
              0),//arith_intensity
            ni_(ni),nj_(nj),nk_(nk), size_(ni*nj*nk),
            is_(is),ie_(ie),
            js_(js),je_(je),
            ks_(ks),ke_(ke),
            mi_(ie+1-is),mj_(je+1-js),mk_(ke+1-ks),
            pre_step_type_(pre_step_type),
            step_type_(step_type),post_step_type_(post_step_type),
            cons_(nvars_,nk_,nj_,ni_),prim_(nvars_,nk_,nj_,ni_)
        {}



    //Allocate Memory
    virtual int Malloc(){
      cons_.Malloc();
      prim_.Malloc();

      //Set conserved data to something reasonable
      for(int k = 0; k < nk_; k++){
        for(int j = 0; j < nj_; j++){
          for(int i = 0; i < ni_; i++){
            T c = (i + ni_*(j + nj_*k) + 1.0)/size_;
            cons_(IDN,k,j,i) = c;
            cons_(IM1,k,j,i) = sin(c);
            cons_(IM2,k,j,i) = cos(c);
            cons_(IM3,k,j,i) = tan(c);
            cons_(IEN,k,j,i) = c*c+4.0;
          }
        }
      }
      //Set primitive data to garbage
      for(int l = 0; l < nvars_; l++){
        for(int k = 0; k < nk_; k++){
          for(int j = 0; j < nj_; j++){
            for(int i = 0; i < ni_; i++){
              prim_(l,k,j,i) = (l+1)*-111;
            }
          }
        }
      }

      return 0;
    }

    //Start timing (from 0)
    virtual void StartTest(int dim){
      startTime_ = std::chrono::high_resolution_clock::now();
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
          OpenACCNaiveConsToPrimAH(dim);
          break;
        case StepType::kNaiveOMP:
          OpenACCNaiveOMPConsToPrimAH(dim);
          break;
        case StepType::kNaiveSIMD:
          OpenACCNaiveSIMDConsToPrimAH(dim);
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
      endTime_ = std::chrono::high_resolution_clock::now();
    }

    //Get the seconds between the last start and end
    virtual double ElapsedTime(){
      return std::chrono::duration_cast<std::chrono::nanoseconds>( 
          endTime_ - startTime_ ).count()/1e9;
    }

    //Release memory
    virtual int Free(){
      cons_.Free();
      prim_.Free();

      return 0;
    }

    //Different Kernel implementations
    
    //Naive triple for-loop
    void OpenACCNaiveConsToPrimAH(int dim);

    //OMP naive triple for-loop
    void OpenACCNaiveOMPConsToPrimAH(int dim);

    //NaiveSIMD on inner for-loop
    void OpenACCNaiveSIMDConsToPrimAH(int dim);

    virtual void PrintTest(std::ostream& os){
      os <<"OpenACCConsToPrimAH<"<< TypeName<T>() <<">";
      Test::PrintTest(os);
      os << "ni_=" << ni_ <<"\t";
      os << "nj_=" << nj_ <<"\t";
      os << "nk_=" << nk_ <<"\t";
      os << "size_=" << size_ <<"\t";
      os << "is_=" << is_ <<"\t";
      os << "ie_=" << ie_ <<"\t";
      os << "js_=" << js_ <<"\t";
      os << "je_=" << je_ <<"\t";
      os << "ks_=" << ks_ <<"\t";
      os << "ke_=" << ke_ <<"\t";
      os << "nvars_=" << nvars_ <<"\t";
      os << "pre_step_type_=" << ToString(pre_step_type_)<<"\t";
      os << "step_type_=" << ToString(step_type_)<<"\t";
      os << "post_step_type_=" << ToString(post_step_type_)<<"\t";
    }

    virtual void PrintU(std::ostream& os){
      os<<"#"<< ni_ << " "<< nj_ << " " << nk_ << " " << nvars_<<std::endl;

      os<<std::scientific;
      std::cout.precision(15);

      for(int l = 0; l < nvars_; l++){
        for(int k = 0; k < nk_; k++){
          for(int j = 0; j < nj_; j++){
            for(int i = 0; i < ni_; i++){
              os<< cons_(l,k,j,i);
            }
            os<<std::endl;
          }
          os<<std::endl;//Double space for z breaks
        }
        os<<std::endl;//Triple space for var breaks
      }
      os<<std::endl;//Quad space

      for(int l = 0; l < nvars_; l++){
        for(int k = 0; k < nk_; k++){
          for(int j = 0; j < nj_; j++){
            for(int i = 0; i < ni_; i++){
              os<< prim_(l,k,j,i);
            }
            os<<std::endl;
          }
          os<<std::endl;//Double space for z breaks
        }
        os<<std::endl;//Triple space for var breaks
      }
      os<<std::endl;//Quad space
    }


};

#endif //OpenACC_CONS_TO_PRIM_AH_H_
