#ifndef Kokkos_CONS_TO_PRIM_AH_H_
#define Kokkos_CONS_TO_PRIM_AH_H_

#include <chrono>
#include <ostream>
#include <string>
#include <sstream>
#include <cmath>

#include "Kokkos_Core.hpp"

#include "../common/Test.hpp"
#include "../common/TypeName.hpp"
#include "../common/Matrix.hpp"


ENABLE_TYPENAME(Kokkos::LayoutLeft);
ENABLE_TYPENAME(Kokkos::LayoutRight);

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




//Class for tests on Kokkos for centered 2nd derivative
template <class T, class Layout=Kokkos::LayoutLeft,class IType=int64_t, 
         Kokkos::Iterate ItOuter=Kokkos::Iterate::Default, Kokkos::Iterate ItInner=Kokkos::Iterate::Default>
class KokkosConsToPrimAH : public Test{
    const unsigned int id_;

  public:
    //Size of the grid
    const unsigned int ni_,nj_,nk_,size_;

    //Starting and ending indicies [start,end]
    const unsigned int is_,ie_,js_,je_,ks_,ke_;

    //Interior size
    const unsigned int mi_,mj_,mk_;

    //Number of fluid variables
    const unsigned int nvars_ = NHYDRO;

    //MDRange Parameters
    const Kokkos::Array<IType,3> tiling_;

    std::string ToString( Kokkos::Iterate it){
      switch(it){
        case Kokkos::Iterate::Default:
          return "Kokkos::Iterate::Default";
        case Kokkos::Iterate::Left:
          return "Kokkos::Iterate::Left";
        case Kokkos::Iterate::Right:
          return "Kokkos::Iterate::Right";
        default:
          return "UNKNOWN ITERATOR";
      }
    }

    //1D Parameters
    //Chunk Size?

    //TVR Parameters (and TTR?)
    const unsigned int vector_length_;

    //Fluid constants
    const double density_floor_ = 0.01;
    const double pressure_floor_ = 0.01;
    const double gm1_ = 1.666666667;

    //PostStep,Step,PreStep Types
    enum class PreStepType: int{
      kNone,kCopyFromHost,
    };
    const PreStepType pre_step_type_;

    std::string ToString(PreStepType type){
      switch(type){
      case PreStepType::kNone:
        return "kNone";
      case PreStepType::kCopyFromHost:
        return "kCopyFromHost";
      }
      return "UNKNOWN";
    }

    enum class StepType: int{
      kMDRange,k1DRange,kTVR,kTTR,
    };
    const StepType step_type_;

    std::string ToString(StepType type){
      switch(type){
      case StepType::kMDRange:
        return "kMDRange";
      case StepType::k1DRange:
        return "k1DRange";
      case StepType::kTVR:
        return "kTVR";
      case StepType::kTTR:
        return "kTTR";
      }
      return "UNKNOWN";
    }


    enum class PostStepType: int{
      kNone,kCopyToHost
    };
    const PostStepType post_step_type_;

    std::string ToString(PostStepType type){
      switch(type){
      case PostStepType::kNone:
        return "kNone";
      case PostStepType::kCopyToHost:
        return "kCopyToHost";
      }
      return "UNKNOWN";
    }

    typedef Kokkos::DefaultExecutionSpace DevSpace;

    //The Conserved and Primitive data
    Kokkos::View<T****, Layout,DevSpace> cons_,prim_;

    //Copy of data on host
    Kokkos::View<T****, Layout,Kokkos::HostSpace> h_cons_,h_prim_;
    
    //Timers 
    std::chrono::high_resolution_clock::time_point startTime_,endTime_;

    //MDRange Constuctor
    KokkosConsToPrimAH(unsigned int ni, unsigned int nj, unsigned int nk, 
                    unsigned int is, unsigned int ie,
                    unsigned int js, unsigned int je,
                    unsigned int ks, unsigned int ke,
                    unsigned int nsteps,
                   Kokkos::Array<IType,3> tiling,
                   PreStepType pre_step_type, 
                   StepType step_type, 
                   PostStepType post_step_type,
                   int id):
        Test( (ie+1-is)*(je+1-js)*(ke+1-ks),
              nsteps, 1,
              0, //flops_per_cell
              0),//arith_intensity
            ni_(ni),nj_(nj),nk_(nk), size_(ni*nj*nk),
            is_(is),ie_(ie),
            js_(js),je_(je),
            ks_(ks),ke_(ke),
            mi_(ie+1-is),mj_(je+1-js),mk_(ke+1-ks),
            tiling_(tiling),
            vector_length_(0),
            pre_step_type_(pre_step_type),
            step_type_(step_type),post_step_type_(post_step_type),
            //cons_(nvars_,nk_,nj_,ni_),prim_(nvars_,nk_,nj_,ni_),
            id_(id)
        {
          if(step_type != StepType::kMDRange && step_type != StepType::k1DRange){
            std::stringstream ss;
            ss  << "Incorrect constructor for '"
                << ToString(step_type_)
                << "' step_type";
            throw ss.str();
          }
        }
    //1D Constuctor
    KokkosConsToPrimAH(unsigned int ni, unsigned int nj, unsigned int nk, 
                    unsigned int is, unsigned int ie,
                    unsigned int js, unsigned int je,
                    unsigned int ks, unsigned int ke,
                    unsigned int nsteps,
                   PreStepType pre_step_type, 
                   StepType step_type, 
                   PostStepType post_step_type,
                   int id):
        Test( (ie+1-is)*(je+1-js)*(ke+1-ks),
              nsteps, 1,
              0, //flops_per_cell
              0),//arith_intensity
            ni_(ni),nj_(nj),nk_(nk), size_(ni*nj*nk),
            is_(is),ie_(ie),
            js_(js),je_(je),
            ks_(ks),ke_(ke),
            mi_(ie+1-is),mj_(je+1-js),mk_(ke+1-ks),
            tiling_({0,0,0}),
            vector_length_(0),
            pre_step_type_(pre_step_type),
            step_type_(step_type),post_step_type_(post_step_type),
            //cons_(nvars_,nk_,nj_,ni_),prim_(nvars_,nk_,nj_,ni_),
            id_(id)
        {
          if(step_type != StepType::kMDRange && step_type != StepType::k1DRange){
            std::stringstream ss;
            ss  << "Incorrect constructor for '"
                << ToString(step_type_)
                << "' step_type";
            throw ss.str();
          }
        }

    //TTR/TVR Contructor
    KokkosConsToPrimAH(unsigned int ni, unsigned int nj, unsigned int nk, 
                    unsigned int is, unsigned int ie,
                    unsigned int js, unsigned int je,
                    unsigned int ks, unsigned int ke,
                    unsigned int nsteps,
                   unsigned int vector_length,
                   PreStepType pre_step_type, 
                   StepType step_type, 
                   PostStepType post_step_type,
                   int id):
        Test( (ie+1-is)*(je+1-js)*(ke+1-ks),
              nsteps, 1,
              0, //flops_per_cell
              0),//arith_intensity
            ni_(ni),nj_(nj),nk_(nk), size_(ni*nj*nk),
            is_(is),ie_(ie),
            js_(js),je_(je),
            ks_(ks),ke_(ke),
            mi_(ie+1-is),mj_(je+1-js),mk_(ke+1-ks),
            tiling_({0,0,0}),
            vector_length_(vector_length),
            pre_step_type_(pre_step_type),
            step_type_(step_type),post_step_type_(post_step_type),
            //cons_(nvars_,nk_,nj_,ni_),prim_(nvars_,nk_,nj_,ni_),
            id_(id)
        {
          if(step_type != StepType::kTVR && step_type_ != StepType::kTTR){
            std::stringstream ss;
            ss  << "Incorrect constructor for '"
                << ToString(step_type_)
                << "' step_type";
            throw ss.str();
          }
        }



    //Allocate Memory
    virtual int Malloc(){
      cons_ = Kokkos::View<T****, Layout,DevSpace>("cons_",nvars_,nk_,nj_,ni_);
      prim_ = Kokkos::View<T****, Layout,DevSpace>("prim_",nvars_,nk_,nj_,ni_);
      h_cons_ = Kokkos::create_mirror_view(cons_);
      h_prim_ = Kokkos::create_mirror_view(prim_);

      //Set conserved data to something reasonable
      for(int k = 0; k < nk_; k++){
        for(int j = 0; j < nj_; j++){
          for(int i = 0; i < ni_; i++){
            T c = (i + ni_*(j + nj_*k) + 1.0)/size_;
            h_cons_(IDN,k,j,i) = c;
            h_cons_(IM1,k,j,i) = sin(c);
            h_cons_(IM2,k,j,i) = cos(c);
            h_cons_(IM3,k,j,i) = tan(c);
            h_cons_(IEN,k,j,i) = c*c+4.0;
          }
        }
      }
      //Set primitive data to garbage
      for(int l = 0; l < nvars_; l++){
        for(int k = 0; k < nk_; k++){
          for(int j = 0; j < nj_; j++){
            for(int i = 0; i < ni_; i++){
              h_prim_(l,k,j,i) = (l+1)*-111;
            }
          }
        }
      }

      Kokkos::deep_copy(cons_,h_cons_);
      Kokkos::deep_copy(prim_,h_prim_);

      return 0;
    }

    //Start timing (from 0)
    virtual void StartTest(int dim){
      //Sync to device?
      Kokkos::fence();
      startTime_ = std::chrono::high_resolution_clock::now();
    }


    //preStep things (borders, prefetching, copy to-from host/device memory)
    virtual void PreStep(int dim){
      switch(pre_step_type_){ 
        case PreStepType::kNone: 
          break; 
        case PreStepType::kCopyFromHost:
          Kokkos::deep_copy(cons_,h_cons_);
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
        case StepType::kMDRange:
          KokkosMDRangeConsToPrimAH(dim);
          break;
        case StepType::k1DRange:
          Kokkos1DRangeConsToPrimAH(dim);
          break;
        case StepType::kTVR:
          KokkosTVRConsToPrimAH(dim);
          break;
        case StepType::kTTR:
          KokkosTTRConsToPrimAH(dim);
          break;
        default:
          std::stringstream ss;
          ss  << "StepType '"
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
        case PostStepType::kCopyToHost:
          Kokkos::deep_copy(h_prim_,prim_);
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
      Kokkos::fence();
      endTime_ = std::chrono::high_resolution_clock::now();
    }

    //Get the seconds between the last start and end
    virtual double ElapsedTime(){
      return std::chrono::duration_cast<std::chrono::nanoseconds>( 
          endTime_ - startTime_ ).count()/1e9;
    }

    //Release memory
    virtual int Free(){
      //Free Kokkos data, since the memory might be needed before this object
      //(and the Views) are destroyed.
      //(is there a better way to do this?
      Kokkos::resize(cons_,0,0,0,0);
      Kokkos::resize(prim_,0,0,0,0);
      Kokkos::resize(h_cons_,0,0,0,0);
      Kokkos::resize(h_prim_,0,0,0,0);

      return 0;
    }

    //Different Kernel implementations
    
    //MDRangePolicy with Kokkos
    void KokkosMDRangeConsToPrimAH(int dim);
    
    //1D Range with Kokkos
    void Kokkos1DRangeConsToPrimAH(int dim);

    //Team Vector Range with Kokkos
    void KokkosTVRConsToPrimAH(int dim);

    //Team Thread Range with Kokkos
    void KokkosTTRConsToPrimAH(int dim);


    virtual void PrintTest(std::ostream& os){
      os <<"KokkosConsToPrimAH<";
      os << TypeName<T>() << " , ";
      os << TypeName<Layout>() << " , ";
      os << TypeName<IType>() << " , ";
      os << ToString(ItOuter) <<" , ";
      os << ToString(ItInner);
      os << ">";
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
      os << "ItOuter_=" << ToString(ItOuter) <<"\t";
      os << "ItInner_=" << ToString(ItInner) <<"\t";
      os << "tiling_=" << tiling_[0] << "," << tiling_[1] << "," << tiling_[2]<<"\t";
      os << "vector_length_=" << vector_length_ <<"\t";
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

#endif //Kokkos_CONS_TO_PRIM_AH_H_
