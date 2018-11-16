#ifndef CUDA_CONS_TO_PRIM_AH_H_
#define CUDA_CONS_TO_PRIM_AH_H_

#include <chrono>
#include <ostream>
#include <string>
#include <sstream>
#include <cmath>

#include "../common/Test.hpp"
#include "../common/TypeName.hpp"
#include "../common/Matrix.hpp"
#include "CheckCuda.cuh"

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


//Class for tests on CUDA for centered 2nd derivative
template <class T>
class CUDAConsToPrimAH : public Test{

  public:
    //Size of the grid
    const unsigned int ni_,nj_,nk_,size_;

    //Starting and ending indicies [start,end]
    const unsigned int is_,ie_,js_,je_,ks_,ke_;

    //Interior size
    const unsigned int mi_,mj_,mk_;

    //Max threads in a cuda block
    const unsigned int max_block_size_;

    //Number of fluid variables
    const unsigned int nvars_ = NHYDRO;

    //Size of one array in memory
    const unsigned int mem_size_;

    const double density_floor_ = 0.01;
    const double pressure_floor_ = 0.01;
    const double gm1_ = 1.666666667;


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
      return "UNKNOWN";
    }

    //PostStep,Step,PreStep Types
    enum class PreStepType: int{
      kNone,MemPrefetchAsync,MemcpyAsync
    };
    const PreStepType pre_step_type_;

    std::string ToString(PreStepType type){
      switch(type){
      case PreStepType::kNone:
        return "kNone";
      case PreStepType::MemPrefetchAsync:
        return "MemPrefetchAsync";
      case PreStepType::MemcpyAsync:
        return "MemcpyAsync";
      }
      return "UNKNOWN";
    }

    enum class StepType: int{
      kNaive,k1D,
    };
    const StepType step_type_;

    std::string ToString(StepType type){
      switch(type){
      case StepType::kNaive:
        return "kNaive";
      case StepType::k1D:
        return "k1D";
      }
      return "UNKNOWN";
    }


    enum class PostStepType: int{
      kNone,MemPrefetchAsync,MemcpyAsync
    };
    const PostStepType post_step_type_;

    std::string ToString(PostStepType type){
      switch(type){
      case PostStepType::kNone:
        return "kNone";
      case PostStepType::MemPrefetchAsync:
        return "MemPrefetchAsync";
      case PostStepType::MemcpyAsync:
        return "MemcpyAsync";
      }
      return "UNKNOWN";
    }

    //The Conserved and Primitive data on the host
    Matrix<T> cons_,prim_;
    T *h_cons_, *h_prim_; //If used, points to same data as matrix
      
    //The Conserved and Primitive data on the device
    T *d_cons_, *d_prim_;

    //Timers 
    cudaEvent_t start_time_, end_time_;

    CUDAConsToPrimAH(unsigned int ni, unsigned int nj, unsigned int nk, 
                     unsigned int is, unsigned int ie,
                     unsigned int js, unsigned int je,
                     unsigned int ks, unsigned int ke,
                     unsigned int max_block_size,
                     unsigned int nsteps,
                     MemType mem_type,
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
            max_block_size_(max_block_size),
            mi_(ie+1-is),mj_(je+1-js),mk_(ke+1-ks),
            mem_size_(size_*nvars_*sizeof(T)),
            mem_type_(mem_type),
            pre_step_type_(pre_step_type),
            step_type_(step_type),post_step_type_(post_step_type),
            cons_(nvars_,nk_,nj_,ni_),prim_(nvars_,nk_,nj_,ni_)
        {}



    //Allocate Memory
    virtual int Malloc(){
      switch( mem_type_){
        case MemType::kMalloc:
          //Allocate memory normally, with a device and host space
          cons_.Malloc();
          prim_.Malloc();

          CheckCuda( cudaMalloc(&d_cons_, mem_size_) );
          CheckCuda( cudaMalloc(&d_prim_, mem_size_) );
          break;
        case MemType::kPinned:
          //Use pinned memory for host data
          CheckCuda( cudaMallocHost( (void**)&h_cons_, mem_size_ ) );
          CheckCuda( cudaMallocHost( (void**)&h_prim_, mem_size_ ) );

          cons_.UseMem(h_cons_);
          prim_.UseMem(h_prim_);

          CheckCuda( cudaMalloc(&d_cons_,mem_size_) );
          CheckCuda( cudaMalloc(&d_prim_,mem_size_) );
          break;
        case MemType::kUVM:
          //Use UVM memory
          CheckCuda( cudaMallocManaged(&d_cons_,mem_size_) );
          CheckCuda( cudaMallocManaged(&d_prim_,mem_size_) );

          h_cons_ = d_cons_;
          h_prim_ = d_prim_;

          cons_.UseMem(h_cons_);
          prim_.UseMem(h_prim_);
          break;
        default:
          std::stringstream ss;
          ss  << "MemType '"
              << ToString(mem_type_)
              << "' unsupported!";
          throw ss.str();
          break;
      }


      //Set conserved data to something reasonable on the host
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

      int device;
      //Copy the data to device
      switch( mem_type_){
        case MemType::kMalloc:
        case MemType::kPinned:
          //With separate host and device memory, these do the same thing
          CheckCuda(cudaMemcpy(d_cons_,h_cons_,mem_size_,cudaMemcpyHostToDevice));
          CheckCuda(cudaMemcpy(d_prim_,h_prim_,mem_size_,cudaMemcpyHostToDevice));
          break;
        case MemType::kUVM:
          //Prefetch to device (or make this a test?)
          device = -1;
          CheckCuda(cudaGetDevice(&device));
          CheckCuda(cudaMemPrefetchAsync(d_cons_,mem_size_,device,NULL));
          CheckCuda(cudaMemPrefetchAsync(d_prim_,mem_size_,device,NULL));
          break;
        default:
          std::stringstream ss;
          ss  << "MemType '"
              << ToString(mem_type_)
              << "' unsupported!";
          throw ss.str();
          break;
      }

      //Move the constant symbols onto the device
      MemcpyConstants();

      //Create the cuda timers
      cudaEventCreate(&start_time_);
      cudaEventCreate(&end_time_);

      //Make sure everything is synced up, before timing
      cudaDeviceSynchronize();

      return 0;
    }

    void MemcpyConstants();

    //Start timing (from 0)
    virtual void StartTest(int dim){
      cudaEventRecord(start_time_);
    }


    //preStep things (borders, prefetching, copy to-from host/device memory)
    virtual void PreStep(int dim){
      int device;
      switch(pre_step_type_){ 
        case PreStepType::kNone: 
          break; 
        case PreStepType::MemPrefetchAsync:
          //Prefetch to device (or make this a test?)
          device = -1;
          CheckCuda(cudaGetDevice(&device));
          CheckCuda(cudaMemPrefetchAsync(d_cons_,mem_size_,device,NULL));
          break;
        case PreStepType::MemcpyAsync:
          //Memcpy to Device
          CheckCuda(cudaMemcpyAsync(d_cons_,h_cons_,mem_size_,cudaMemcpyHostToDevice));
          break;
        default:
          std::stringstream ss;
          ss  << "PreStepType '"
              << ToString(mem_type_)
              << "' unsupported!";
          throw ss.str();
          break;
      }
    }

    //Perform a step
    virtual void Step(int dim){
      switch(step_type_){
        case StepType::kNaive:
          CUDANaiveConsToPrimAH(dim);
          break;
        case StepType::k1D:
          CUDA1DConsToPrimAH(dim);
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
        case PostStepType::MemPrefetchAsync:
          //Prefetch to host
          CheckCuda(cudaMemPrefetchAsync(d_prim_,mem_size_,cudaCpuDeviceId,NULL));
          break;
        case PostStepType::MemcpyAsync:
          //Memcpy to host
          CheckCuda(cudaMemcpyAsync(h_prim_,d_prim_,mem_size_,cudaMemcpyDeviceToHost));
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
      cudaEventRecord(end_time_);
    }

    //Get the seconds between the last start and end
    virtual double ElapsedTime(){
      //Get the total time used on the GPU
      cudaEventSynchronize(end_time_);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start_time_,end_time_);

      return milliseconds/1000;

    }

    //Release memory
    virtual int Free(){
      switch( mem_type_){
        case MemType::kMalloc:
          cons_.Free();
          prim_.Free();

          CheckCuda(cudaFree(d_cons_));
          CheckCuda(cudaFree(d_prim_));
          break;
        case MemType::kPinned:
          CheckCuda(cudaFreeHost(h_cons_));
          CheckCuda(cudaFreeHost(h_prim_));

          CheckCuda(cudaFree(d_cons_));
          CheckCuda(cudaFree(d_prim_));
          break;
        case MemType::kUVM:
          CheckCuda(cudaFree(d_cons_));
          CheckCuda(cudaFree(d_prim_));
          break;
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

    //Different Kernel implementations
    
    //My naive implementation
    void CUDANaiveConsToPrimAH(int dim);

    //1D flattened indices implementation
    void CUDA1DConsToPrimAH(int dim);

    virtual void PrintTest(std::ostream& os){
      os <<"CUDAConsToPrimAH<"<< TypeName<T>() <<">";
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
      os << "mem_size_=" << mem_size_ <<"\t";
      os << "mem_type_=" << ToString(mem_type_)<<"\t";
      os << "pre_step_type_=" << ToString(pre_step_type_)<<"\t";
      os << "step_type_=" << ToString(step_type_)<<"\t";
      os << "post_step_type_=" << ToString(post_step_type_)<<"\t";
    }

    virtual void PrintU(std::ostream& os){
      //Copy the data to host
      switch( mem_type_){
        case MemType::kMalloc:
        case MemType::kPinned:
          //Copy to host
          CheckCuda(cudaMemcpy(h_cons_,d_cons_,mem_size_,cudaMemcpyDeviceToHost));
          CheckCuda(cudaMemcpy(h_prim_,d_prim_,mem_size_,cudaMemcpyDeviceToHost));
          break;
        case MemType::kUVM:
          //Prefetch to host (unneccessary)
          CheckCuda(cudaMemPrefetchAsync(d_cons_,mem_size_,cudaCpuDeviceId,NULL));
          CheckCuda(cudaMemPrefetchAsync(d_prim_,mem_size_,cudaCpuDeviceId,NULL));
          break;
        default:
          std::stringstream ss;
          ss  << "MemType '"
              << ToString(mem_type_)
              << "' unsupported!";
          throw ss.str();
          break;
      }
      cudaDeviceSynchronize();

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

#endif //CUDA_CONS_TO_PRIM_AH_H_
