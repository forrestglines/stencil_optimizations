#ifndef MATRIX_H_
#define MATRIX_H_
template <class T>
class Matrix{
  public:
    const unsigned int n1_,n2_,n3_,n4_;


    Matrix(int n3, int n2, int n1):
      n1_(n1),n2_(n2),n3_(n3),n4_(1), data_(NULL),is_my_mem_(false){}
    Matrix(int n4, int n3, int n2, int n1):
      n1_(n1),n2_(n2),n3_(n3),n4_(n4), data_(NULL),is_my_mem_(false){} 
    ~Matrix(){
      Free();
    }

    T &operator() (const int k, const int j, const int i) {
      return data_[i + n1_*j + n1_*n2_*k];
    }

    T &operator() (const int l, const int k, const int j, const int i) {
      return data_[i + n1_*j + n1_*n2_*k + n1_*n2_*n3_*l];
    }

    void Malloc(){
      Free();
      data_ = new T[n1_*n2_*n3_*n4_];
      is_my_mem_ = true;
    }
    void Free(){
      if(is_my_mem_ && data_ != NULL)
        delete data_;
    }

    void UseMem(T* mem){
      //Instead of mallocing, take a memory block
      Free();
      data_ = mem;
      is_my_mem_ = false;
    }

  private:
    bool is_my_mem_;
    T* data_;

};
#endif //MATRIX_H_
