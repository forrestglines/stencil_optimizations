#ifndef MATRIX_H_
#define MATRIX_H_
template <class T>
class Matrix{
  public:
    const unsigned int n1_,n2_,n3_,n4_;


    Matrix(int n3, int n2, int n1):
      n1_(n1),n2_(n2),n3_(n3),n4_(1), data_(NULL){} 
    Matrix(int n4, int n3, int n2, int n1):
      n1_(n1),n2_(n2),n3_(n3),n4_(n4), data_(NULL) {} 
    ~Matrix(){
      if(data_ == NULL)
        Free();
    }

    T &operator() (const int k, const int j, const int i) {
      return data_[i + n1_*j + n1_*n2_*k];
    }

    T &operator() (const int l, const int k, const int j, const int i) {
      return data_[i + n1_*j + n1_*n2_*k + n1_*n2_*n3_*l];
    }

    void Malloc(){
      data_ = new T[n1_*n2_*n3_*n4_];
    }
    void Free(){
      delete data_;
      data_ = NULL;
    }

  private:
    T* data_;

};
#endif //MATRIX_H_
