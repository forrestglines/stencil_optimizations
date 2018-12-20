#include <string>
#include <iostream>
#include <cstdio>


#define SQR(X) (X)*(X)


////////////////////////////////////////////////////////////
//    MDRange Implmentation
////////////////////////////////////////////////////////////

////////////////////
// Functor
////////////////////

template<typename T,typename Layout,typename IType,Kokkos::Iterate ItOuter, Kokkos::Iterate ItInner>
struct KokkosConsToPrimAH<T,Layout,IType,ItOuter,ItInner>::KokkosMDRangeConsToPrimAHFunctor{

  Kokkos::View<T****, Layout,DevSpace> cons_;
  Kokkos::View<T****, Layout,DevSpace> prim_;

  const double density_floor_, pressure_floor_, gm1_;

  KokkosMDRangeConsToPrimAHFunctor(
      Kokkos::View<T****, Layout,DevSpace> cons,
      Kokkos::View<T****, Layout,DevSpace> prim,
      const double density_floor, const double pressure_floor, const double gm1):
    cons_(cons),prim_(prim),
    density_floor_(density_floor),pressure_floor_(pressure_floor),gm1_(gm1){}

  KOKKOS_INLINE_FUNCTION
  void operator()( const IType k, const IType j, const IType i ) const {

      T& u_d  = cons_(IDN,k,j,i);
      T& u_m1 = cons_(IM1,k,j,i);
      T& u_m2 = cons_(IM2,k,j,i);
      T& u_m3 = cons_(IM3,k,j,i);
      T& u_e  = cons_(IEN,k,j,i);

      T& w_d  = prim_(IDN,k,j,i);
      T& w_vx = prim_(IVX,k,j,i);
      T& w_vy = prim_(IVY,k,j,i);
      T& w_vz = prim_(IVZ,k,j,i);
      T& w_p  = prim_(IPR,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > density_floor_) ?  u_d : density_floor_;
      w_d = u_d;

      T di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1_*(u_e - ke);

      // apply pressure floor, correct total energy
      u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
      w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

  }

};

////////////////////
// Kernel
////////////////////

template<typename T,typename Layout,typename IType,Kokkos::Iterate ItOuter, Kokkos::Iterate ItInner>
void KokkosConsToPrimAH<T,Layout,IType,ItOuter,ItInner>::KokkosMDRangeConsToPrimAH(int dim){

  //Create an MDRangePolicy depending on the test parameters
  Kokkos::MDRangePolicy<Kokkos::Rank<3,ItOuter,ItInner>> policy({0,0,0},{0,0,0});

  if(tiling_[0] == 0){
    //Use the default tiling
    policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,ItOuter,ItInner>>({ks_,js_,is_},{ke_+1,je_+1,ie_+1});
  } else {
    //Use the test provided tiling
    policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,ItOuter,ItInner>>({ks_,js_,is_},{ke_+1,je_+1,ie_+1},tiling_);
  }

  if(func_type_ == FuncType::kLambda){
    //Alias the member variables to work on the GPU
    Kokkos::View<T****, Layout,DevSpace>& cons_ = this->cons_;
    Kokkos::View<T****, Layout,DevSpace>& prim_ = this->prim_;

    const double& density_floor_ = this->density_floor_;
    const double& pressure_floor_ = this->pressure_floor_;
    const double& gm1_ = this->gm1_;

    Kokkos::parallel_for("MDRangeConsToPrimAH id_: "+id_,policy,
      KOKKOS_LAMBDA ( const IType k, const IType j, const IType i){

        T& u_d  = cons_(IDN,k,j,i);
        T& u_m1 = cons_(IM1,k,j,i);
        T& u_m2 = cons_(IM2,k,j,i);
        T& u_m3 = cons_(IM3,k,j,i);
        T& u_e  = cons_(IEN,k,j,i);

        T& w_d  = prim_(IDN,k,j,i);
        T& w_vx = prim_(IVX,k,j,i);
        T& w_vy = prim_(IVY,k,j,i);
        T& w_vz = prim_(IVZ,k,j,i);
        T& w_p  = prim_(IPR,k,j,i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
        w_d = u_d;

        T di = 1.0/u_d;
        w_vx = u_m1*di;
        w_vy = u_m2*di;
        w_vz = u_m3*di;

        T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
        w_p = gm1_*(u_e - ke);

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
        w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

      }
    );
  } else {
    Kokkos::parallel_for("MDRangeConsToPrimAH id_: "+id_,policy,
      KokkosMDRangeConsToPrimAHFunctor(cons_,prim_,density_floor_,pressure_floor_,gm1_)
    );

  }
}

////////////////////////////////////////////////////////////
//    1DRange Implmentation
////////////////////////////////////////////////////////////

////////////////////
// Functor
////////////////////

template<typename T,typename Layout,typename IType,Kokkos::Iterate ItOuter, Kokkos::Iterate ItInner>
struct KokkosConsToPrimAH<T,Layout,IType,ItOuter,ItInner>::Kokkos1DRangeConsToPrimAHFunctor{

  Kokkos::View<T****, Layout,DevSpace> cons_;
  Kokkos::View<T****, Layout,DevSpace> prim_;

  const double density_floor_, pressure_floor_, gm1_;

  const IType NI, NJ, NK, NJNI, is_, js_, ks_;

  Kokkos1DRangeConsToPrimAHFunctor(
      Kokkos::View<T****, Layout,DevSpace> cons,
      Kokkos::View<T****, Layout,DevSpace> prim,
      IType is, IType ie, IType js, IType je, IType ks, IType ke,
      const double density_floor, const double pressure_floor, const double gm1):
    cons_(cons),prim_(prim),
    NI(ie-is+1),NJ(je-js+1),NK(ke-ks+1),NJNI(NI*NJ),is_(is),js_(js),ks_(ks),
    density_floor_(density_floor),pressure_floor_(pressure_floor),gm1_(gm1){}

  KOKKOS_INLINE_FUNCTION
  void operator()(const IType& IDX) const {
			IType k = IDX / NJNI;
			IType j = (IDX - k*NJNI) / NI;
			IType i = IDX - k*NJNI - j*NI;

			k += ks_;
			j += js_;
			i += is_;

      T& u_d  = cons_(IDN,k,j,i);
      T& u_m1 = cons_(IM1,k,j,i);
      T& u_m2 = cons_(IM2,k,j,i);
      T& u_m3 = cons_(IM3,k,j,i);
      T& u_e  = cons_(IEN,k,j,i);

      T& w_d  = prim_(IDN,k,j,i);
      T& w_vx = prim_(IVX,k,j,i);
      T& w_vy = prim_(IVY,k,j,i);
      T& w_vz = prim_(IVZ,k,j,i);
      T& w_p  = prim_(IPR,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > density_floor_) ?  u_d : density_floor_;
      w_d = u_d;

      T di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1_*(u_e - ke);

      // apply pressure floor, correct total energy
      u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
      w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

  }

};

////////////////////
// Kernel
////////////////////

template<typename T,typename Layout,typename IType,Kokkos::Iterate ItOuter, Kokkos::Iterate ItInner>
void KokkosConsToPrimAH<T,Layout,IType,ItOuter,ItInner>::Kokkos1DRangeConsToPrimAH(int dim){

  if(func_type_ == FuncType::kLambda){
    //Alias the member variables to work on the GPU
    Kokkos::View<T****, Layout,DevSpace>& cons_ = this->cons_;
    Kokkos::View<T****, Layout,DevSpace>& prim_ = this->prim_;

    const double& density_floor_ = this->density_floor_;
    const double& pressure_floor_ = this->pressure_floor_;
    const double& gm1_ = this->gm1_;

    const int is_ = this->is_, js_ = this->js_, ks_ = this->ks_;
    const int ie_ = this->ie_, je_ = this->je_, ke_ = this->ke_;

    //Compute some constants for use in the kernel
    const IType NK = ke_ - ks_ + 1;
    const IType NJ = je_ - js_ + 1;
    const IType NI = ie_ - is_ + 1;
    const IType NKNJNI = NK*NJ*NI;
    const IType NJNI = NJ * NI;
    Kokkos::parallel_for("1DRangeNaiveConsToPrimAH id_: "+id_,
      NKNJNI,
      KOKKOS_LAMBDA (const int& IDX) {
        IType k = IDX / NJNI;
        IType j = (IDX - k*NJNI) / NI;
        IType i = IDX - k*NJNI - j*NI;
        k += ks_;
        j += js_;
        i += is_;

        T& u_d  = cons_(IDN,k,j,i);
        T& u_m1 = cons_(IM1,k,j,i);
        T& u_m2 = cons_(IM2,k,j,i);
        T& u_m3 = cons_(IM3,k,j,i);
        T& u_e  = cons_(IEN,k,j,i);

        T& w_d  = prim_(IDN,k,j,i);
        T& w_vx = prim_(IVX,k,j,i);
        T& w_vy = prim_(IVY,k,j,i);
        T& w_vz = prim_(IVZ,k,j,i);
        T& w_p  = prim_(IPR,k,j,i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
        w_d = u_d;

        T di = 1.0/u_d;
        w_vx = u_m1*di;
        w_vy = u_m2*di;
        w_vz = u_m3*di;

        T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
        w_p = gm1_*(u_e - ke);

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
        w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

      }
    );
  }else{
    const IType NK = ke_ - ks_ + 1;
    const IType NJ = je_ - js_ + 1;
    const IType NI = ie_ - is_ + 1;
    const IType NKNJNI = NK*NJ*NI;
    Kokkos::parallel_for("1DRangeConsToPrimAH id_: "+id_,NKNJNI,
        Kokkos1DRangeConsToPrimAHFunctor(cons_,prim_,
          is_,ie_,js_,je_,ks_,ke_,
          density_floor_,pressure_floor_,gm1_)
    );
  }
}

typedef Kokkos::TeamPolicy<>               team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;

////////////////////////////////////////////////////////////
//    TVR Implmentation
////////////////////////////////////////////////////////////

////////////////////
// Functor
////////////////////

template<typename T,typename Layout,typename IType,Kokkos::Iterate ItOuter, Kokkos::Iterate ItInner>
struct KokkosConsToPrimAH<T,Layout,IType,ItOuter,ItInner>::KokkosTVRConsToPrimAHFunctor{

  Kokkos::View<T****, Layout,DevSpace> cons_;
  Kokkos::View<T****, Layout,DevSpace> prim_;

  const double density_floor_, pressure_floor_, gm1_;

  const IType NJ, is_, ie_, js_, ks_;

  KokkosTVRConsToPrimAHFunctor(
      Kokkos::View<T****, Layout,DevSpace> cons,
      Kokkos::View<T****, Layout,DevSpace> prim,
      IType is, IType ie, IType js, IType je, IType ks, IType ke,
      const double density_floor, const double pressure_floor, const double gm1):
    cons_(cons),prim_(prim),
    NJ(je-js+1),is_(is),ie_(ie),js_(js),ks_(ks),
    density_floor_(density_floor),pressure_floor_(pressure_floor),gm1_(gm1){}

  KOKKOS_INLINE_FUNCTION
  void operator()(member_type team_member) const {
      const IType k = team_member.league_rank() / NJ + ks_;
      const IType j = team_member.league_rank() % NJ + js_;
      Kokkos::parallel_for(
				Kokkos::ThreadVectorRange<>(team_member,is_,ie_ + 1),
          [&] (IType i) __attribute__((always_inline)) {
      T& u_d  = cons_(IDN,k,j,i);
      T& u_m1 = cons_(IM1,k,j,i);
      T& u_m2 = cons_(IM2,k,j,i);
      T& u_m3 = cons_(IM3,k,j,i);
      T& u_e  = cons_(IEN,k,j,i);

      T& w_d  = prim_(IDN,k,j,i);
      T& w_vx = prim_(IVX,k,j,i);
      T& w_vy = prim_(IVY,k,j,i);
      T& w_vz = prim_(IVZ,k,j,i);
      T& w_p  = prim_(IPR,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > density_floor_) ?  u_d : density_floor_;
      w_d = u_d;

      T di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1_*(u_e - ke);

      // apply pressure floor, correct total energy
      u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
      w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;
    });
  }

};

////////////////////
// Kernel
////////////////////

template<typename T,typename Layout,typename IType,Kokkos::Iterate ItOuter, Kokkos::Iterate ItInner>
void KokkosConsToPrimAH<T,Layout,IType,ItOuter,ItInner>::KokkosTVRConsToPrimAH(int dim){

  if(func_type_ == FuncType::kLambda){
    //Alias the member variables to work on the GPU
    Kokkos::View<T****, Layout,DevSpace>& cons_ = this->cons_;
    Kokkos::View<T****, Layout,DevSpace>& prim_ = this->prim_;

    const double& density_floor_ = this->density_floor_;
    const double& pressure_floor_ = this->pressure_floor_;
    const double& gm1_ = this->gm1_;

    const int is_ = this->is_, js_ = this->js_, ks_ = this->ks_;
    const int ie_ = this->ie_, je_ = this->je_, ke_ = this->ke_;

    const IType NK = ke_ - ks_ + 1;
    const IType NJ = je_ - js_ + 1;
    const IType NKNJ = NK * NJ;
    Kokkos::parallel_for("TVRConsToPrimAH",
      team_policy (NKNJ, Kokkos::AUTO,vector_length_),
      KOKKOS_LAMBDA (member_type team_member) {
        const IType k = team_member.league_rank() / NJ + ks_;
        const IType j = team_member.league_rank() % NJ + js_;
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange<>(team_member,is_,ie_ + 1),
            [&] (IType i) __attribute__((always_inline)) {
        T& u_d  = cons_(IDN,k,j,i);
        T& u_m1 = cons_(IM1,k,j,i);
        T& u_m2 = cons_(IM2,k,j,i);
        T& u_m3 = cons_(IM3,k,j,i);
        T& u_e  = cons_(IEN,k,j,i);

        T& w_d  = prim_(IDN,k,j,i);
        T& w_vx = prim_(IVX,k,j,i);
        T& w_vy = prim_(IVY,k,j,i);
        T& w_vz = prim_(IVZ,k,j,i);
        T& w_p  = prim_(IPR,k,j,i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
        w_d = u_d;

        T di = 1.0/u_d;
        w_vx = u_m1*di;
        w_vy = u_m2*di;
        w_vz = u_m3*di;

        T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
        w_p = gm1_*(u_e - ke);

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
        w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

            }
        );
      }
    );
  } else {
    const IType NK = ke_ - ks_ + 1;
    const IType NJ = je_ - js_ + 1;
    const IType NKNJ = NK * NJ;
    Kokkos::parallel_for("TVRConsToPrimAH",
      team_policy (NKNJ, Kokkos::AUTO,vector_length_),
      KokkosTVRConsToPrimAHFunctor(cons_,prim_,
        is_,ie_,js_,je_,ks_,ke_,
        density_floor_,pressure_floor_,gm1_)
      );

  }
}

////////////////////////////////////////////////////////////
//    TTR Implmentation
////////////////////////////////////////////////////////////

////////////////////
// Functor
////////////////////

template<typename T,typename Layout,typename IType,Kokkos::Iterate ItOuter, Kokkos::Iterate ItInner>
struct KokkosConsToPrimAH<T,Layout,IType,ItOuter,ItInner>::KokkosTTRConsToPrimAHFunctor{

  Kokkos::View<T****, Layout,DevSpace> cons_;
  Kokkos::View<T****, Layout,DevSpace> prim_;

  const double density_floor_, pressure_floor_, gm1_;

  const IType NJ, is_, ie_, js_, ks_;

  KokkosTTRConsToPrimAHFunctor(
      Kokkos::View<T****, Layout,DevSpace> cons,
      Kokkos::View<T****, Layout,DevSpace> prim,
      IType is, IType ie, IType js, IType je, IType ks, IType ke,
      const double density_floor, const double pressure_floor, const double gm1):
    cons_(cons),prim_(prim),
    NJ(je-js+1),is_(is),ie_(ie),js_(js),ks_(ks),
    density_floor_(density_floor),pressure_floor_(pressure_floor),gm1_(gm1){}

  KOKKOS_INLINE_FUNCTION
  void operator()(member_type team_member) const {
      const IType k = team_member.league_rank() / NJ + ks_;
      const IType j = team_member.league_rank() % NJ + js_;
      Kokkos::parallel_for(
				Kokkos::TeamThreadRange<>(team_member,is_,ie_ + 1),
          [&] (IType i) __attribute__((always_inline)) {
      T& u_d  = cons_(IDN,k,j,i);
      T& u_m1 = cons_(IM1,k,j,i);
      T& u_m2 = cons_(IM2,k,j,i);
      T& u_m3 = cons_(IM3,k,j,i);
      T& u_e  = cons_(IEN,k,j,i);

      T& w_d  = prim_(IDN,k,j,i);
      T& w_vx = prim_(IVX,k,j,i);
      T& w_vy = prim_(IVY,k,j,i);
      T& w_vz = prim_(IVZ,k,j,i);
      T& w_p  = prim_(IPR,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > density_floor_) ?  u_d : density_floor_;
      w_d = u_d;

      T di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1_*(u_e - ke);

      // apply pressure floor, correct total energy
      u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
      w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;
    });
  }

};

////////////////////
// Kernel
////////////////////

template<typename T,typename Layout,typename IType,Kokkos::Iterate ItOuter, Kokkos::Iterate ItInner>
void KokkosConsToPrimAH<T,Layout,IType,ItOuter,ItInner>::KokkosTTRConsToPrimAH(int dim){

  if(func_type_ == FuncType::kLambda){
    //Alias the member variables to work on the GPU
    Kokkos::View<T****, Layout,DevSpace>& cons_ = this->cons_;
    Kokkos::View<T****, Layout,DevSpace>& prim_ = this->prim_;

    const double& density_floor_ = this->density_floor_;
    const double& pressure_floor_ = this->pressure_floor_;
    const double& gm1_ = this->gm1_;

    const int is_ = this->is_, js_ = this->js_, ks_ = this->ks_;
    const int ie_ = this->ie_, je_ = this->je_, ke_ = this->ke_;

    const IType NK = ke_ - ks_ + 1;
    const IType NJ = je_ - js_ + 1;
    const IType NKNJ = NK * NJ;
    Kokkos::parallel_for("TTRConsToPrimAH",
      team_policy (NKNJ, Kokkos::AUTO,vector_length_),
      KOKKOS_LAMBDA (member_type team_member) {
        const IType k = team_member.league_rank() / NJ + ks_;
        const IType j = team_member.league_rank() % NJ + js_;
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange<>(team_member,is_,ie_ + 1),
            [&] (IType i) __attribute__((always_inline)) {
        T& u_d  = cons_(IDN,k,j,i);
        T& u_m1 = cons_(IM1,k,j,i);
        T& u_m2 = cons_(IM2,k,j,i);
        T& u_m3 = cons_(IM3,k,j,i);
        T& u_e  = cons_(IEN,k,j,i);

        T& w_d  = prim_(IDN,k,j,i);
        T& w_vx = prim_(IVX,k,j,i);
        T& w_vy = prim_(IVY,k,j,i);
        T& w_vz = prim_(IVZ,k,j,i);
        T& w_p  = prim_(IPR,k,j,i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
        w_d = u_d;

        T di = 1.0/u_d;
        w_vx = u_m1*di;
        w_vy = u_m2*di;
        w_vz = u_m3*di;

        T ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
        w_p = gm1_*(u_e - ke);

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
        w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

            }
        );
      }
    );
  } else {
    const IType NK = ke_ - ks_ + 1;
    const IType NJ = je_ - js_ + 1;
    const IType NKNJ = NK * NJ;
    Kokkos::parallel_for("TTRConsToPrimAH",
      team_policy (NKNJ, Kokkos::AUTO,vector_length_),
      KokkosTTRConsToPrimAHFunctor(cons_,prim_,
        is_,ie_,js_,je_,ks_,ke_,
        density_floor_,pressure_floor_,gm1_) );

  }
}
