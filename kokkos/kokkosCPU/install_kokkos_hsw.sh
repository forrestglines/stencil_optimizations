#!/bin/bash -ex

KOKKOSDIR=$HOME/sources/kokkos

$KOKKOSDIR/generate_makefile.bash \
--prefix=$(pwd) \
--with-openmp \
--cxxflags="-O3" \
--arch="HSW" \
--compiler=g++ \
--with-options=disable_deprecated_code

make -j 8 install

# disable kokkos-clean target which would otherwise be the default target when
# included in the kathena build process
sed -i 's/kokkos-clean/#kokkos-clean/' Makefile.kokkos

