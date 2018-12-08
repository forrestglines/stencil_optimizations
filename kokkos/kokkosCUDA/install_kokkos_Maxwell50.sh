#!/bin/bash -ex

KOKKOSDIR=$HOME/sources/kokkos

$KOKKOSDIR/generate_makefile.bash \
--prefix=$(pwd) \
--with-cuda \
--cxxflags="-O3" \
--arch="Maxwell50" \
--compiler=$KOKKOSDIR/bin/nvcc_wrapper \
--with-options=disable_deprecated_code \
--with-cuda-options=enable_lambda

make -j 8 install

# disable kokkos-clean target which would otherwise be the default target when
# included in the kathena build process
sed -i 's/kokkos-clean/#kokkos-clean/' Makefile.kokkos

