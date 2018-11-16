
#include Kokkos build vars
#include kokkos/kokkos/Makefile.kokkos

#Compiler flags
CXX := g++
CPPFLAGS := 
CXXFLAGS := -O3 -std=c++11
LDFLAGS := 
LDLIBS := 

#CUDA Compiler flags
NVCC := nvcc
NVCCFLAGS := 

# Preliminary definitions

COMMON_SRC := $(wildcard common/*.cpp)
CPU_SRC    := $(wildcard cpu/*.cpp)
MPI_SRC    := $(wildcard mpi/*.cpp)
CUDA_SRC   := $(wildcard cuda/*.cu)
KOKKOS_SRC := $(wildcard kokkos/*.cu)

VPATH := common/:cpu/:cuda/
vpath %h common/ cpu/ mpi/ cuda/ kokkos/
vpath %hpp common/ cpu/ mpi/ cuda/ kokkos/

OBJ_DIR := obj/
COMMON_OBJ_DIR := obj/common/
CPU_OBJ_DIR    := obj/cpu/
MPI_OBJ_DIR    := obj/mpi/
CUDA_OBJ_DIR   := obj/cuda/
KOKKOS_OBJ_DIR := obj/kokkos/
OBJ_DIRS = $(COMMON_OBJ_DIR) $(CPU_OBJ_DIR) $(MPI_OBJ_DIR) $(CUDA_OBJ_DIR) $(KOKKOS_OBJ_DIR)

CPU_OBJ    := $(addprefix $(CPU_OBJ_DIR),$(notdir $(CPU_SRC:.cpp=.o))) \
			  $(addprefix $(CPU_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))
MPI_OBJ    := $(addprefix $(MPI_OBJ_DIR),$(notdir $(MPI_SRC:.cpp=.o))) \
			  $(addprefix $(MPI_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))
CUDA_OBJ   := $(addprefix $(CUDA_OBJ_DIR),$(notdir $(CUDA_SRC:.cu=.o))) \
			  $(addprefix $(CUDA_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))
KOKKOS_OBJ := $(addprefix $(KOKKOS_OBJ_DIR),$(notdir $(KOKKOS_SRC:.cpp=.o))) \
			  $(addprefix $(KOKKOS_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))

OBJS := $(COMMON_OBJ) $(CPU_OBJ) $(MPI_OBJ) $(CUDA_OBJ) $(KOKKOS_OBJ)

BIN_DIR := bin/
CPU_BIN := $(BIN_DIR)testCPU
MPI_BIN := $(BIN_DIR)testMPI
CUDA_BIN := $(BIN_DIR)testCUDA
KOKKOS_BIN := $(BIN_DIR)testKokkos

BINS := $(CPU_BIN) $(CUDA_BIN)

# Generally useful targets
.PHONY : all clean cpu mpi cuda kokkos

all : dirs $(BINS) 
cpu : dirs $(CPU_BIN)
mpi : dirs $(MPI_BIN)
cuda : dirs $(CUDA_BIN)
kokkos : dirs $(KOKKOS_BIN)

objs : dirs $(OBJS)

dirs : $(BIN_DIR) $(OBJ_DIRS)


$(OBJ_DIRS):
	mkdir -p $(BIN_DIR) $(OBJ_DIRS)

# Link objects into executable
$(CPU_BIN) : $(CPU_OBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@  $(CPU_OBJ) $(COMMON_OBJ) $(LDLIBS)

$(MPI_BIN) : $(MPI_OBJ)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -o $@  $(MPI_OBJ) $(COMMON_OBJ) $(LDLIBS)

$(CUDA_BIN) : $(CUDA_OBJ)
	nvcc $(CPPFLAGS) $(CXXFLAGS) -o $@  $(CUDA_OBJ) $(LDFLAGS) $(LDLIBS)

include kokkos/kokkos/Makefile.kokkos
$(KOKKOS_BIN) : $(KOKKOS_OBJ) $(COMMON_OBJ)
	$(HOME)/sources/kokkos/bin/nvcc_wrapper $(CPPFLAGS) $(CXXFLAGS) -o $@  $(KOKKOS_OBJ) $(COMMON_OBJ) $(KOKKOS_LDFLAGS) $(LDLIBS) $(KOKKOS_LIBS)

# Create objects from source files
$(CPU_OBJ_DIR)%.o : %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS)  -c $< -o $@

$(CUDA_OBJ_DIR)%.o : %.cu
	$(NVCC) $(KOKKOS_CXXFLAGS) -c $< -o $@


# Cleanup
clean :
	rm -rf $(OBJ_DIR)*
	rm -rf $(BIN_DIR)*




