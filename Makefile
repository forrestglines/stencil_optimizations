
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

#Kokkos Compiler flags

# Preliminary definitions

COMMON_SRC := $(wildcard common/*.cpp)
CPU_SRC    := $(wildcard cpu/*.cpp)
MPI_SRC    := $(wildcard mpi/*.cpp)
CUDA_SRC   := $(wildcard cuda/*.cu)
KOKKOS_SRC := $(wildcard kokkos/*.cpp)

VPATH := common/:cpu/:cuda/:kokkos/
vpath %h common/ cpu/ mpi/ cuda/ kokkos/
vpath %hpp common/ cpu/ mpi/ cuda/ kokkos/

OBJ_DIR := obj/
COMMON_OBJ_DIR := obj/common/
CPU_OBJ_DIR    := obj/cpu/
MPI_OBJ_DIR    := obj/mpi/
CUDA_OBJ_DIR   := obj/cuda/

KOKKOS_CPU_OBJ_DIR := obj/kokkos/cpu/
KOKKOS_CUDA_OBJ_DIR := obj/kokkos/cuda/
KOKKOS_UVM_OBJ_DIR := obj/kokkos/uvm/

OBJ_DIRS = $(COMMON_OBJ_DIR) $(CPU_OBJ_DIR) $(MPI_OBJ_DIR) $(CUDA_OBJ_DIR) \
		   $(KOKKOS_CPU_OBJ_DIR) $(KOKKOS_CUDA_OBJ_DIR) $(KOKKOS_UVM_OBJ_DIR)

CPU_OBJ    := $(addprefix $(CPU_OBJ_DIR),$(notdir $(CPU_SRC:.cpp=.o))) \
			  $(addprefix $(CPU_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))
MPI_OBJ    := $(addprefix $(MPI_OBJ_DIR),$(notdir $(MPI_SRC:.cpp=.o))) \
			  $(addprefix $(MPI_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))
CUDA_OBJ   := $(addprefix $(CUDA_OBJ_DIR),$(notdir $(CUDA_SRC:.cu=.o))) \
			  $(addprefix $(CUDA_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))

KOKKOS_CPU_OBJ := $(addprefix $(KOKKOS_CPU_OBJ_DIR),$(notdir $(KOKKOS_SRC:.cpp=.o))) \
			      $(addprefix $(KOKKOS_CPU_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))

KOKKOS_CUDA_OBJ := $(addprefix $(KOKKOS_CUDA_OBJ_DIR),$(notdir $(KOKKOS_SRC:.cpp=.o))) \
			       $(addprefix $(KOKKOS_CUDA_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))

KOKKOS_UVM_OBJ := $(addprefix $(KOKKOS_UVM_OBJ_DIR),$(notdir $(KOKKOS_SRC:.cpp=.o))) \
			      $(addprefix $(KOKKOS_UVM_OBJ_DIR),$(notdir $(COMMON_SRC:.cpp=.o)))

OBJS := $(COMMON_OBJ) $(CPU_OBJ) $(MPI_OBJ) $(CUDA_OBJ) $(KOKKOS_OBJ)

BIN_DIR := bin/
CPU_BIN := $(BIN_DIR)testCPU
MPI_BIN := $(BIN_DIR)testMPI
CUDA_BIN := $(BIN_DIR)testCUDA
KOKKOS_CPU_BIN := $(BIN_DIR)testKokkosCPU
KOKKOS_CUDA_BIN := $(BIN_DIR)testKokkosCUDA
KOKKOS_UVM_BIN := $(BIN_DIR)testKokkosUVM

BINS := $(CPU_BIN) $(CUDA_BIN)

# Generally useful targets
.PHONY : all clean cpu mpi cuda kokkosCPU kokkosCUDA kokkosUVM 
.PHONY : debug-CPU debug-mpi debug-cuda debug-kokkosCPU debug-kokkosCUDA debug-kokkosUVM

all : dirs $(BINS) 

ifneq (,$(findstring kokkosCPU,$(MAKECMDGOALS)))
include kokkos/kokkosCPU/Makefile.kokkos
endif

ifneq (,$(findstring kokkosCUDA,$(MAKECMDGOALS)))
include kokkos/kokkosCUDA/Makefile.kokkos
endif

ifneq (,$(findstring kokkosUVM,$(MAKECMDGOALS)))
include kokkos/kokkosUVM/Makefile.kokkos
endif

cpu : dirs $(CPU_BIN)
mpi : dirs $(MPI_BIN)
cuda : dirs $(CUDA_BIN)

kokkosCPU : KOKKOS_CXX := g++
kokkosCPU : dirs $(KOKKOS_CPU_BIN)

kokkosCUDA : KOKKOS_CXX := $(HOME)/sources/kokkos/bin/nvcc_wrapper
kokkosCUDA : dirs $(KOKKOS_CUDA_BIN)

kokkosUVM : KOKKOS_CXX = $(HOME)/sources/kokkos/bin/nvcc_wrapper
kokkosUVM : dirs $(KOKKOS_UVM_BIN)

debug-cpu : CXXFLAGS := $(CXXFLAGS) -g -DDEBUG
debug-cpu : cpu

debug-cuda : NVCCFLAGS := $(NVCCFLAGS) -g -DDEBUG
debug-cuda : cuda

debug-kokkosCPU : KOKKOS_CXXFLAGS := $(KOKKOS_CXXFLAGS) -g -DDEBUG
debug-kokkosCPU : kokkosCPU

debug-kokkosCUDA : KOKKOS_CXXFLAGS := $(KOKKOS_CXXFLAGS) -g -DDEBUG
debug-kokkosCUDA : kokkosCUDA

debug-kokkosUVM : KOKKOS_CXXFLAGS := $(KOKKOS_CXXFLAGS) -g -DDEBUG
debug-kokkosUVM : kokkosUVM

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
	nvcc $(NVCCFLAGS) -o $@  $(CUDA_OBJ) $(LDFLAGS) $(LDLIBS)

$(KOKKOS_CPU_BIN) : $(KOKKOS_CPU_OBJ)
	$(KOKKOS_CXX) $(KOKKOS_FLAGS) -o $@  $(KOKKOS_CPU_OBJ) $(KOKKOS_LDFLAGS) $(LDLIBS) $(KOKKOS_LIBS)

$(KOKKOS_CUDA_BIN) : $(KOKKOS_CUDA_OBJ)
	$(KOKKOS_CXX) $(KOKKOS_FLAGS) -o $@  $(KOKKOS_CUDA_OBJ) $(KOKKOS_LDFLAGS) $(LDLIBS) $(KOKKOS_LIBS)

$(KOKKOS_UVM_BIN) : $(KOKKOS_UVM_OBJ)
	$(KOKKOS_CXX) $(KOKKOS_FLAGS) -o $@  $(KOKKOS_UVM_OBJ) $(KOKKOS_LDFLAGS) $(LDLIBS) $(KOKKOS_LIBS)

# Create objects from source files
$(CPU_OBJ_DIR)%.o : %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS)  -c $< -o $@

$(CUDA_OBJ_DIR)%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(KOKKOS_CPU_OBJ_DIR)%.o : %.cpp
	$(KOKKOS_CXX) $(KOKKOS_CXXFLAGS) -c $< -o $@

$(KOKKOS_CUDA_OBJ_DIR)%.o : %.cpp
	$(KOKKOS_CXX) $(KOKKOS_CXXFLAGS) -c $< -o $@

$(KOKKOS_UVM_OBJ_DIR)%.o : %.cpp
	$(KOKKOS_CXX) $(KOKKOS_CXXFLAGS) -c $< -o $@

# Cleanup
clean :
	rm -rf $(OBJ_DIR)*
	rm -rf $(BIN_DIR)*




