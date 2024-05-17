# compiler and linker
NVCC = nvcc
LD = $(NVCC)

# flags
OPT = -O2
OMP = -fopenmp
CPPSTD = -std=c++20
NVCCFLAGS = $(OPT) $(CPPSTD) $(GENCODE_FLAGS) -Xcompiler $(OMP)
LDFLAGS = $(GENCODE_FLAGS) -Xcompiler $(OMP)
LDLIBS = -lpng ./cryptanalysislib/deps/m4ri/.libs/libm4ri.a

# include paths
NVCCINC = -I./cryptanalysislib/src -I./cryptanalysislib/deps/m4ri

# optimize for target compute capability
SMS := $(shell nvidia-smi --query-gpu=compute_cap --format=csv)
SMS := $(lastword $(SMS))
SMS := $(subst .,,$(SMS))
GENCODE_FLAGS = -gencode arch=compute_$(SMS),code=sm_$(SMS)

.SUFFIXES:
.SUFFIXES: .cu.o .cu

.PHONY: all clean

all: bjmm.out

cryptanalysislib: install_cryptanalysislib.sh cryptanalysislib_patch.diff
	rm -rf $@ && bash $<

bjmm_gpu.cu.o: bjmm_gpu.cu bjmm_gpu.cuh bjmm_constant.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

bjmm_cpu.cu.o: bjmm_cpu.cu bjmm_constant.cuh cryptanalysislib
	$(NVCC) $(NVCCFLAGS) $(NVCCINC) -o $@ -c $<

bjmm.out: bjmm_gpu.cu.o bjmm_cpu.cu.o
	$(LD) -o $@ $(LDFLAGS) $^ $(LDLIBS)

clean:
	rm -f *.o *.out
