#/******************************************************************************
# * Copyright (c) 2011, Duane Merrill.  All rights reserved.
# * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
# * 
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *	 * Redistributions of source code must retain the above copyright
# *	   notice, this list of conditions and the following disclaimer.
# *	 * Redistributions in binary form must reproduce the above copyright
# *	   notice, this list of conditions and the following disclaimer in the
# *	   documentation and/or other materials provided with the distribution.
# *	 * Neither the name of the NVIDIA CORPORATION nor the
# *	   names of its contributors may be used to endorse or promote products
# *	   derived from this software without specific prior written permission.
# * 
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
#******************************************************************************/

#-------------------------------------------------------------------------------
#
# Makefile usage
# 
# CPU:
# make cpu_spmv [mkl=<0|1>]
#
# GPU:
# make gpu_spmv [sm=<XXX,...>] [verbose=<0|1>] 
#
#-------------------------------------------------------------------------------
 
include ./common.mk 

#-------------------------------------------------------------------------------
# Commandline Options
#-------------------------------------------------------------------------------

# [mkl=<0|1>] compile against Intel MKL
ifneq ($(mkl), 0)
	DEFINES 	+= -DCUB_MKL

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
	LIBS 		+=	mkl_intel_lp64.lib mkl_intel_thread.lib  mkl_core.lib libiomp5md.lib
	NVCCFLAGS 	+= -Xcompiler /openmp
else
	LIBS		+= -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
	NVCCFLAGS 	+= -Xcompiler -fopenmp
	
endif	

endif


#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

# OMP compiler
OMPCC=icpc
OMPCC_FLAGS=-openmp -O3 -lrt -fno-alias -xHost -lnuma -O3 -mkl

# Includes
INC += -I$(CUB_DIR) -I$(CUB_DIR)test 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

exp_rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

EXP_DEPS = 	$(call rwildcard, ./,*.cuh) \
			$(call rwildcard, ./,*.h) \
            Makefile

DEPS =				$(CUB_DEPS) \
					$(EXP_DEPS) \


clean :
	rm -f gpu_spmv cpu_spmv

		
#-------------------------------------------------------------------------------
# make gpu_spmv
#-------------------------------------------------------------------------------

gpu_spmv : gpu_spmv.cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o gpu_spmv gpu_spmv.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -lcusparse -O3

	
#-------------------------------------------------------------------------------
# make cpu_spmv
#-------------------------------------------------------------------------------

cpu_spmv : cpu_spmv.cpp $(DEPS)
	$(OMPCC) $(DEFINES) -o cpu_spmv cpu_spmv.cpp $(OMPCC_FLAGS)

