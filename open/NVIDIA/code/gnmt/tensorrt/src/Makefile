# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.SUFFIXES:
TARGET?=$(shell uname -m)
CUDA_INSTALL_DIR?=/usr/local/cuda
CUDNN_INSTALL_DIR?=$(CUDA_INSTALL_DIR)
CUB_INSTALL_DIR?=/usr/lib/aarch64-linux-gnu
TRT_INCLUDE_DIR?=/usr/include/aarch64-linux-gnu
TRT_LIB_DIR?=/usr/lib/aarch64-linux-gnu
CUDA_LIBDIR=lib64
CUDNN_LIBDIR=lib64
CC = g++
CUCC =$(CUDA_INSTALL_DIR)/bin/nvcc -m64 -arch compute_70 -code sm_72

ifdef VERBOSE
AT=
else
AT=@
endif

ECHO = @echo

SHELL=/bin/sh

ROOT_PATH=../../../..
OUT_PATH=$(ROOT_PATH)/build/bin/GNMT
OUTDIR=$(OUT_PATH)
CODE_DIRECTORIES=core plugin
SIMPLEJSON_LIB=/usr/lib/aarch64-linux-gnu/libSimpleJSON.a

OUTNAME_RELEASE = gnmt
OUTNAME_DEBUG   = gnmt_debug

define concat
$1$2$3$4$5$6$7$8
endef

#$(call make-depend,source-file,object-file,depend-file)
define make-depend
  $(AT)$(CC) -MM -MF $3 -MP -MT $2 $(COMMON_FLAGS) $1
endef

#$(call make-cuda-depend,source-file,object-file,depend-file,flags)
define make-cuda-depend
  $(AT)$(CUCC) -M -MT $2 $4 $(COMMON_CUFLAGS) $1 > $3
endef

#########################

INCPATHS = -I"common" -I"core" -I"$(JSON_SRCDIR)" -I"$(CUDA_INSTALL_DIR)/include" -I"$(CUDNN_INSTALL_DIR)/include" -I"$(CUB_INSTALL_DIR)" -I"$(CUB_INSTALL_DIR)/cub" -I"$(TRT_INCLUDE_DIR)" -I"$(TRT_INCLUDE_DIR)/../parsers/onnxOpenSource"
LIBPATHS = -L"$(CUDA_INSTALL_DIR)" -L"$(CUDA_INSTALL_DIR)/$(CUDA_LIBDIR)" -L"$(CUDNN_INSTALL_DIR)" -L"$(CUDNN_INSTALL_DIR)/$(CUDNN_LIBDIR)" -L"$(TRT_LIB_DIR)"

COMMON_FLAGS += -Wall -std=c++11 $(INCPATHS) -Wno-deprecated-declarations
COMMON_CUFLAGS += --compiler-options -Wall -std=c++11 $(INCPATHS)
COMMON_LD_FLAGS += $(LIBPATHS) -L$(OUTDIR)
COMMON_LIBS = -lcudnn -lcublas -lcudart

OBJDIR    =$(call concat,$(OUTDIR),/chobj)
DOBJDIR   =$(call concat,$(OUTDIR),/dchobj)

LIBS  =-lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser $(COMMON_LIBS)
DLIBS =-lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser $(COMMON_LIBS)
OBJS   =$(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard $(addsuffix /*.cpp, $(CODE_DIRECTORIES))))
DOBJS  =$(patsubst %.cpp, $(DOBJDIR)/%.o, $(wildcard $(addsuffix  /*.cpp, $(CODE_DIRECTORIES))))
CUOBJS =$(patsubst %.cu, $(OBJDIR)/%.o, $(wildcard $(addsuffix  /*.cu, $(CODE_DIRECTORIES))))
CUDOBJS =$(patsubst %.cu, $(DOBJDIR)/%.o, $(wildcard $(addsuffix  /*.cu, $(CODE_DIRECTORIES))))

CFLAGS=$(COMMON_FLAGS) -O3
CFLAGSD=$(COMMON_FLAGS) -g
CUFLAGS=$(COMMON_CUFLAGS) -O3
CUFLAGSD=$(COMMON_CUFLAGS) -g
LFLAGS=$(COMMON_LD_FLAGS) -O3
LFLAGSD=$(COMMON_LD_FLAGS)

all: debug release
release : $(OUTDIR)/$(OUTNAME_RELEASE)

debug   : $(OUTDIR)/$(OUTNAME_DEBUG)

test: test_debug test_release

test_debug:
	$(AT)cd $(OUTDIR) && ./$(OUTNAME_DEBUG)

test_release:
	$(AT)cd $(OUTDIR) && ./$(OUTNAME_RELEASE)

# Link to libgnmtcore.a with --whole-archive to make sure plugins are registered.
$(OUTDIR)/$(OUTNAME_RELEASE) : $(OUTDIR)/libgnmtcore.a
	$(ECHO) Linking: $@
	$(AT)$(CC) $(CFLAGS) -c -o $(OBJDIR)/GNMT.o GNMT.cpp
	$(AT)$(CC) -o $@ $(OBJDIR)/GNMT.o -Wl,--whole-archive $^ -Wl,--no-whole-archive $(SIMPLEJSON_LIB) $(LFLAGS) -Wl,--start-group $(LIBS) -Wl,--end-group

$(OUTDIR)/$(OUTNAME_DEBUG) : $(OUTDIR)/libgnmtcore_debug.a
	$(ECHO) Linking: $@
	$(AT)$(CC) $(CFLAGS) -c -o $(DOBJDIR)/GNMT.o GNMT.cpp
	$(AT)$(CC) -o $@ $(DOBJDIR)/GNMT.o -Wl,--whole-archive $^ -Wl,--no-whole-archive $(SIMPLEJSON_LIB) $(LFLAGSD) -Wl,--start-group $(DLIBS) -Wl,--end-group

$(OUTDIR)/libgnmtcore.a: $(OBJS) $(CUOBJS)
	$(ECHO) Archiving: $@
	$(AT)$(AR) rcs $(OUTDIR)/libgnmtcore.a $^

$(OUTDIR)/libgnmtcore_debug.a: $(DOBJS) $(CUDOBJS)
	$(ECHO) Archiving: $@
	$(AT)$(AR) rcs $(OUTDIR)/libgnmtcore_debug.a $^

$(OBJDIR)/%.o: %.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(AT)$(foreach XDIR,$(CODE_DIRECTORIES), if [ ! -d $(OBJDIR)/$(XDIR) ]; then mkdir -p $(OBJDIR)/$(XDIR); fi;) :
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(DOBJDIR)/%.o: %.cpp
	$(AT)if [ ! -d $(DOBJDIR) ]; then mkdir -p $(DOBJDIR); fi
	$(AT)$(foreach XDIR,$(CODE_DIRECTORIES), if [ ! -d $(OBJDIR)/$(XDIR) ]; then mkdir -p $(DOBJDIR)/$(XDIR); fi;) :
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGSD) -c -o $@ $<

######################################################################### CU
$(OBJDIR)/%.o: %.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(AT)$(foreach XDIR,$(CODE_DIRECTORIES), if [ ! -d $(OBJDIR)/$(XDIR) ]; then mkdir -p $(OBJDIR)/$(XDIR); fi;) :
	$(call make-cuda-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling CUDA release: $<
	$(AT)$(CUCC) $(CUFLAGS) -c -o $@ $<

$(DOBJDIR)/%.o: %.cu
	$(AT)if [ ! -d $(DOBJDIR) ]; then mkdir -p $(DOBJDIR); fi
	$(AT)$(foreach XDIR,$(CODE_DIRECTORIES), if [ ! -d $(DOBJDIR)/$(XDIR) ]; then mkdir -p $(DOBJDIR)/$(XDIR); fi;) :
	$(call make-cuda-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling CUDA debug: $<
	$(AT)$(CUCC) $(CUFLAGSD) -c -o $@ $<

clean:
	$(ECHO) Cleaning...
	$(AT)-rm -rf $(OBJDIR) $(DOBJDIR) $(OUTDIR)/$(OUTNAME_RELEASE) $(OUTDIR)/$(OUTNAME_DEBUG)

ifneq "$(MAKECMDGOALS)" "clean"
  -include $(OBJDIR)/*.d $(DOBJDIR)/*.d

ifeq ($(DO_CUDNN_CHECK), 1)
# To display newlines in the message.
define _cudnn_missing_newline_5020fd0

endef
SHELL=/bin/bash
CUDNN_CHECK = $(shell echo -e '\#include <cudnn.h>\nint main(){ cudnnCreate(nullptr); return 0; }' | $(CC) -xc++ -o /dev/null $(CFLAGS) $(LFLAGS) - $(COMMON_LIBS) 2> /dev/null && echo 'passed_cudnn_exists_check')
ifneq ($(CUDNN_CHECK), passed_cudnn_exists_check)
$(error $(_cudnn_missing_newline_5020fd0)$(_cudnn_missing_newline_5020fd0)This sample requires CUDNN, but it could not be found.$(_cudnn_missing_newline_5020fd0)Please install CUDNN from https://developer.nvidia.com/cudnn or specify CUDNN_INSTALL_DIR when compiling.$(_cudnn_missing_newline_5020fd0)For example, `make CUDNN_INSTALL_DIR=/path/to/CUDNN/` where /path/to/CUDNN/ contains include/ and lib/ subdirectories.$(_cudnn_missing_newline_5020fd0)$(_cudnn_missing_newline_5020fd0))
endif # ifneq ($(CUDNN_CHECK), passed_cudnn_exists_check)
endif # ifeq ($(DO_CUDNN_CHECK), 1)
endif # ifneq "$(MAKECMDGOALS)" "clean"
