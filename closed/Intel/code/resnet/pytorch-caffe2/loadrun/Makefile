
CAFFE2_DIR=../../../pytorch
MKLROOT=/opt/intel/compilers_and_libraries_2019/linux/mkl
LOADGEN_DIR=../../../pytorch/third_party/mlperf_inference
BOOST_DIR=/home/mlt/tools/boost_1_63_0-install

INCLUDE = -I$(BOOST_DIR)/include/ -I$(MKLROOT)/include -I$(CAFFE2_DIR)/torch/include -I$(CAFFE2_DIR)/third_party/ideep/include -I$(CAFFE2_DIR)/caffe2/core/nomnigraph/include  -I$(LOADGEN_DIR)/loadgen
LIBOPTS = -L/usr -L$(MKLROOT)/lib -L~/.local/lib -L$(CAFFE2_DIR)/build/lib -L$(LOADGEN_DIR) -L../scripts/backend -L$(BOOST_DIR)/lib/
LDFLAGS := -lpytorch_backend -lcaffe2 -lc10 -lmlperf_loadgen -lprotobuf `pkg-config --libs opencv` -lrt 

CFLAGS = -O3 -fpic -Wall -std=gnu++14 -g
CC = gcc
CXX = g++ -fopenmp
NB_THREADS = 8

.PHONY : all
all : loadrun netrun

loadrun : loadrun.cc inferencer.h multi_instance.h 
	$(CXX) loadrun.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ $(LDFLAGS)

netrun: netrun.cc inferencer.h multi_instance.h 
	$(CXX) netrun.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ $(LDFLAGS)

clean :
	rm -f *.o loadrun netrun
