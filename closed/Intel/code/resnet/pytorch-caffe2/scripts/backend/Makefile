
BOOST_DIR=~/tools/boost_1_63_0-install
CAFFE2_DIR=../../../../pytorch

INCLUDE = -I$(MKLROOT)/include -I$(CAFFE2_DIR)/torch/include -I$(CAFFE2_DIR)/third_party/ideep/include -I$(CAFFE2_DIR)/third_party/euler/include -I$(CAFFE2_DIR)/caffe2/core/nomnigraph/include -I../../loadrun -I$(BOOST_DIR)/include
LIBOPTS = -L$(MKLROOT)/lib -L$(CAFFE2_DIR)/third_party/euler/build/lib -L ~/.local/lib -L$(CAFFE2_DIR)/build/lib
LDFLAGS := -lcaffe2 -lgflags -lglog -lc10  -lprotobuf  `pkg-config --cflags --libs opencv` -lpthread -lrt

CFLAGS = -O3 -fpic -Wall -std=c++11 -g
CC = gcc
CXX = g++ -fopenmp
NB_THREADS = 8

.PHONY : all
all : libpytorch_backend.a

pytorch_backend.o : pytorch_backend.cc 
	$(CXX) pytorch_backend.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ -c $(LDFLAGS)

libpytorch_backend.a : pytorch_backend.o
	ar -crv $@ pytorch_backend.o

clean :
	rm -f *.o pytorch_backend.a

