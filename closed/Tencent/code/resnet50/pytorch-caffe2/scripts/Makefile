BOOST_DIR=/home/chenaili/download/boost_1_70_0/
CAFFE2_DIR=../..

INCLUDE = -I$(MKLROOT)/include -I$(CAFFE2_DIR)/torch/include -I$(CAFFE2_DIR)/third_party/ideep/include -I$(CAFFE2_DIR)/third_party/ideep/euler/include -I$(CAFFE2_DIR)/caffe2/core/nomnigraph/include -I$(BOOST_DIR)/include
LIBOPTS = -L$(MKLROOT)/lib -L$(CAFFE2_DIR)/third_party/ideep/euler/build/lib -L ~/.local/lib -L$(CAFFE2_DIR)/build/lib -L$(BOOST_DIR)/lib 
#LDFLAGS := -lcaffe2 -lc10  -lprotobuf -lyaml-cpp `pkg-config --cflags --libs opencv` -lpthread -lrt
LDFLAGS := -lcaffe2 -lc10  -lprotobuf  `pkg-config --cflags --libs opencv` -lpthread -lrt

CFLAGS = -O3 -fpic -Wall -std=c++11 -g
CC = gcc
CXX = g++ -fopenmp
NB_THREADS = 8

.PHONY : all
all : inferencer

inferencer :
	$(CXX) inferencer.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ $(LDFLAGS)

clean :
	rm -f *.o inferencer

