CXX := g++

INCLUDE 	:= .
EXE		:= cnn

SRCS    	:= $(wildcard *.cc)
OBJS    	:= $(patsubst %.cc,%.o,$(SRCS))
DATA_OBJS 	:= images.o labels.o mysnap.o
OBJS		:= $(DATA_OBJS) $(OBJS)

CFLAGS  := -I$(INCLUDE) -std=c++11

all: $(EXE)

$(EXE): $(OBJS) libprotobuf.a 
	$(CXX) $(CFLAGS) $^ -o $@
    
%.o: %.cc
	$(CXX) $(CFLAGS) -c $< -o $@
