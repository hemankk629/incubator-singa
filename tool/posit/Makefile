CXX := g++

INCLUDE 	:= ../include
INCLUDE2	:= ../build/include
SRC     	:= ../src
OBJ     	:= ./obj
EXE			:= cnn

SRCS    := $(wildcard *.cc)
OBJS    := $(patsubst %.cc,%.o,$(SRCS))

LDFLAGS := -L/usr/local/lib/
LDLIBS	:= -lglog -lprotobuf

CFLAGS  := -I$(INCLUDE) -I$(INCLUDE2) -std=c++11

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)
    
%.o: %.cc
	$(CXX) $(CFLAGS) -c $< -o $@
