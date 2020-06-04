CXX := g++

INCLUDE 	:= ../include
INCLUDE2	:= ../build/include
EXE			:= cnn

SRCS    	:= $(wildcard *.cc)
OBJS    	:= $(patsubst %.cc,%.o,$(SRCS))
DATA_OBJS 	:= images.o labels.o mysnap.o
OBJS		:= $(OBJS) $(DATA_OBJS)

LDFLAGS := -L/usr/local/lib/
LDLIBS	:= -lglog -lprotobuf

CFLAGS  := -I$(INCLUDE) -I$(INCLUDE2) -std=c++11

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)
    
%.o: %.cc
	$(CXX) $(CFLAGS) -c $< -o $@
