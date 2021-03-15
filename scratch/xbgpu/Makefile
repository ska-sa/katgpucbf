CXX = g++
CXXFLAGS = $(shell pkg-config --cflags spead2) -std=c++17 -Wall -Wextra -g -O3 -pthread
LDFLAGS = $(shell pkg-config --static --libs spead2) -lboost_program_options
TARGETS = fsim

all: $(TARGETS)

fsim: fsim.o
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean