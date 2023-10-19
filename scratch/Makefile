CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -pthread
TARGETS = memcpy_loop

all: $(TARGETS)

memcpy_loop: memcpy_loop.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
