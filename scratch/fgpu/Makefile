CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -O3 -pthread
LDFLAGS = -lspead2 -lboost_program_options -lboost_system -ldl -pthread

all: dsim

dsim: dsim.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)
