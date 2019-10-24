CXX = g++
CXXFLAGS = -pthread -Wall -g -std=c++17 -fvisibility=hidden
LDFLAGS = -pthread -g -lspead2 -lboost_system -lpcap -ldl

SOURCES = demo.cpp $(filter-out $(wildcard src/py_*.cpp), $(wildcard src/*.cpp))
OBJECTS = $(patsubst %.cpp, %.o, $(SOURCES))
HEADERS = $(wildcard src/*.h)

demo: $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

clean:
	rm -f demo $(OBJECTS)
