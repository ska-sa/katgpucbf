CXX = g++
CXXFLAGS = -pthread -Wall -g -std=c++17
LDFLAGS = -pthread -g -lspead2 -lboost_system -lpcap

SOURCES = demo.cpp $(wildcard src/*.cpp)
OBJECTS = $(patsubst %.cpp, %.o, $(SOURCES))
HEADERS = $(wildcard src/*.h)

demo: $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

clean:
	rm -f demo $(OBJECTS)
