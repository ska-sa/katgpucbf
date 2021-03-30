# Target executable
TARGETS := libfunc
all: $(TARGETS)

remake: clean all

libfunc: functionlib.c
	# Example here: https://www.geeksforgeeks.org/how-to-call-a-c-function-in-python/
	# -fPIC is a singular flag, yet to find more details on it
	# -shared because it is a shared library
	cc -fPIC -shared -o libfunc.so functionlib.c

clean:
	rm -f $(TARGETS)
