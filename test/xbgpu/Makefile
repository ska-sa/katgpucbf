# Target executable
TARGETS := lib_verification_functions
all: $(TARGETS)

remake: clean all

# Example here: https://www.geeksforgeeks.org/how-to-call-a-c-function-in-python/
# -fPIC is a singular flag, yet to find more details on it
# -shared because it is a shared library
lib_verification_functions: verification_functions_lib.c
	cc -fPIC -shared -o lib_verification_functions.so verification_functions_lib.c

clean:
	rm -f $(TARGETS)
