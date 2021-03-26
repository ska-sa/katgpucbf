# Target executable
TARGETS := libfunc
all: $(TARGETS)

remake: clean all

libfunc: functionlib.c
	cc -fPIC -shared -o libfunc.so functionlib.c

clean:
	rm -f $(TARGETS)
