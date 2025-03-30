CC ?= gcc
#CFLAGS = -O0 -g
CFLAGS = -O3 -fopenmp
INCLUDE = -I$(CONDA_PREFIX)/include
LDFLAGS = -fopenmp
#LDFLAGS =
LIBRARIES = -L$(CONDA_PREFIX)/lib -lFLAC

OBJ = test_low_level.o utils.o compress.o decompress.o verify.o


all : test_low_level

test_low_level : $(OBJ)
	$(CC) -o $@ $(OBJ) $(LDFLAGS) $(LIBRARIES)

%.o : %.c flacarray.h
	$(CC) $(CFLAGS) $(INCLUDE) -I. -c $<

clean :
	@rm -f test_low_level *.o

