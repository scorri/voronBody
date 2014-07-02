#
# Makefile for example which creates a display to 
# show which parts of the buffer are accessed by 
# regions
#

CC=nvcc
CFLAGS=-arch=sm_20
LFLAGS=-lglut

all:
	$(CC) $(CFLAGS) bufferAreaUse.cu -o bufferAreaUse $(LFLAGS)

clean:
	rm -rf *o *png bufferAreaUse 
