NVCC = nvcc
INCLUDE_DIR = ./include
SRC_DIR = ./src
CFLAGS = -O3 -I$(INCLUDE_DIR) -I$(SRC_DIR)

all: dft

dft: src/dft.cu
	$(NVCC) $(CFLAGS) src/dft.cu src/converter/png.c src/converter/jpeg.c src/converter/bmp.c -o dft

clean:
	rm -f dft input.pgm graph.pgm output.pgm