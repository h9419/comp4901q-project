CC = /usr/bin/gcc
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -Xcompiler -g -w -I..  
MPI = -I /usr/local/software/openmpi/include/ 
LIB = -L /usr/local/software/openmpi/lib -lmpi  

BINS:= wfc

wfc: wfc.cu
	${NVCC} ${NVCCFLAGS} ${MPI} ${LIB} --std=c++11 wfc.cu -o wfc


test: $(BINS)
	@mpiexec -n 4 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./wfc 1024 test
save: $(BINS)
	@mpiexec -n 4 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./wfc 64 save

clean:
	-rm multiple


