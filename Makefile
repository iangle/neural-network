NVCC = nvcc
CC = g++

make: project

project: main.o gpuNN.o timing.o
	$(NVCC) -arch=sm_52 -o neuralNetwork main.o gpuNN.o timing.o -I.

#cpuNN.o: neuralNetworkCPU.cpp
#	$(CC) -c neuralNetworkCPU.cpp

gpuNN.o: neuralNetworkGPU.h neuralNetworkGPU.cu
	${NVCC} -arch=sm_52 -c neuralNetworkGPU.cu

timing.o: timing.h timing.cpp 
	${CC} -c -o timing.o timing.cpp -I.

main.o: main.cu
	${NVCC} -arch=sm_52 -c main.cu
clean:
	rm -f *.o
	rm -f neuralNetwork