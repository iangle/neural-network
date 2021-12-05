NVCC = nvcc
CC = g++

make: project

project: main.o gpuNN.o timing.o
	$(NVCC) -arch=sm_52 -o neuralNetwork main.o gpuNN.o timing.o

#cpuNN.o: neuralNetworkCPU.cpp
#	$(CC) -c neuralNetworkCPU.cpp

gpuNN.o: neuralNetworkGPU.cu neuralNetworkGPU.h
	${NVCC} -arch=sm_52 -c neuralNetworkGPU.cu

timing.o: timing.cpp timing.h
	${CC} -c timing.cpp

main.o: main.cu
	${NVCC} -arch=sm_52 -c main.cu
clean:
	rm -f *.o
	rm -f neuralNetwork