NVCC = nvcc
CC = g++

make: project

project: cpuNN.o gpuNN.o timing.o main.o
	$(NVCC) -o neuralNetwork cpuNN.o gpuNN.o timing.o main.o

cpuNN.o: neuralNetworkCPU.cpp neuralNetworkCPU.h
	$(CC) -c neuralNetworkCPU.cpp

gpuNN.o: neuralNetworkGPU.cu neuralNetworkGPU.h
	${NVCC} -arch=sm_52 -c neuralNetworkGPU.cu

timing.o: timing.cpp timing.h
	${CC} -c timing.cpp

main.o: main.cu
	${NVCC} -arch=sm_52 -c main.cu
clean:
	rm -f *.o
	rm -f neuralNetwork