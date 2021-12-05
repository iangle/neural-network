CC = nvcc
C = g++

make: project

project: main.o cpuNN.o gpuNN.o timing.o 
	$(CC) -o neuralNetwork cpuNN.o gpuNN.o timing.o main.o

cpuNN.o: neuralNetworkCPU.cpp neuralNetworkCPU.h
	$(C) -c neuralNetworkCPU.cpp

gpuNN.o: neuralNetworkGPU.cu neuralNetworkGPU.h
	${CC} -arch=sm_52 -c neuralNetworkGPU.cu

timing.o: timing.cpp timing.h
	${C} -c timing.cpp

main.o: main.cu
	${CC} -arch=sm_52 -c main.cu
clean:
	rm -f *.o
	rm -f neuralNetwork