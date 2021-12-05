CC = nvcc
C = g++

make: project

project: cpuNN.o gpuNN.o main.o
	$(CC) -c cpuNN.o gpuNN.o main.o

cpuNN.o: neuralNetworkCPU.cpp neuralNetworkCPU.h
	$(C) -c neuralNetworkCPU.cpp

gpuNN.o: neuralNetworkGPU.cu neuralNetworkGPU.h
	${CC} -c neuralNetworkGPU.cu

main.o: main.cu
	${CC} -c main.cu
clean:
	$(RM) project *.o