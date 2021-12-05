CC = nvcc
C = g++

make: project

project: cpuNN.o gpuNN.o main.cu
	$(CC) -o project cpuNN.o gpuNN.o main.o

cpuNN.o: neuralNetworkCPU.cpp neuralNetworkCPU.h
	$(C) -c neuralNetworkCPU.cpp

gpuNN.o: neuralNetworkGPU.cu neuralNetworkGPU.h
	${CC} -o neuralNetworkGPU.cu

main.o: main.cu
	${CC} -o main.cu
clean:
	$(RM) project *.o