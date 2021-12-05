CC = nvcc
C = g++

make: project

project: cpuNN.o gpuNN.o main.cu
	$(CC) -o project cpuNN.o gpuNN.o main.o

cpuNN.o: neuralNetworkCPU.cpp neuralNetworkCPU.h
	$(CC) -o neuralNetworkCPU.cpp neuralNetworkCPU.h

gpuNN.o: neuralNetworkGPU.cu neuralNetworkGPU.h
	${CC} -o neuralNetworkGPU.cu neuralNetworkGPU.h

main.o: main.cu
	${CC} -o main.cu
clean:
	$(RM) project *.o