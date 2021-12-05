CC = nvcc
C = g++

make: project

project: cpuNN.o gpuNN.o timing.o main.o
	$(CC) -c cpuNN.o gpuNN.o timing.o main.o

cpuNN.o: neuralNetworkCPU.cpp neuralNetworkCPU.h
	$(C) -c neuralNetworkCPU.cpp

gpuNN.o: neuralNetworkGPU.cu neuralNetworkGPU.h
	${CC} -c neuralNetworkGPU.cu

timing.o: timing.cpp timing.h
	${C} -c timing.cpp

main.o: main.cu
	${CC} -c main.cu
clean:
	$(RM) project *.o