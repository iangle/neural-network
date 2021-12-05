NVCC = nvcc
CC = g++

make: project

project: main.o NeuralNetworkGPU.o timing.o
	$(NVCC) -arch=sm_52 -o neuralNetwork main.o NeuralNetworkGPU.o timing.o -I.

#cpuNN.o: neuralNetworkCPU.cpp
#	$(CC) -c neuralNetworkCPU.cpp

NeuralNetworkGPU.o: NeuralNetworkGPU.h NeuralNetworkGPU.cu
	${NVCC} -arch=sm_52 -c NeuralNetworkGPU.cu

timing.o: timing.h timing.cpp 
	${CC} -c -o timing.o timing.cpp -I.

main.o: main.cu
	${NVCC} -arch=sm_52 -c main.cu
clean:
	rm -f *.o
	rm -f neuralNetwork