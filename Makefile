CC = nvcc

make: project

project: matrix.o cpuNN.o gpuNN.o
	$(CC) -o project matrix.o cpuNN.o gpuNN.o

matrix.o: Matrix.cu Matrix.h
	$(CC) -o Matrix.cu

cpuNN.o: neuralNetworkCPU.cpp neuralNetworkCPU.h
	$(CC) -o neuralNetworkCPU.cpp

gpuNN.o: neuralNetworkGPU.cu neuralNetworkGPU.h
	${CC} -o neuralNetworkGPU.cu

clean:
	$(RM) project *.o