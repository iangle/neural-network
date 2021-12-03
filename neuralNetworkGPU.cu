#include "neuralNetworkGPU.h"

NeuralNetworkGPU::NeuralNetworkGPU(float learningRate)
{
    learning_rate = learningRate;
}

__device__ Sigmoid(float x) {return 1.0f / (1.0f + exp(-x)); }

__global__ void hiddenLayerForward(float* Z, float* A, int Z_x_dim, int Z_y_dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < (Z_x_dim * Z_y_dim))
        A[index] = Sigmoid(Z[index]);
}

__global__ void hiddenLayerBackPropagation(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < (Z_x_dim * Z_y_dim))
        A[index] = dA[index] * Sigmoid(Z[index]) * (1-sigmoid(Z[]index));
}

Matrix& NeuralNetworkGPU::forward(Matrix& Z)
{
    this->Z = Z;

    A.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    hiddenLayerForward<<<num_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);

    return A;
}

Matrix& NeuralNetworkGPU::backPropagation(Matrix& dA)
{
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    hiddenLayerBackPropagation<<<num_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(), dZ.data_device.get(), Z.shape.x, Z.shape.y);

    return dZ;
}

void NeuralNetworkGPU::initializeNN()
{

}