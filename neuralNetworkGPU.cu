#include "neuralNetworkGPU.h"

NeuralNetworkGPU::NeuralNetworkGPU(int numNeuronInput, int numNeuronHidden, int numNeuronOutput, float learningRate)
{
    learning_rate = learningRate;

    _numNeuronInput = numNeuronInput;
    _numNeuronHidden = numNeuronHidden;
    _numNeuronOut = numNeuronOutput;

    allocateMemoryCPU();

    initializeNN();

}

__device__ Sigmoid(float x) {return 1.0f / (1.0f + exp(-x)); }

/*__global__ void hiddenLayerForward(float* Z, float* A, int Z_x_dim, int Z_y_dim)
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
}*/

void NeuralNetworkGPU::initializeNN()
{
    //initialize our random seed
    srand ((unsigned)time(NULL));

    //initialize the hidden layer with random numbers between (-.5,.5)
    for(int i = 0; i < _numNeuronHidden; i++)
    {
        weightsHidden[i] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;

        for(int j = 1; j < _numNeuronInput + 1; j++)
            weightsHidden[i + (j * _numNeuronHidden)] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;
        
    }

    //initialize the output layer with random numbers between (-.5,.5)
    for(int i = 0; i < _numNeuronOut; i++)
    {
        weightsOut[i] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;

        for(int j = 1; j < _numNeuronHidden + 1; j++)
            weightsOut[i + (j * _numNeuronOut)] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;

    }
}

void NeuralNetworkGPU::allocateMemoryCPU()
{
    //these two are 2D arrays that are flattened into a 1D array
    weightsHidden = (float*) malloc(_numNeuronHidden * (_numNeuronInput + 1) * sizeof(float));
    weightsOut = (float*) malloc(_numNeuronOut * (_numNeuronHidden + 1) * sizeof(float));

    //these are 1D arrays
    valuesHidden = (float*) malloc(_numNeuronHidden * sizeof(float));
    valuesOut = (float*) malloc(_numNeuronOut * sizeof(float));
}