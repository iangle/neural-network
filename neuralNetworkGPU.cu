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

__global__ void forward(float* Z, float* A, int Z_x_dim, int Z_y_dim)
{

}

__global__ void backPropagation(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim)
{

}

void NeuralNetworkGPU::forward()
{

}

void NeuralNetworkGPU::backPropagation()
{

}

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