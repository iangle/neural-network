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

__device__ float Sigmoid(float x) {return 1.0f / (1.0f + exp(-x)); }

__global__ void forwardHidden(float* weightHidden, float* inData, float* valuesHidden, float numNeuronsHidden, float numNeuronsIn)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float value = 0;

    for(int i = 0; i < numNeuronsHidden; i++)
    {
        for(int j = 0; j < numNeuronsIn; j++)
            value = value + inData[j] * weightHidden[i + ((j + 1) * numNeuronsHidden)]; //double check the postion in the weightsHidden array
        
        value = value + weightHidden[i];
        valuesHidden[i] = Sigmoid(static_cast<float>(value));
        
    }
}

__global__ void forwardOut(float* weightOut, float* valuesOut, float* valuesHidden, float numNeuronsHidden, float numNeuronsOut)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float value = 0;

    for(int i = 0; i < numNeuronsOut; i++)
    {

        for(int j = 0; j < numNeuronsHidden; j++)
            value = value + valuesHidden[j] * weightOut[i + ((j + 1) * numNeuronsOut)]; //double check
        
        value = value + weightOut[i];
        valuesOut[i] = Sigmoid(static_cast<float>(value));
    }
}
__global__ void backPropagationY_Error(float* yError, float* valuesOut, float* trueOut, float numNeuronsOut)
{

}

__global__ void backPropagationH_Error(float* yError, float* hError, float* valuesHidden, float* weightOut, float numNeuronsOut)
{

}

__global__ void adjustWeights(float* weightOut, float* yError, float* hError, float* weightHidden, float* inData, float learningRate, float numNeuronIn, float numNeuronHidden, float numNeuronOut)
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