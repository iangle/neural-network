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

__global__ void forwardHidden(float* weightHidden, float* valuesHidden, float* weightOut, float* valuesOut, 
float* yError, float* hError, float* trueOut, float* results, float numNeuronsHidden, float numNeuronsIn, float numNeuronsOut, float learningRate, int numInputValuesX, int numInputValuesY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * numInputValuesX + x;


    float value = 0;

    // compute valuesHidden
    for(int i = 0; i < numNeuronsHidden; i++)
    {
        value = value + x * weightHidden[i + numNeuronsHidden + idx * numNeuronsHidden];
        value = value + y * weightHidden[i + (2 * numNeuronsHidden) + idx * numNeuronsHidden]; //double check
        value = value + weightHidden[i + idx * numNeuronsHidden];
        valuesHidden[i + idx * numNeuronsHidden] = Sigmoid(static_cast<float>(value));
        
    }
    
    value = 0;

    // compute valuesOut
    for(int i = 0; i < numNeuronsOut; i++)
    {

        for(int j = 0; j < numNeuronsHidden; j++)
            value = value + valuesHidden[j + idx * numNeuronsHidden] * weightOut[i + ((j + 1) * numNeuronsOut) + idx * numNeuronsHidden]; //double check
        
        value = value + weightOut[i + idx * numNeuronsOut];
        valuesOut[i + idx * numNeuronsOut] = Sigmoid(static_cast<float>(value));
    }

    // backwards prop

    //compute yError
    for(int i = 0; i < numNeuronsOut; i++)
    {
        yError[i + idx * numNeuronsOut] = valuesOut[i + idx * numNeuronsOut] * (1 - valuesOut[i + idx * numNeuronsOut]) * (valuesOut[i + idx * numNeuronsOut] - trueOut[i + idx * numNeuronsOut]);
    }

    //compute hError
    for (int i = 0; i < numNeuronsHidden; i++)
    {
        temp = 0;
		for(j = 0; j < numNeuronsOut; j++)
			temp = temp + weightOut[j + (numNeuronsHidden + 1) + idx * numNeuronsOut] * yError[j + idx * numNeuronsOut];
        hError[i + idx * numNeuronsHidden] = temp * valuesHidden[i + idx * numNeuronsHidden] * (1 - valuesHidden[i + idx * numNeuronsHidden]);
    }

    for(int i = 0; i < numNeuronsOut; i++)
        weightOut[i + idx * numNeuronsOut] = weightOut[i + idx * numNeuronsOut] - (learningRate * yError[i + idx * numNeuronsOut]);
    
    for(int i = 0; i < numNeuronsOut; i++)
    {
        for(int j = 0; j < numNeuronsHidden; j++)
        {
            weightOut[i + (j + 1) * numNeuronsHidden] = weightOut[i + (j + 1) * numNeuronsHidden] - (learningRate * yError[i + idx * numNeuronsOut] * valuesHidden[j + idx * numNeuronsHidden]);
        }
    }

    for (int i = 0; i < numNeuronsHidden; i++)
    {
        weightHidden[i + idx * numNeuronsOut] = weightHidden[i + idx * numNeuronsOut] - (learningRate * hError[i + idx * numNeuronsOut]);
        for (int j = 0; j < numNeuronsIn; j++)
        {
            weightHidden[i + (j + 1) * numNeuronsHidden] = weightHidden[i + (j + 1) * numNeuronsHidden] - (learningRate * hError[i + idx * numNeuronsOut] * value[j + idx * numNeuronsHidden]);
        }
    }

    results[idx] = valuesOut[idx];
}

void NeuralNetworkGPU::initializeNN()
{
    //initialize our random seed
    srand ((unsigned)time(NULL));

    //initialize the hidden layer with random numbers between (-.5,.5)
    for(int i = 0; i < _numNeuronHidden; i++)
    {
        weightHidden[i] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;

        for(int j = 1; j < _numNeuronInput + 1; j++)
            weightHidden[i + (j * _numNeuronHidden)] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;
        
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
    weightHidden = malloc(_numNeuronHidden * (_numNeuronInput + 1) * numInputValuesX * numInputValuesY * sizeof(float));
    weightsOut = malloc(_numNeuronOut * (_numNeuronHidden + 1) * numInputValuesX * numInputValuesY * sizeof(float));

    //these are 1D arrays
    valuesHidden = malloc(_numNeuronHidden * numInputValuesX * numInputValuesY * sizeof(float));
    valuesOut = malloc(_numNeuronOut * numInputValuesX * numInputValuesY * sizeof(float));
}