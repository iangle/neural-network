#include "NeuralNetworkGPU.h"
#include <iomanip>
#include <iostream>
#include "NeuralNetworkCPU.h"

using namespace std;

NeuralNetworkGPU::NeuralNetworkGPU(int* valuesIn, int numNeuronInput, int numNeuronHidden, int numNeuronOutput, int numInputValuesX, int numInputValuesY, float learningRate)
{
    learning_rate = learningRate;

    _numNeuronInput = numNeuronInput;
    _numNeuronHidden = numNeuronHidden;
    _numNeuronOut = numNeuronOutput;
    _numInputValuesX = numInputValuesX;
    _numInputValuesY = numInputValuesY;
    _valuesIn = valuesIn;

    allocateMemoryCPU();

    initializeNN();

}

__device__ float Sigmoid(float x) {return 1.0f / (1.0f + exp(-x)); }

__global__ void forwardHidden(float* weightHidden, float* valuesHidden, float* weightOut, float* valuesOut,
int numNeuronsHidden, int numNeuronsOut, int numInputValuesX, int numInputValuesY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * numInputValuesY + x;

    if(x >= numInputValuesX || y >= numInputValuesY) return;

    float value = 0;

    // compute valuesHidden
    for(int i = 0; i < numNeuronsHidden; i++)
    {
        value = value + x * weightHidden[(i + numNeuronsHidden) + idx * numNeuronsHidden];
        value = value + y * weightHidden[i + (2 * numNeuronsHidden) + idx * numNeuronsHidden];
        value = value + weightHidden[i + idx * numNeuronsHidden];
        valuesHidden[i + idx * numNeuronsHidden] = Sigmoid(static_cast<float>(value));

        value = 0;
    }

    // compute valuesOut
    for(int i = 0; i < numNeuronsOut; i++)
    {

        for(int j = 0; j < numNeuronsHidden; j++)
            value = value + valuesHidden[j + idx * numNeuronsHidden] * weightOut[i + ((j + 1) * numNeuronsHidden)];
        
        value = value + weightOut[i + idx * numNeuronsHidden];
        valuesOut[idx] = Sigmoid(static_cast<float>(value));

        value = 0;
    }
}

__global__ void backHidden(float* valuesHidden, float* weightOut, float* valuesOut, 
float* yError, float* hError, int* trueOut, int numNeuronsHidden, int numInputValuesX, int numInputValuesY)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * numInputValuesY + x;

    float temp = 0;

    if(x >= numInputValuesX || y >= numInputValuesY) return;

    //compute yError
    yError[idx] = valuesOut[idx] * (1 - valuesOut[idx]) * (valuesOut[idx] - trueOut[idx]);

    //compute hError
    for (int i = 0; i < numNeuronsHidden; i++)
    {
        temp = 0;

		temp = temp + weightOut[((i + 1) * numNeuronsHidden)] * yError[idx];

        hError[i + idx * numNeuronsHidden] = temp * valuesHidden[i + idx * numNeuronsHidden] * (1 - valuesHidden[i + idx * numNeuronsHidden]);

    }
}

__global__ void adjustWeights(float* weightHidden, float* valuesHidden, float* weightOut, float* valuesOut, float* yError, float* hError,
float* results, int numNeuronsHidden, int numNeuronsOut, float learningRate, int numInputValuesX, int numInputValuesY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * numInputValuesY + x;

    if(x >= numInputValuesX || y >= numInputValuesY) return;

    //adjusting weights out
        weightOut[idx * numNeuronsHidden] = weightOut[idx * numNeuronsHidden] - (learningRate * yError[idx]);

    
        for(int j = 0; j < numNeuronsHidden; j++)
        {
            weightOut[(j + 1) * numNeuronsHidden] = weightOut[(j + 1) * numNeuronsHidden] - (learningRate * yError[idx] * valuesHidden[j + idx * numNeuronsHidden]);
        }

    //adjusting weights hidden
    for (int i = 0; i < numNeuronsHidden; i++)
    {
        weightHidden[i + idx * numNeuronsHidden] = weightHidden[i + idx * numNeuronsHidden] - (learningRate * hError[i + idx * numNeuronsHidden]);

        weightHidden[i + numNeuronsHidden] = weightHidden[i + numNeuronsHidden] - (learningRate * hError[i + idx * numNeuronsHidden] * x);
        weightHidden[i + (2 * numNeuronsHidden)] = weightHidden[i + (2 * numNeuronsHidden)] - (learningRate * hError[i + idx * numNeuronsHidden] * y);
    }

    //saving output values in results
    results[idx] = valuesOut[idx];
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
    weightsHidden = (float*) malloc(_numNeuronHidden * (_numNeuronInput + 1) * _numInputValuesX * _numInputValuesY * sizeof(float));
    weightsOut = (float*) malloc(_numNeuronOut * (_numNeuronHidden + 1) * _numInputValuesX * _numInputValuesY * sizeof(float));

    //these are 1D arrays
    valuesHidden = (float*) malloc(_numNeuronHidden * _numInputValuesX * _numInputValuesY * sizeof(float));
    valuesOut = (float*) malloc(_numNeuronOut * _numInputValuesX * _numInputValuesY * sizeof(float));
}

float* NeuralNetworkGPU::train(int numIterations)
{
    int n = _numInputValuesX * _numInputValuesY;

    dim3 blockSize, gridSize;

    blockSize.x = _numInputValuesX;
    blockSize.y = _numInputValuesY;

    gridSize.x = ceil((float) _numInputValuesX / blockSize.x);
    gridSize.y = ceil((float) _numInputValuesY / blockSize.y);

    //create some arrays that we will allocate on the device
    float* cudaWeightHidden;
    float* cudaValuesHidden;
    float* cudaWeightOut;
    float* cudaValuesOut; 
    float* cudaYError;
    float* cudaHError;
    int* cudaTrueOut;
    float* cudaResults;

    float* temp = (float*) malloc(sizeof(float) * n);

    float* results = (float*) malloc(n * sizeof(float));

    //allocate memory on the device
    cudaMalloc(&cudaWeightHidden, _numNeuronHidden * (_numNeuronInput + 1) * _numInputValuesX * _numInputValuesY * sizeof(float));
    cudaMalloc(&cudaValuesHidden, _numNeuronHidden * _numInputValuesX * _numInputValuesY * sizeof(float));
    cudaMalloc(&cudaWeightOut, _numNeuronOut * (_numNeuronHidden + 1) * _numInputValuesX * _numInputValuesY * sizeof(float));
    cudaMalloc(&cudaValuesOut, _numNeuronOut * _numInputValuesX * _numInputValuesY * sizeof(float));
    cudaMalloc(&cudaYError, _numNeuronOut * _numInputValuesX * _numInputValuesY * sizeof(float));
    cudaMalloc(&cudaHError, _numNeuronHidden * _numInputValuesX * _numInputValuesY * sizeof(float));
    cudaMalloc(&cudaTrueOut, n * sizeof(int));
    cudaMalloc(&cudaResults, n * sizeof(float));

    cudaMemcpy(cudaWeightHidden, weightsHidden,  _numNeuronHidden * (_numNeuronInput + 1) * _numInputValuesX * _numInputValuesY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaValuesHidden, valuesHidden, _numNeuronHidden * _numInputValuesX * _numInputValuesY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaWeightOut, weightsOut, _numNeuronOut * (_numNeuronHidden + 1) * _numInputValuesX * _numInputValuesY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaValuesOut, valuesOut, _numNeuronOut * _numInputValuesX * _numInputValuesY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaTrueOut, _valuesIn, n * sizeof(int), cudaMemcpyHostToDevice);

    cout << "GPU Tests: \n";
    cout << "================== \n";

    for(int i = 1; i <= numIterations; i++)
    {
        forwardHidden<<<gridSize, blockSize>>>(cudaWeightHidden, cudaValuesHidden, cudaWeightOut, cudaValuesOut,
        _numNeuronHidden, _numNeuronOut, _numInputValuesX, _numInputValuesY);

         cudaDeviceSynchronize();

        cudaMemcpy(temp, cudaValuesOut, sizeof(float) * n, cudaMemcpyDeviceToHost);

        backHidden<<<gridSize, blockSize>>>(cudaValuesHidden, cudaWeightOut, cudaValuesOut, cudaYError, cudaHError,
        cudaTrueOut, _numNeuronHidden, _numInputValuesX, _numInputValuesY);

        cudaDeviceSynchronize();

        adjustWeights<<<gridSize, blockSize>>>(cudaWeightHidden, cudaValuesHidden, cudaWeightOut, cudaValuesOut,
        cudaYError, cudaHError, cudaResults, _numNeuronHidden, _numNeuronOut, learning_rate, _numInputValuesX, _numInputValuesY);
        
        cudaDeviceSynchronize();
        
        if(i == 10 || i == 100 || i == 1000 || i == 10000 || i == 100000 )
	    {   
	        cudaMemcpy(results, cudaResults, sizeof(float) * n, cudaMemcpyDeviceToHost);
                printResults(results, _valuesIn, _numInputValuesX, _numInputValuesY, i);
	    }
    }

    cudaMemcpy(results, cudaResults, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cudaFree(cudaWeightHidden);
    cudaFree(cudaValuesHidden);
    cudaFree(cudaWeightOut);
    cudaFree(cudaValuesOut);
    cudaFree(cudaYError);
    cudaFree(cudaHError);
    cudaFree(cudaTrueOut);
    cudaFree(cudaResults);

    free(weightsHidden);  
    free(valuesHidden);
    free(weightsOut);
    free(valuesOut);
    free(_valuesIn);

    return results;
}

//Method that contains the print routine for the CUDA results.
void NeuralNetworkGPU::printResults(float* results, int* trueRes, int _numInputValuesX, int _numInputValuesY, int iteration)
{
    int errors = 0;
    int j, k;
    cout << "Ouput values after " << iteration << " iterations: \n";
	for(j = 0; j < 8; j++)
	{
		for(k = 0; k < 8; k ++)
        {
			if ( (int)(results[j * 8 + k] + 0.5) == trueRes[j * 8 + k])
			{
				cout << setw(6) << trim(results[j * 8 + k]) << "  ";
			}
		    else
			{
				cout << setw(6) << trim(results[j * 8 + k]) << "* ";
				errors++;
			}
        }
			cout << "\n";
	}
	cout << "==> " << errors << "  errors\n\n";
}

// trim to make the result a precision of 3 digit
float NeuralNetworkGPU::trim(float x)
{
	int a = 0;
	a = static_cast<int>(x * 1000 + 0.5);      // keep a precision of 3 digit
	return (static_cast<float>(a) / 1000);
}
