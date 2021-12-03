#include "neuralNetworkGPU.h"


__device__ SigmoidForward(float x) {return 1.0f / (1.0f + exp(-x)); }


__device__ SigmoidBackward(float x){ return x * (1.0f - x); }


void NeuralNetworkGPU::initializeNN()
{
}