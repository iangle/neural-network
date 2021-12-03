#include "neuralNetworkGPU.h"

float NeuralNetworkGPU::activationFunctionForward(float x)
{
    if(_activation == TANH)
        return tanh(x);

    if(_activation == SIGMOID)
        return 1.0f / (1.0f + exp(-x));

    return 0;
}

float NeuralNetworkGPU::activationFunctionBackward(float x)
{
    if(_activation == TANH)
        return 1.0f - (tanh(x) * tanh(x));

    if(_activation == SIGMOID)
        return x * (1.0f - x);

    return 0;
}