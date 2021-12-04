#include<iostream>
#include <math.h>
#include <vector>

#include "Matrix.h"

using namespace std;

class NeuralNetworkGPU
{
private:

    //these are the values and the weights of the hidden layer and output layer
    float* weightsHidden;
    float* weightsOut;
    float* valuesHidden;
    float* valuesOut;

    //the learning rate of the backpropagation of the neural network
    float learning_rate;

    //functions to initialize the neural network and to allocate memory of the layers
    void initializeNN();
    void allocateMemoryCPU();

    //the size of the neural network layers
    int _numNeuronInput = 0;
    int _numNeuronHidden = 0;
    int _numNeuronOut = 0;

public:

    //default constructor
    NeuralNetworkGPU(int numNeuronInput, int numNeuronHidden, int numNeuronOutput, float learningRate);

    //forward and backward propagation through the network
    Matrix& forward(Matrix& Z);
    Matrix& backPropagation(Matrix& dA);

    //train the neural network
    void train();

};