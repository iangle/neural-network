#include<iostream>
#include <math.h>
#include <vector>

using namespace std;

class NeuralNetworkGPU
{
private:

    //these are the values and the weights of the hidden layer and output layer
    float* weightsHidden;
    float* weightsOut;
    float* valuesHidden;
    float* valuesOut;
    float* _valuesIn;

    //the learning rate of the backpropagation of the neural network
    float learning_rate;

    //functions to initialize the neural network and to allocate memory of the layers
    void initializeNN();
    void allocateMemoryCPU();

    //the size of the neural network layers
    int _numNeuronInput = 0;
    int _numNeuronHidden = 0;
    int _numNeuronOut = 0;
    int _numInputValuesX;
    int _numInputValuesY;

public:

    //default constructor
    NeuralNetworkGPU(float* valuesIn, int numNeuronInput, int numNeuronHidden, int numNeuronOutput,int numInputValuesX, int numInputValuesY, float learningRate);

    //train the neural network
    float* train(int numIterations, int tile_width);

};