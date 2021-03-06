#include<iostream>
#include <math.h>
#include <vector>
#include <stdio.h>
//#include <chrono>
//#include <ctime>

using namespace std;

class NeuralNetworkGPU
{
private:

    //these are the values and the weights of the hidden layer and output layer
    float* weightsHidden;
    float* weightsOut;
    float* valuesHidden;
    float* valuesOut;
    int* _valuesIn;

    //the learning rate of the backpropagation of the neural network
    float learning_rate;

    //functions to initialize the neural network and to allocate memory of the layers
    void initializeNN();
    void allocateMemoryCPU();

    //the size of the neural network layers
    int _numNeuronInput;
    int _numNeuronHidden;
    int _numNeuronOut;
    int _numInputValuesX;
    int _numInputValuesY;

public:

    //default constructor
    NeuralNetworkGPU(int* valuesIn, int numNeuronInput, int numNeuronHidden, int numNeuronOutput,int numInputValuesX, int numInputValuesY, float learningRate);

    //train the neural network
    float* train(int numIterations);
    
    // trim to make the result a precision of 3 digit
    float trim(float x);

    //Method that contains the print routine for the CUDA results.
    void printResults(float* results, int* trueRes, int _numInputValuesX, int _numInputValuesY, int iteration);
};


