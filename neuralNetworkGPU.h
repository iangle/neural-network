#include<iostream>
#include <math.h>

#include "Matrix.h"

using namespace std;

enum Activation{TANH, SIGMOID};

class NeuralNetworkGPU
{
private:

    Matrix neuralNetwork;
    Activation _activation;

public:

    NeuralNetworkGPU(Activation activation);

    //code and idea for these two functions from: https://www.codeproject.com/Articles/5292985/Artificial-Neural-Network-Cplusplus-Class
    float activationFunctionForward(float x);
    float activationFunctionBackward(float x);

    void matrixMultiplication();

    void test();
    void train();


};