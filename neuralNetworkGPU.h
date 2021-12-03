#include<iostream>
#include <math.h>
#include <vector>

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

    void initializeNN();

    void matrixMultiplication();

    void test();
    void train();


};