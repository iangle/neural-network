#include<iostream>
#include <math.h>
#include <vector>

#include "Matrix.h"

using namespace std;

class NeuralNetworkGPU
{
private:

    Matrix A;
    Matrix Z;
    Matrix dz;

    float learning_rate;

public:

    NeuralNetworkGPU(float learningRate);

    Matrix& forward(Matrix& Z);
    Matrix& backPropagation(Matrix& dA);

    void test();
    void train();

};