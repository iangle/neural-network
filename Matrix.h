/* This class will be used to create a matrix that will allow us to create the neural network.
The idea for this class and its code is derived from: https://github.com/pwlnk/cuda-neural-network/blob/master/cuda-neural-network/src/nn_utils/ 
*/
#pragma once

#include <iostream>
#include <memory>

using namespace std;

class Matrix
{

private:

    //tells us whether the device and host have had memory allocated yet
    bool device_allocated;
    bool host_allocated;

    //these are private because they are used in the public allocate memory function
    void allocateCudaMemory();
    void allocateHostMemory();

public:

    Shape shape;

    //share pointers that represent the data on the device and the host
    shared_ptr<float> data_device;
    shared_ptr<float> data_host;

    //constructors that gives us the ability to create matrices
    Matrix(size_t x_dim = 1, size_t y_dim = 1);
    Matrix(Shape shape);
    
    //allocate memory for the matrix and for the shape if needed
    void allocateMemory();
    void allocateMemoryIfNotAllocated(Shape shape);
    
    //copy data from host to device and from device to host
    void copyHostToDevice();
    void copyDeviceToHost();

    //overloading the [] operator to allow for indexing a created Matrix
    float& operator[](const int index);
    const float& operator[](const int index) const;

};

//a structure that allows use to create a shape with an x and y size
struct Shape
{
    size_t x,y;

    Shape(size_t x = 1, size_t y = 1);
};