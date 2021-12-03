#include "Matrix.h"

//initialize all the values for the constructor
Matrix::Matrix(size_t x_dim, size_t y_dim) :
    data_device(nullptr), data_host(nullptr),
    device_allocated(false), host_allocated(false)
{}

//allocate the memory on the cuda side
void Matrix::allocateCudaMemory()
{
    if(!device_allocated)
    {
        float* device_memory = nullptr;
        cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
        data_device = shared_ptr<float>(device_memory, [&](float* ptr){ cudaFree(ptr); });
        device_allocated = true;
    }
}
//allocate the memory on the host side
void Matrix::allocateHostMemory()
{
    if(!host_allocated)
    {
        data_host = shared_ptr<float>(new float[shape.x * shape.y], [&](float* ptr){ delete[] ptr; });
        host_allocated = true;
    }
}

//allocate memory for the host and the device
void Matrix::allocateMemory()
{
    allocateCudaMemory();
    allocateHostMemory();
}

//allocate memory if not already allocated for our shape
void Matrix::allocateMemoryIfNotAllocated(Shape shape)
{
    if(!device_allocated && !host_allocated)
    {
        this->shape = shape;
        allocateMemory();
    }
}

//copy memory from the host back to the device
void Matrix::copyHostToDevice()
{
    if(device_allocated && host_allocated)
    {
        cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
    }else
    {
        cout << "sorry, could not copy memory from the host to the device because the device or host memory has not been allocated yet";
    }
}


//copy memory from the device to the host
void Matrix::copyDeviceToHost()
{
    if(device_allocated && host_allocated)
    {
        cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
    }else
    {
        cout << "sorry, could not copy memory from the device to the host because the device or host memory has not been allocated yet";
    }
}


float& Matrix::operator[](const int index) {   return data_host.get()[index]; }

const float& Matrix::operator[](const int index) const{ return data_host.get()[index]; }

Shape::Shape(size_t x, size_t y) :
    x(x), y(y)
{}