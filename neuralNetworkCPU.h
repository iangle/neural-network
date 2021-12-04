#pragma once

using namespace std;

#define numOut_ 1;
#define numIn_ 2;
#define numH_ 3;

class neuralNetworkCPU
{
    class bpNeuralNetwork
    {
        private:
            int numNeuronIn_;
	        int numNeuronHidden_;
	        int numNeuronOut_;

	        T indata_[numIn_];								//the input data for the input layer
	        float wHidden_[numH_][numIn_ + 1];			                //the weight belongs to Hidden Layer
	        float wOut_[numOut_][numH_ + 1];							//the weight belongs to Output Layer
	        float vHidden_[numH_];							//the value of Neuron in Hidden Layer
	        float vOut_[numOut_];								//the value of Neuron in Output Layer

  
	    public:
    }

    float fx(float x);
    float myTrim(float x);
}