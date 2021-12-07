#include <iostream>
#include <iomanip>
#include "timing.h"
#include "NeuralNetworkCPU.cpp"
#include "NeuralNetworkGPU.h"

using namespace std;

int main()
{

	int indata[8][8] = {
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 0,0,0,0, 0,0,0,0}
						};

	int indata2[64][64] = {0};
	float fx(float);

	int xSize = 8;
	int ySize = 8;

	int i,j,k = 0,input[64][2] = {0},output[64][1] = {0};

	int* GPUOutput = (int*) malloc(xSize * ySize * sizeof(int));

	float* GPUAnswer = (float*) malloc(xSize * ySize * sizeof(float));

	for(i = 0; i < 8; i ++)
		for(j = 0; j< 8; j++)
		{
			input[k][0] = i;
			input[k][1] = j;
			output[k][0] = indata[i][j];
			GPUOutput[k] = output[k][0];
			k ++;
		}


	bpNeuralNetwork<int> myBPNN;
	
	//CPU Run and Timing Block.
	//auto then = chrono::system_clock::now();
	myBPNN.training(input,output,64,0.02f,100000l,fx);
	//auto now = chrono::system_clock::now(); 
	//chrono::duration<double> cTimeCost = (now - then);

	//GPU Run and Timing Block.

	
	NeuralNetworkGPU::NeuralNetworkGPU NN_GPU(GPUOutput, 2, 3, 1, 8, 8, 0.02f);

	//then = chrono::system_clock::now();
	
	GPUAnswer = NN_GPU.train(100000);

	//now = chrono::system_clock::now();

	//chrono::duration<double> gTimeCost = (now - then);

	//printTimes(gTimeCost.count(), cTimeCost.count());

	cout << "\n\n\n                Press any key to exit!";
	getchar();
	return 0;
}
