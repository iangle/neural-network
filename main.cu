#include <iostream>
#include "timing.h"
#include <iomanip>
#include "NeuralNetworkCPU.cpp"
#include "NeuralNetworkGPU.h"

using namespace std;

int main()
{
	float now, then, gTimeCost, cTimeCost;

	int indata[8][8] = {
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,0,1, 1,1,1,1},
							{ 1,0,0,0, 1,1,1,1},
							{ 1,0,0,0, 0,0,1,1},
							{ 0,0,0,0, 0,0,1,1},
							{ 0,0,0,0, 0,1,1,1},
							{ 0,0,0,1, 1,1,1,1}
						};

	int indata2[8][8] = {
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,1},
							{ 1,1,1,1, 1,1,1,0}
						};
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
			output[k][0] = indata2[i][j];
			GPUOutput[k] = output[k][0];
			k ++;
		}


	bpNeuralNetwork<int> myBPNN;
	
	//CPU Run and Timing Block.
	then = currentTime();
	myBPNN.training( input,output,64,0.02f,100000l,fx);
	now = currentTime();
	cTimeCost = then - now;

	//GPU Run and Timing Block.
	then = currentTime();
	
	NeuralNetworkGPU::NeuralNetworkGPU NN_GPU(GPUOutput, 2, 3, 1, 8, 8, 0.02f);

	GPUAnswer = NN_GPU.train(10000);

	now = currentTime();
	gTimeCost = then - now;

	printTimes(gTimeCost, cTimeCost);

	cout << "\n\n\n                Press any key to exit!";
	getchar();
	return 0;
}
