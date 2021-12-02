#include <iostream>
#include <timing.h>
#include <iomanip>
#include <neuralNetworkCPU.h>
#include <neuralNetworkGPU.cu>

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
	float fx(float);
	int i,j,k = 0,input[64][2] = {0},output[64][1] = {0};

	for(i = 0; i < 8; i ++)
		for(j = 0; j< 8; j++)
		{
			input[k][0] = i;
			input[k][1] = j;
			output[k][0] = indata[i][j];
			k ++;
		}


	//bpNeuralNetwork<int> myBPNN;
	
	//CPU Run and Timing Block.
	then = currentTime();
	//myBPNN.training( input,output,64,0.02f,100000l,fx);
	now = currentTime();
	cTimeCost = then - now;

	//GPU Run and Timing Block.
	then = currentTime();
	//GPU Call
	now = currentTime();
	gTimeCost = then - now;

	printTimes(gTimeCost, cTimeCost);

	cout << "\n\n\n                Press any key to exit!";
	getchar();
	return 0;
}

//Takes the GPU and CPU time cost and prints the values and speedup factor. 
void printTimes(float gTimeCost, float cTimeCost)
{
	cout << setprecision(3) << "Training the network via the CPU resulted in a time cost of " << cTimeCost << " in seconds.\n";
    cout << setprecision(3) << "Training the network via the GPU resulted in a time cost of " << gTimeCost << " in seconds.\n";
	cout << setprecision(3) << "Training with the GPU resulted in a speed up factor of " << cTimeCost/gTimeCost << ".\n";

}