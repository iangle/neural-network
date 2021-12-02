#include <iostream>

int main()
{
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


	bpNeuralNetwork<int> myBPNN;
	myBPNN.training( input,output,64,0.02f,100000l,fx);
	cout << "\n\n\n                Press any key to exit!";
	getchar();
	return 0;
}