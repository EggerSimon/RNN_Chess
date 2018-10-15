#include "Variables.h"
#include "LayerCalculation.cuh"
#include "CudaErrors.h"

class RNN_Chess
{
public:
	RNN_Chess(int Dimensions[]);
	
	int InitializeVariables(float** InputWeights, float** RecurrentWeights, float** Biases);
	int InitializeConstants(float learningrate);

	int UpdateWeightMatrices(float** InputWeights, float** RecurrentWeights, float** Biases);
	void UpdateDimensions(int Dimensions[]);

	float* RunRNN(float* InputState);
	int BackPropagation(int color);

	int FreeWorkSpace();

	Variables variables;
private:
	LayerCalculation layerCalculation;

	int GateCalculation(float* d_Gate, int counter, int stackCount, char* description);
	int UpdateGates(float** GateError, int stackCount, int gateCount, char* description);
};
