#include "Variables.h"
#include "LayerCalculation.cuh"
#include "GateCalculation.h"
#include "InputScaling.h"
#include "Evaluation.cuh"

#include <cstdio>
#include <ctime>

class RNN_Chess
{
public:
	RNN_Chess(int Dimensions[]);
	
	int InitializeVariables(float** InputWeights, float** RecurrentWeights, float** Biases);
	int InitializeConstants(float learningrate);

	int UpdateWeightMatrices(float** InputWeights, float** RecurrentWeights, float** Biases);
	void UpdateDimensions(int Dimensions[]);

	float* RunRNN(float* InputState);
	int ErrorCalculation(int color);
	int BackPropagation();

	int FreeWorkSpace();

	Variables variables;
private:
	LayerCalculation layerCalculation;
	GateCalculations gateCalculations;
	InputScaling inputScaling;
	Evaluation evaluation;
};
