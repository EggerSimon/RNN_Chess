#include "RNN_Chess.cuh"

//Constructor used to allocate memory on the graphics card
RNN_Chess::RNN_Chess(int Dimensions[])
{
	variables.AllocateWorkspace(Dimensions);
	layerCalculation.InitializeVariables(variables);
	evaluation.KernelSize(Dimensions[1] * Dimensions[3]);
}

//Initializes the learning rate
int RNN_Chess::InitializeConstants(float learningrate)
{
	variables.InitializeConstants(learningrate);
	return 0;
}

//Initializes the Weight Matrices
int RNN_Chess::InitializeVariables(float** InputWeights, float** RecurrentWeights, float** Biases)
{
	variables.InitializeVariables(InputWeights, RecurrentWeights, Biases);
	return 0;
}

//Runs recurrent nerual net
float* RNN_Chess::RunRNN(float* InputState)
{
	char** descriptions = new char*[4]{ "ERR_FORGET_FORWARD" ,"ERR_INPUT_FORWARD" ,"ERR_OUTPUT_FORWARD" ,"ERR_CELL_FORWARD" };
	float*** Gates = new float**[4]{ variables.d_ForgetGate, variables.d_InputGate, variables.d_OutputGate, variables.d_CellGate };

	cudaError_t error = cudaMemcpy(variables.d_InputStates[variables.h_StateCount], InputState, variables.h_Dimensions[2] * sizeof(float), cudaMemcpyHostToDevice);
	variables.CheckCudaError(error, "ERR_VAR_INIT (InputState)");

	//Scales InputMatrix to [-1,1] => mean = 0
	inputScaling.ScaleInput(variables);

	//for each stacked LSTM block
	for (int i = 0; i < variables.h_Dimensions[3]; i++)
	{
		//for each gate
		for (int j = 0; j < 4; j++)
		{
			gateCalculations.GateCalculation(Gates[j][variables.h_StateCount], j, i, descriptions[j], variables);
		}
		layerCalculation.StateCalculation(i, variables);
	}

	variables.GetResults();
	variables.h_StateCount++;

	return variables.h_Results;
}

int RNN_Chess::ErrorCalculation(int color)
{
	cudaError_t error;
	char** descriptions = new char*[4]{ "ERR_FORGET_ERRORCALCULATION" ,"ERR_INPUT_ERRORCALCULATION" ,"ERR_OUTPUT_ERRORCALCULATION" ,"ERR_CELL_ERRORCALCULATION" };
	float*** ErrorGates = new float**[4]{ variables.d_Error_ForgetGate, variables.d_Error_InputGate, variables.d_Error_OutputGate, variables.d_Error_CellGate };

	//Sets last Inputstate to detect the end of the match
	if (variables.h_StateCount == variables.h_Dimensions[0])
	{
		inputScaling.setLastInput(variables, color);

		error = cudaMemset(variables.d_Error_HiddenStates[variables.h_StateCount], 0, variables.h_Dimensions[3] * 64 * sizeof(float));
		variables.CheckCudaError(error, "ERR_MEMSET");

		color = -1;
	}

	variables.h_StateCount--;

	error = cudaMemset(variables.d_Error_HiddenStates[variables.h_StateCount], 0, variables.h_Dimensions[3] * 64 * sizeof(float));
	variables.CheckCudaError(error, "ERR_MEMSET");

	layerCalculation.GetStateError(color, variables, evaluation);

	for (int i = 0; i < variables.h_Dimensions[3]; i++)
	{
		layerCalculation.UpdateGates(variables.h_Dimensions[3] - i - 1, variables);

		for (int j = 0; j < 4; j++)
		{
			gateCalculations.BackwardPass(ErrorGates[j], variables.h_Dimensions[3] - i - 1, j, descriptions[j], variables);
		}
	}

	return 0;
}

int RNN_Chess::BackPropagation()
{
	char** descriptions = new char*[4]{ "ERR_FORGET_BACKWARD" ,"ERR_INPUT_BACKWARD" ,"ERR_OUTPUT_BACKWARD" ,"ERR_CELL_BACKWARD" };
	float*** ErrorGates = new float**[4]{ variables.d_Error_ForgetGate, variables.d_Error_InputGate, variables.d_Error_OutputGate, variables.d_Error_CellGate };

	for (int i = 0; i < variables.h_Dimensions[0]; i++) {
		for (int j = 0; j < variables.h_Dimensions[3]; j++) {
			for (int k = 0; k < 4; k++) {
				gateCalculations.UpdateGates(ErrorGates[k], j, k, i, descriptions[k], variables);
			}
		}
	}

	return 0;
}

//Updates host weight variables
int RNN_Chess::UpdateWeightMatrices(float** InputWeights, float** RecurrentWeights, float** Biases)
{
	variables.UpdateWeightMatrices(InputWeights, RecurrentWeights, Biases);
	return 0;
}

void RNN_Chess::UpdateDimensions(int Dimensions[])
{
	cudaError_t error;

	variables.h_Dimensions = new int[6];

	for (int i = 0; i < 6; i++)
	{
		variables.h_Dimensions[i] = Dimensions[i];
	}

	if (variables.h_StateCount != 0)
	{
		std::cout << "ERR_CALCULATION" << std::endl;
		variables.h_StateCount = 0;
	}
	evaluation.UpdateEpoch(&variables);
}

//Frees the before needed workspace
int RNN_Chess::FreeWorkSpace()
{
	evaluation.UpdateEpoch(&variables);

	variables.FreeWorkspace();
	return 0;
}



