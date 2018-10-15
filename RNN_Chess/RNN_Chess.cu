#pragma comment(lib,"cublas.lib")
#include "RNN_Chess.cuh"
#include "cublas_v2.h"
#include "cudnn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdio>
#include <ctime>

//Constructor used to allocate memory on the graphics card
RNN_Chess::RNN_Chess(int Dimensions[])
{
	variables.AllocateWorkspace(Dimensions);
	layerCalculation.InitializeVariables(variables);
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

//Calculates the float values for each gate
int RNN_Chess::GateCalculation(float* d_Gate, int counter, int stackCount, char* description)
{
	cublasStatus_t cublasState;
	cudnnStatus_t cudnnStatus;

	const float alpha = 1;
	const float beta = 0;
	int lastOffset = (stackCount - 1) * variables.h_Dimensions[1];
	int stackOffset = stackCount * variables.h_Dimensions[1];
	int weightOffset = stackCount * pow(variables.h_Dimensions[1], 2);

	//matrix calculation of the hidden state
	cublasState = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1], 1, variables.h_Dimensions[1], &alpha, variables.d_RecurrentWeights[counter] + weightOffset,
		variables.h_Dimensions[1], variables.d_HiddenStates[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1], &beta, variables.d_InterstageVar[0], variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasState, description);
	//matrix calculation of the input state
	if(stackCount == 0)
	{
		cublasState = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1], 1, variables.h_Dimensions[2], &alpha, variables.d_InputWeights[counter], variables.h_Dimensions[1],
			variables.d_InputStates[variables.h_StateCount], variables.h_Dimensions[1], &beta, variables.d_InterstageVar[1], variables.h_Dimensions[1]);
	}
	//input state is equal to the hidden state of the stack below
	else
	{
		cublasState = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1], 1, variables.h_Dimensions[2], &alpha, variables.d_InputWeights[counter] + weightOffset,
			variables.h_Dimensions[1], variables.d_HiddenStates[variables.h_StateCount] + lastOffset, variables.h_Dimensions[1], &beta, variables.d_InterstageVar[1], variables.h_Dimensions[1]);
	}
	//adds both matrices
	variables.CheckCublasStatus(cublasState, description);
	cublasState = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1], 1, &alpha, variables.d_InterstageVar[0], variables.h_Dimensions[1], &alpha,
		variables.d_InterstageVar[1], variables.h_Dimensions[1], variables.d_InterstageVar[2], variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasState, description);

	cudaMemset(variables.d_InterstageVar[1], 0, 64 * sizeof(float));

	//adds Bias
	variables.CheckCublasStatus(cublasState, description);
	cublasState = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1], 1, &alpha, variables.d_InterstageVar[2], variables.h_Dimensions[1], &alpha,
		variables.d_Biases[counter] + stackOffset, variables.h_Dimensions[1], variables.d_InterstageVar[1], variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasState, description);

	//Sigmoid or Tanh activation
	cudnnStatus = cudnnActivationForward(variables.cudnn, variables.activation_descriptor[counter], &alpha, variables.tensor_descriptor, variables.d_InterstageVar[1], &beta,
		variables.tensor_descriptor, d_Gate + stackOffset);
	variables.CheckCudnnStatus(cudnnStatus, description);

	return 0;
}

//Runs recurrent nerual net
float* RNN_Chess::RunRNN(float* InputState)
{
	char** descriptions = new char*[4]{"ERR_FORGET_FORWARD" ,"ERR_INPUT_FORWARD" ,"ERR_OUTPUT_FORWARD" ,"ERR_CELL_FORWARD"};
	float*** Gates = new float**[4]{variables.d_ForgetGate, variables.d_InputGate, variables.d_OutputGate, variables.d_CellGate};

	cudaError_t error = cudaMemcpy(variables.d_InputStates[variables.h_StateCount], InputState, variables.h_Dimensions[2] * sizeof(float), cudaMemcpyHostToDevice);
	variables.CheckCudaError(error, "ERR_VAR_INIT (InputState)");

	//for each stacked LSTM block
	for(int i = 0; i < variables.h_Dimensions[3]; i++)
	{
		//for each gate
		for(int j = 0; j < 4; j++)
		{
			GateCalculation(Gates[j][variables.h_StateCount], j, i, descriptions[j]);
		}
		layerCalculation.StateCalculation(i, variables);
	}

	//variables.GetResults();
	variables.h_StateCount++;

	return variables.h_Results;
}

int RNN_Chess::UpdateGates(float** GateError, int stackCount, int gateCount, char* description)
{
	cublasStatus_t cublasStatus;

	const float alpha = variables.getLearningRate();
	const float beta = 1;
	const float beta1 = 0;

	int lastOffset = (stackCount - 1)* variables.h_Dimensions[1];
	int stackOffset = stackCount * variables.h_Dimensions[1];
	int weightOffset = stackCount * pow(variables.h_Dimensions[1], 2);

	//Backpropagation of the Recurrent weights
	cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_T, variables.h_Dimensions[1], variables.h_Dimensions[1], 1, &alpha, GateError[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1],
		variables.d_HiddenStates[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1], &beta, variables.d_RecurrentWeights[gateCount] + weightOffset, variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasStatus, description);

	if(stackCount == 0)
	{
		//Backpropagation of the Input weights
		cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_T, variables.h_Dimensions[1], variables.h_Dimensions[1], 1, &alpha, GateError[variables.h_StateCount] + stackOffset,
			variables.h_Dimensions[1], variables.d_InputStates[variables.h_StateCount], variables.h_Dimensions[1], &beta, variables.d_InputWeights[gateCount] + weightOffset, variables.h_Dimensions[1]);
		variables.CheckCublasStatus(cublasStatus, description);
	}
	else
	{
		//Backward feed of error to InputState
		cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_T, CUBLAS_OP_N, variables.h_Dimensions[1], 1, variables.h_Dimensions[1], &beta, variables.d_InputWeights[gateCount] + weightOffset,
			variables.h_Dimensions[1], GateError[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1], &beta, variables.d_Error_HiddenStates[variables.h_StateCount + 1] + lastOffset, variables.h_Dimensions[1]);
		variables.CheckCublasStatus(cublasStatus, description);

		//Backpropagation of the Input weights
		cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_T, variables.h_Dimensions[1], variables.h_Dimensions[1], 1, &alpha, GateError[variables.h_StateCount] + stackOffset,
			variables.h_Dimensions[1], variables.d_HiddenStates[variables.h_StateCount] + lastOffset, variables.h_Dimensions[1], &beta, variables.d_InputWeights[gateCount] + weightOffset, variables.h_Dimensions[1]);
		variables.CheckCublasStatus(cublasStatus, description);
	}

	//Backpropagation of the Biases
	cublasStatus = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1], 1, &alpha, GateError[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1], &beta,
		variables.d_Biases[gateCount] + stackOffset, variables.h_Dimensions[1], variables.d_Biases[gateCount] + stackOffset, variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasStatus, description);

	//Error Calculation of Hiddenstate
	cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_T, CUBLAS_OP_N, variables.h_Dimensions[1], 1, variables.h_Dimensions[1], &beta, variables.d_RecurrentWeights[gateCount] + weightOffset,
		variables.h_Dimensions[1], GateError[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1], &beta, variables.d_Error_HiddenStates[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasStatus, description);

	return 0;
}

int RNN_Chess::BackPropagation(int color)
{
	cudaError_t error;
	char** descriptions = new char*[4]{"ERR_FORGET_BACKWARD" ,"ERR_INPUT_BACKWARD" ,"ERR_OUTPUT_BACKWARD" ,"ERR_CELL_BACKWARD"};
	float*** ErrorGates = new float**[4]{variables.d_Error_ForgetGate, variables.d_Error_InputGate, variables.d_Error_OutputGate, variables.d_Error_CellGate};

	//Sets last Inputstate to detect the end of the match
	if(variables.h_StateCount == variables.h_Dimensions[0])
	{
		cublasStatus_t cublasStatus;

		const float alpha = 1;
		const float beta = 1 / -0.00572916633f;

		error = cudaMemset(variables.d_InterstageVar[0], 187, 32 * sizeof(float));
		variables.CheckCudaError(error, "ERR_ENDMATCH");
		error = cudaMemset(variables.d_InputStates[variables.h_StateCount], 0, 64 * sizeof(float));
		variables.CheckCudaError(error, "ERR_ENDMATCH");

		cublasStatus = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1] / 2, 1, &alpha, variables.d_InputStates[variables.h_StateCount] + color * 32, variables.h_Dimensions[1] / 2,
			&beta, variables.d_InterstageVar[0], variables.h_Dimensions[1] / 2, variables.d_InputStates[variables.h_StateCount] + color * 32, variables.h_Dimensions[1] / 2);
		variables.CheckCublasStatus(cublasStatus, "ERR_ENDMATCH");

		error = cudaMemset(variables.d_Error_HiddenStates[variables.h_StateCount], 0, variables.h_Dimensions[3] * 64 * sizeof(float));
		variables.CheckCudaError(error, "ERR_MEMSET");

		color = -1;
	}

	variables.h_StateCount--;

	error = cudaMemset(variables.d_Error_HiddenStates[variables.h_StateCount], 0, variables.h_Dimensions[3] * 64 * sizeof(float));
	variables.CheckCudaError(error, "ERR_MEMSET");

	layerCalculation.GetStateError(color, variables);

	for(int i = 0; i < variables.h_Dimensions[3]; i++)
	{
		layerCalculation.UpdateGates(variables.h_Dimensions[3] - i - 1, variables);

		for(int j = 0; j < 4; j++)
		{
			UpdateGates(ErrorGates[j], variables.h_Dimensions[3] - i - 1, j, descriptions[j]);
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
	variables.h_Dimensions = new int[4];

	for(int i = 0; i < 4; i++)
	{
		variables.h_Dimensions[i] = Dimensions[i];
	}
	
	if(variables.h_StateCount != 0)
	{
		std::cout << "ERR_CALCULATION" << std::endl;
		variables.h_StateCount = 0;
	}
}

//Frees the before needed workspace
int RNN_Chess::FreeWorkSpace()
{
	variables.FreeWorkspace();
	return 0;
}



