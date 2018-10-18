#include "GateCalculation.h"

//Calculates the float values for each gate
int GateCalculations::GateCalculation(float* d_Gate, int counter, int stackCount, char* description, Variables variables)
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
	if (stackCount == 0)
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

int GateCalculations::BackwardPass(float** d_GateError, int stackCount, int gateCount, char* description, Variables variables)
{
	cublasStatus_t cublasStatus;

	const float alpha = 1;
	int lastOffset = (stackCount - 1)* variables.h_Dimensions[1];
	int stackOffset = stackCount * variables.h_Dimensions[1];
	int weightOffset = stackCount * pow(variables.h_Dimensions[1], 2);

	//Error Calculation of Hiddenstate
	cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_T, CUBLAS_OP_N, variables.h_Dimensions[1], 1, variables.h_Dimensions[1], &alpha, variables.d_RecurrentWeights[gateCount] + weightOffset,
		variables.h_Dimensions[1], d_GateError[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1], &alpha, variables.d_Error_HiddenStates[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasStatus, description);

	if (stackCount != 0)
	{
		//Backward feed of error to InputState
		cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_T, CUBLAS_OP_N, variables.h_Dimensions[1], 1, variables.h_Dimensions[1], &alpha, variables.d_InputWeights[gateCount] + weightOffset,
			variables.h_Dimensions[1], d_GateError[variables.h_StateCount] + stackOffset, variables.h_Dimensions[1], &alpha, variables.d_Error_HiddenStates[variables.h_StateCount + 1] + lastOffset, variables.h_Dimensions[1]);
		variables.CheckCublasStatus(cublasStatus, description);
	}

	return 0;
}


int GateCalculations::UpdateGates(float** GateError, int stackCount, int gateCount, int stateCount, char* description, Variables variables)
{
	cublasStatus_t cublasStatus;

	const float alpha = -0.001f;
	const float beta = 1;

	int lastOffset = (stackCount - 1)* variables.h_Dimensions[1];
	int stackOffset = stackCount * variables.h_Dimensions[1];
	int weightOffset = stackCount * pow(variables.h_Dimensions[1], 2);

	//Backpropagation of the Recurrent weights
	cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_T, variables.h_Dimensions[1], variables.h_Dimensions[1], 1, &alpha, GateError[stateCount] + stackOffset, variables.h_Dimensions[1],
		variables.d_HiddenStates[stateCount] + stackOffset, variables.h_Dimensions[1], &beta, variables.d_RecurrentWeights[gateCount] + weightOffset, variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasStatus, description);

	if (stackCount == 0)
	{
		//Backpropagation of the Input weights
		cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_T, variables.h_Dimensions[1], variables.h_Dimensions[1], 1, &alpha, GateError[stateCount] + stackOffset,
			variables.h_Dimensions[1], variables.d_InputStates[stateCount], variables.h_Dimensions[1], &beta, variables.d_InputWeights[gateCount] + weightOffset, variables.h_Dimensions[1]);
		variables.CheckCublasStatus(cublasStatus, description);
	}
	else
	{
		//Backpropagation of the Input weights
		cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_T, variables.h_Dimensions[1], variables.h_Dimensions[1], 1, &alpha, GateError[stateCount] + stackOffset,
			variables.h_Dimensions[1], variables.d_HiddenStates[stateCount] + lastOffset, variables.h_Dimensions[1], &beta, variables.d_InputWeights[gateCount] + weightOffset, variables.h_Dimensions[1]);
		variables.CheckCublasStatus(cublasStatus, description);
	}

	//Backpropagation of the Biases
	cublasStatus = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1], 1, &alpha, GateError[stateCount] + stackOffset, variables.h_Dimensions[1], &beta,
		variables.d_Biases[gateCount] + stackOffset, variables.h_Dimensions[1], variables.d_Biases[gateCount] + stackOffset, variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasStatus, description);

	return 0;
}