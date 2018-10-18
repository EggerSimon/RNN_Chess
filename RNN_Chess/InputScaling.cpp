#include "InputScaling.h"

int InputScaling::ScaleInput(Variables variables)
{
	cublasStatus_t cublasStatus;
	cudaError_t error;

	/*Scalars for the Input variables*/
	const float alpha = 1 / 7.0f;
	const float beta = 1 / (2 * 0.00572916633f);

	error = cudaMemset(variables.d_InterstageVar[0], 187, 64 * sizeof(float));
	variables.CheckCudaError(error, "ERR_ENDMATCH");

	//Shifts the scaled inputs by -0.5, such that its mean is at 0
	cublasStatus = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1], 1, &alpha, variables.d_InputStates[variables.h_StateCount], variables.h_Dimensions[1],
		&beta, variables.d_InterstageVar[0], variables.h_Dimensions[1], variables.d_InputStates[variables.h_StateCount], variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasStatus, "ERR_ENDMATCH");

	return 0;
}

//Sets unified last input, such that the RNN network learns to recognize a games end
int InputScaling::setLastInput(Variables variables, int color)
{
	cublasStatus_t cublasStatus;
	cudaError_t error;

	const float alpha = 1;
	const float beta = 1 / -(2 * 0.00572916633f);

	error = cudaMemset(variables.d_InterstageVar[0], 187, 32 * sizeof(float));
	variables.CheckCudaError(error, "ERR_ENDMATCH");
	error = cudaMemset(variables.d_InputStates[variables.h_StateCount], 0, 64 * sizeof(float));
	variables.CheckCudaError(error, "ERR_ENDMATCH");

	cublasStatus = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1] / 2, 1, &alpha, variables.d_InputStates[variables.h_StateCount] + color * 32, variables.h_Dimensions[1] / 2,
		&beta, variables.d_InterstageVar[0], variables.h_Dimensions[1] / 2, variables.d_InputStates[variables.h_StateCount] + (1-color) * 32, variables.h_Dimensions[1] / 2);
	variables.CheckCublasStatus(cublasStatus, "ERR_ENDMATCH");

	return 0;
}