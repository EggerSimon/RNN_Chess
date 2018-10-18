#include "InputScaling.h"

int InputScaling::ScaleInput(Variables variables)
{
	cublasStatus_t cublasStatus;
	cudaError_t error;

	const float alpha = 1 / 3.5f;
	const float beta = 1 / (0.00572916633f);

	error = cudaMemset(variables.d_InterstageVar[0], 187, 64 * sizeof(float));
	variables.CheckCudaError(error, "ERR_ENDMATCH");

	cublasStatus = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1], 1, &alpha, variables.d_InputStates[variables.h_StateCount], variables.h_Dimensions[1],
		&beta, variables.d_InterstageVar[0], variables.h_Dimensions[1], variables.d_InputStates[variables.h_StateCount], variables.h_Dimensions[1]);
	variables.CheckCublasStatus(cublasStatus, "ERR_ENDMATCH");

	return 0;
}

int InputScaling::setLastInput(Variables variables, int color)
{
	cublasStatus_t cublasStatus;
	cudaError_t error;

	const float alpha = 1;
	const float beta = 1 / -0.00572916633f;

	error = cudaMemset(variables.d_InterstageVar[0], 187, 32 * sizeof(float));
	variables.CheckCudaError(error, "ERR_ENDMATCH");
	error = cudaMemset(variables.d_InputStates[variables.h_StateCount], 0, 64 * sizeof(float));
	variables.CheckCudaError(error, "ERR_ENDMATCH");

	cublasStatus = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, variables.h_Dimensions[1] / 2, 1, &alpha, variables.d_InputStates[variables.h_StateCount] + color * 32, variables.h_Dimensions[1] / 2,
		&beta, variables.d_InterstageVar[0], variables.h_Dimensions[1] / 2, variables.d_InputStates[variables.h_StateCount] + (1-color) * 32, variables.h_Dimensions[1] / 2);
	variables.CheckCublasStatus(cublasStatus, "ERR_ENDMATCH");

	return 0;
}