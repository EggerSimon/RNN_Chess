#include "Evaluation.h"

void Evaluation::KernelSize(int blockDim) {
	KernelSizes.x = blockDim;
}

__global__
void CalculateLoss(float* Error_HiddenState, float TotalError)
{
	extern __shared__ float* sdata;
	register int dn = threadIdx.x;

	sdata[dn] = Error_HiddenState[dn];

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (dn < s) {
			sdata[dn] += sdata[dn + s];
		}
		__syncthreads();
	}

	if (dn == 0) {
		TotalError += Error_HiddenState[0];
	}
}

void Evaluation::addEpochLoss(Variables variables, int stackOffset)
{
	CalculateLoss << <1, KernelSizes, KernelSizes.x * sizeof(float) >> > (variables.d_Error_HiddenState[variables.h_StateCount + 1] + stackOffset, &variables.d_EvaluationError);
	cudaError_t error = cudaGetLastError();
	variables.CheckCudaError(error, "ERR_EVALUATION");
}

void Evaluation::UpdateEpoch(Variables variables) 
{
	variables.h_SampleCount++;
	if (variables.h_SampleCount > variables.h_Dimensions[5]) {
		cudaError_t error;
		
		error = cudaMemcpy(&variables.h_Loss[variables.h_EpochCount], variables.d_EvaluationError, sizeof(float), cudaMemcpyDeviceToHost);
		variables.CheckCudaError(error, "ERR_EVALUATION");
		error = cudaMemset(variables.d_EvaluationError, 0, sizeof(float));
		variables.CheckCudaError(error, "ERR_EVALUATION");

		variables.h_SampleCount = 0;
		variables.h_EpochCount++;
	}
}