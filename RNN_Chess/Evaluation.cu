#include "Evaluation.cuh"

void Evaluation::KernelSize(int blockDim)
{
	KernelSizes.x = blockDim;
}

__global__
void CalculateLoss(float* Error_HiddenState, float* TotalError)
{
	__shared__ extern  float sdata[];
	register int dn = threadIdx.x;

	sdata[dn] = Error_HiddenState[dn];

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (dn < s)
		{
			sdata[dn] += sdata[dn + s];
		}
		__syncthreads();
	}

	if (dn == 0) {
		TotalError[0] += Error_HiddenState[0];
	}
}

void Evaluation::addEpochLoss(Variables variables)
{
	CalculateLoss << <1, KernelSizes, KernelSizes.x * sizeof(float) >> > (variables.d_Error_HiddenStates[variables.h_StateCount + 1], variables.d_EvaluationError);
	cudaError_t error = cudaGetLastError();
	variables.CheckCudaError(error, "ERR_EVALUATION");
}

void Evaluation::UpdateEpoch(Variables* variables)
{
	if (variables[0].h_SampleCount == variables[0].h_Dimensions[5])
	{
		cudaError_t error;

		error = cudaMemcpy(&variables[0].h_Loss[variables[0].h_EpochCount], variables[0].d_EvaluationError, sizeof(float), cudaMemcpyDeviceToHost);
		variables[0].CheckCudaError(error, "ERR_EVALUATION");
		error = cudaMemset(variables[0].d_EvaluationError, 0, sizeof(float));
		variables[0].CheckCudaError(error, "ERR_EVALUATION");

		std::cout << "Epoch: " << variables[0].h_EpochCount << "\nLoss: " << variables[0].h_Loss[variables[0].h_EpochCount] << std::endl;

		variables[0].h_SampleCount = 0;
		variables[0].h_EpochCount++;
	}
	variables[0].h_SampleCount++;
}