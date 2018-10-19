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

	sdata[dn] = fabsf(Error_HiddenState[dn]);

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
		TotalError[0] += sdata[0];
	}
}

//Calculates the sum of the Loss for the specific Epoch
void Evaluation::addEpochLoss(Variables variables)
{
	int stackOffset = (variables.h_Dimensions[3] - 1) * variables.h_Dimensions[1];

	CalculateLoss << <1, KernelSizes, KernelSizes.x * sizeof(float) >> > (variables.d_Error_HiddenStates[variables.h_StateCount + 1] + stackOffset, variables.d_EvaluationError);
	cudaError_t error = cudaGetLastError();
	variables.CheckCudaError(error, "ERR_EVALUATION");
}

void Evaluation::UpdateEpoch(Variables* variables)
{
	if (variables[0].h_SampleCount == variables[0].h_Dimensions[5])
	{
		cudaError_t error;

		//Gets the value of the summed Loss and resets the variables
		error = cudaMemcpy(&variables[0].h_Loss[variables[0].h_EpochCount], variables[0].d_EvaluationError, sizeof(float), cudaMemcpyDeviceToHost);
		variables[0].CheckCudaError(error, "ERR_EVALUATION");
		error = cudaMemset(variables[0].d_EvaluationError, 0, sizeof(float));
		variables[0].CheckCudaError(error, "ERR_EVALUATION");

		if (variables[0].h_EpochCount >= 1) {
			if (variables[0].h_Loss[variables[0].h_EpochCount] > variables[0].h_Loss[variables[0].h_EpochCount - 1]) {
				variables[0].LearningRate /= 2;
			}
		}

		std::cout << "Epoch: " << variables[0].h_EpochCount << "\nLoss: " << variables[0].h_Loss[variables[0].h_EpochCount] << std::endl;

		variables[0].h_SampleCount = 0;
		variables[0].h_EpochCount++;
	}
	variables[0].h_SampleCount++;
}

void Evaluation::GetTrainingStatistics(Variables variables)
{
	std::cout << "\n\n=====================";
	std::cout << "Training Evaluation";
	std::cout << "=====================\n\n" << std::endl;

	std::cout << "Loss Reduction:\t\t" << variables.h_Loss[variables.h_EpochCount - 1] - variables.h_Loss[0] << std::endl;
	std::cout << "New Learningrate:\t" << variables.LearningRate << std::endl;	

	float meanLoss = 7 * variables.h_Loss[variables.h_EpochCount - 1] / (variables.h_Dimensions[0] * variables.h_Dimensions[1] * variables.h_Dimensions[3]);
	std::cout << "Average Loss:\t\t" << meanLoss << std::endl;
}