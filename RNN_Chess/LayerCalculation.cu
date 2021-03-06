#include "LayerCalculation.cuh"


void LayerCalculation::KernelCalculation(Variables variables)
{
	//calculates the dimensions of the kernels lauched later on
	KernelSizes = new dim3[2];
	int size = variables.h_Dimensions[1] * 2;
	//blockDim.x
	KernelSizes[1].x = 256;
	//gridDim.x
	KernelSizes[0].x = ceil((float)size / KernelSizes[1].x);
}

int LayerCalculation::InitializeVariables(Variables variables)
{
	KernelCalculation(variables);

	return 0;
}

__global__
void StateCalculator(float* InputCellState, float* CellState, float* InputHiddenState, float* HiddenState, float* ForgetGate, float* InputGate,
	float* OutputGate, float* CellGate, int HiddenDim)
{
	__shared__ extern float sdata[];
	register int dn = blockIdx.x * blockDim.x + threadIdx.x;
	register int pos = floorf(dn / 2);

	if (pos < HiddenDim)
	{
		if (dn == pos * 2)
		{
			sdata[dn] = InputCellState[pos] * ForgetGate[pos];
		}
		else
		{
			sdata[dn] = InputGate[pos] * CellGate[pos];
		}

		__syncthreads();

		if (dn == pos * 2)
		{
			CellState[pos] = sdata[dn] + sdata[dn + 1];
			HiddenState[pos] = tanhf(CellState[pos]) * OutputGate[pos];
		}
	}
}

int LayerCalculation::StateCalculation(int stackCount, Variables variables)
{
	int StateCount = variables.h_StateCount;
	int stackOffset = stackCount * variables.h_Dimensions[1];

	//calculates values for each gate and stack
	StateCalculator << <KernelSizes[0], KernelSizes[1], KernelSizes[1].x * sizeof(float) >> > (variables.d_CellStates[StateCount] + stackOffset,
		variables.d_CellStates[StateCount + 1] + stackOffset, variables.d_HiddenStates[StateCount] + stackOffset, variables.d_HiddenStates[StateCount + 1] + stackOffset,
		variables.d_ForgetGate[StateCount] + stackOffset, variables.d_InputGate[StateCount] + stackOffset, variables.d_OutputGate[StateCount] + stackOffset,
		variables.d_CellGate[StateCount] + stackOffset, variables.h_Dimensions[1]);

	cudaError_t error = cudaGetLastError();
	variables.CheckCudaError(error, "ERR_STATE_CALCULATION");

	return 0;
}

__global__
void CalculateStateError(float* Error_HiddenState, float* HiddenState, float* Target)
{
	register int dn = blockIdx.x * blockDim.x + threadIdx.x;
	Error_HiddenState[dn] = HiddenState[dn] - Target[dn];
}

int LayerCalculation::GetStateError(Variables variables, Evaluation evaluation)
{
	//calculates loss of the predictions
	KernelSizes[0].x = 1;
	KernelSizes[1].x = variables.h_Dimensions[1];

	int stackCount = (variables.h_Dimensions[3] - 1) * variables.h_Dimensions[1];

	CalculateStateError << <KernelSizes[0], KernelSizes[1] >> > (variables.d_Error_HiddenStates[variables.h_StateCount + 1] + stackCount,
		variables.d_HiddenStates[variables.h_StateCount + 1] + stackCount, variables.d_InputStates[variables.h_StateCount + 1]);

	cudaError_t error = cudaGetLastError();
	variables.CheckCudaError(error, "ERR_STATE_ERRORCALCULATION");

	evaluation.addEpochLoss(variables);

	return 0;
}

__global__
void UpdateGateErrors(float* Error_HiddenState, float* Error_CellState, float* Last_Error_CellState, float* Error_CellGate, float* Error_ForgetGate,
	float* Error_InputGate, float* Error_OutputGate, float* CellState, float* LastCellState, float* CellGate, float* ForgetGate, float* InputGate, float* OutputGate)
{
	register int dn = blockIdx.x * blockDim.x + threadIdx.x;
	register int index = floorf(dn / 3.0f);
	register int count = dn - index * 3;

	register float tempError = Error_HiddenState[index] * OutputGate[index] * (1 - __powf(fabsf(tanhf(CellState[index])), 2)) + Last_Error_CellState[index];

	if (count == 0)
	{
		Error_OutputGate[index] = Error_HiddenState[index] * tanhf(CellState[index]) * OutputGate[index] * (1 - OutputGate[index]);
	}
	if (count == 1)
	{
		Error_CellGate[index] = tempError * InputGate[index] * (1 - __powf(fabsf(CellGate[index]), 2));
		Error_InputGate[index] = tempError * CellGate[index] * InputGate[index] * (1 - InputGate[index]);
	}
	if (count == 2)
	{
		Error_CellState[index] = tempError * ForgetGate[index];
		Error_ForgetGate[index] = tempError * LastCellState[index] * ForgetGate[index] * (1 - ForgetGate[index]);
	}
}

int LayerCalculation::UpdateGates(int stackCount, Variables variables)
{
	//passes the before calculated gradients backwards to each gate
	int count = variables.h_StateCount;
	int stackOffset = stackCount * variables.h_Dimensions[1];

	KernelSizes[0].x = 1;
	KernelSizes[1].x = variables.h_Dimensions[1] * 3;

	UpdateGateErrors << <KernelSizes[0], KernelSizes[1] >> > (variables.d_Error_HiddenStates[count + 1] + stackOffset, variables.d_Error_CellStates[count] + stackOffset,
		variables.d_Error_CellStates[count + 1] + stackOffset, variables.d_Error_CellGate[count] + stackOffset, variables.d_Error_ForgetGate[count] + stackOffset,
		variables.d_Error_InputGate[count] + stackOffset, variables.d_Error_OutputGate[count] + stackOffset, variables.d_CellStates[count + 1] + stackOffset,
		variables.d_CellStates[count] + stackOffset, variables.d_CellGate[count] + stackOffset, variables.d_ForgetGate[count] + stackOffset, variables.d_InputGate[count] + stackOffset,
		variables.d_OutputGate[count] + stackOffset);

	cudaError_t error = cudaGetLastError();
	variables.CheckCudaError(error, "ERR_GATE_ERRORCALCULATION");

	return 0;
}