#pragma once
#include "Variables.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Evaluation {
public:
	void KernelSize(int blockDim);
	void addEpochLoss(Variables variables, int stackOffset);
	void UpdateEpoch(Variables variables);

private:
	dim3 KernelSizes;
};

