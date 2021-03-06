#pragma once
#include "Variables.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

class Evaluation {
public:
	void KernelSize(int blockDim);
	void addEpochLoss(Variables variables);
	void UpdateEpoch(Variables* variables);
	void GetTrainingStatistics(Variables variabels);

private:
	dim3 KernelSizes;
};

