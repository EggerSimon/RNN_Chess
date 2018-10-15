#pragma once
#include "Variables.h"

class LayerCalculation {
public:
	int InitializeVariables(Variables var);
	int StateCalculation(int stackCount, Variables variables);

	int GetStateError(int color, Variables variables);
	int UpdateGates(int stackCount, Variables variables);


	dim3* KernelSizes;

private:
	void KernelCalculation(Variables variables);
};