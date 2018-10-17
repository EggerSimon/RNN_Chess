#pragma once

#include "Variables.h"
#include <iostream>

class GateCalculations {
public:
	int GateCalculation(float* d_Gate, int counter, int stackCount, char* description, Variables variables);
	int UpdateGates(float** GateError, int stackCount, int gateCount, int stateCount, char* description, Variables variables);
	int BackwardPass(float** d_Gate, int stackCount, int gateCount, char* description, Variables variables);
};
