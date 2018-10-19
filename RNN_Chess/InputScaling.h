#pragma once
#include "Variables.h"

class InputScaling {
public:
	int ScaleInput(Variables variables);
	int setLastInput(Variables variables, int color);
private:
	Variables variables;
};