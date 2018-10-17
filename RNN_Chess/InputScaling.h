#pragma once
#include "Variables.h"
#include <iostream>

class InputScaling {
public:
	int ScaleInput(Variables variables);
	int setLastInput(Variables variables, int color);
private:
	Variables variables;
};