#include "Native_Wrapper.h"

extern "C" LIBRARY_EXPORT RNN_Chess* new_RNN_Chess(int Dimensions[])
{
	return new RNN_Chess(Dimensions);
}

extern "C" LIBRARY_EXPORT void delete_RNN(RNN_Chess* instance)
{
	delete instance;
}

extern "C" LIBRARY_EXPORT int InitializeVariables(RNN_Chess* instance, float** InputWeights, float** RecurrentWeights, float** Biases)
{
	return instance->InitializeVariables(InputWeights, RecurrentWeights, Biases);
}

extern "C" LIBRARY_EXPORT int InitializeConstants(RNN_Chess* instance, float learningrate)
{
	return instance->InitializeConstants(learningrate);
}

extern "C" LIBRARY_EXPORT int UpdateWeightMatrices(RNN_Chess* instance, float** InputWeights, float** RecurrentWeights, float** Biases)
{
	return instance->UpdateWeightMatrices(InputWeights, RecurrentWeights, Biases);
}

extern "C" LIBRARY_EXPORT void UpdateDimensions(RNN_Chess* instance, int Dimensions[])
{
	return instance->UpdateDimensions(Dimensions);
}

extern "C" LIBRARY_EXPORT float* RunRNN(RNN_Chess* instance, float* InputState)
{
	return instance->RunRNN(InputState);
}


extern "C" LIBRARY_EXPORT int ErrorCalculation(RNN_Chess* instance, int color)
{
	return instance->ErrorCalculation(color);
}

extern "C" LIBRARY_EXPORT int BackPropagation(RNN_Chess* instance)
{
	return instance->BackPropagation();
}

extern "C" LIBRARY_EXPORT int FreeWorkSpace(RNN_Chess* instance)
{
	return instance->FreeWorkSpace();
}