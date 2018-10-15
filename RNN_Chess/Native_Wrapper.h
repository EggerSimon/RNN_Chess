#include "RNN_Chess.cuh"

#define LIBRARY_EXPORT __declspec(dllexport)

extern "C" LIBRARY_EXPORT RNN_Chess* new_RNN_Chess(int Dimensions[]);

extern "C" LIBRARY_EXPORT void delete_RNN(RNN_Chess* instance);

extern "C" LIBRARY_EXPORT int InitializeVariables(RNN_Chess* instance, float** InputWeights, float** RecurrentWeights, float** Biases);
extern "C" LIBRARY_EXPORT int InitializeConstants(RNN_Chess* instance, float learningrate);

extern "C" LIBRARY_EXPORT int UpdateWeightMatrices(RNN_Chess* instance, float** InputWeights, float** RecurrentWeights, float** Biases);
extern "C" LIBRARY_EXPORT void UpdateDimensions(RNN_Chess* instance, int Dimensions[]);

extern "C" LIBRARY_EXPORT float* RunRNN(RNN_Chess* instance, float* InputState);
extern "C" LIBRARY_EXPORT int BackPropagation(RNN_Chess* instance, int color);

extern "C" LIBRARY_EXPORT int FreeWorkSpace(RNN_Chess* instance);