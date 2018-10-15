#pragma once
#include "cudnn.h"
#include "cublas_v2.h"
#pragma comment(lib,"cublas.lib")

class Variables
{
public:	
	int AllocateWorkspace(int Dimensions[]);
	int InitializeVariables(float** InputWeights, float** RecurrentWeights, float** Biases);
	int InitializeConstants(float learningrate);
	int UpdateWeightMatrices(float** InputWeights, float** RecurrentWeights, float** Biases);
	const float getLearningRate();
	int GetResults();
	int FreeWorkspace();

	void CheckCudaError(cudaError_t error, char* description);
	void CheckCublasStatus(cublasStatus_t error, char* description);
	void CheckCudnnStatus(cudnnStatus_t error, char* description);


	//Host variables
	float* h_Results;
	int* h_Dimensions;
	int h_StateCount;

	//Workspace variables
	float** d_CellStates;
	float** d_InputStates;
	float** d_HiddenStates;
	float** d_InputWeights;
	float** d_RecurrentWeights;
	float** d_Biases;

	//Workspace Gates
	float** d_ForgetGate;
	float** d_InputGate;
	float** d_OutputGate;
	float** d_CellGate;

	//Workspace for BackPropagation
	float** d_Error_CellStates;
	float** d_Error_HiddenStates;
	float** d_Error_ForgetGate;
	float** d_Error_InputGate;
	float** d_Error_OutputGate;
	float** d_Error_CellGate;

	//Cudnn Variables
	cudnnHandle_t cudnn;
	cudnnTensorDescriptor_t tensor_descriptor;
	cudnnActivationDescriptor_t* activation_descriptor;

	//Workspace interstage variables
	float** d_InterstageVar;

	//Workspace constants
	float LearningRate;

	//Cublas
	cublasHandle_t cublas;
};
