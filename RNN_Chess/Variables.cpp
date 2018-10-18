#include "Variables.h"

/*Dimensions*/
//0: number of steps
//1: number of hidden features
//2: number of input features
//3: number of stacked LSTM blocks

CudaErrors cudaErrors;

//Check Cuda error
void Variables::CheckCudaError(cudaError_t error, char* description)
{
	if (error != cudaSuccess) {
		std::cout << h_StateCount << " " << description << ": ";
		fprintf(stderr, cudaGetErrorString(error));
		std::cout << std::endl;
	}
}

//Checks Cublas opperations for errors
void Variables::CheckCublasStatus(cublasStatus_t error, char* description)
{
	if (error != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << h_StateCount << " " << description << ": ";
		fprintf(stderr, cudaErrors.GetCublasStatus(error));
		std::cout << std::endl;
	}
}

//Checks Cudnn opperations for errors
void Variables::CheckCudnnStatus(cudnnStatus_t error, char* description)
{
	if (error != CUDNN_STATUS_SUCCESS) {
		std::cout << h_StateCount << " " << description << ": ";
		fprintf(stderr, cudaErrors.GetCudnnStatus(error));
		std::cout << std::endl;
	}
}

int Variables::AllocateWorkspace(int Dimensions[])
{
	cudaError_t error;
	cublasStatus_t cublasStatus;
	cudnnStatus_t cudnnStatus;

	h_EpochCount = 0;
	h_StateCount = 0;
	h_SampleCount = 0;
	h_Dimensions = Dimensions;
	h_Results = new float[Dimensions[1]];
	h_Loss = new float[Dimensions[4]];
	h_Accuracy = new float[Dimensions[4]];

	d_CellStates = new float*[Dimensions[0] + 1];
	d_InputStates = new float*[Dimensions[0] + 1];
	d_HiddenStates = new float*[Dimensions[0] + 1];
	d_ForgetGate = new float*[Dimensions[0]];
	d_InputGate = new float*[Dimensions[0]];
	d_OutputGate = new float*[Dimensions[0]];
	d_CellGate = new float*[Dimensions[0]];
	d_Error_CellStates = new float*[Dimensions[0] + 1];
	d_Error_HiddenStates = new float*[Dimensions[0] + 1];
	d_Error_ForgetGate = new float*[Dimensions[0]];
	d_Error_InputGate = new float*[Dimensions[0]];
	d_Error_OutputGate = new float*[Dimensions[0]];
	d_Error_CellGate = new float*[Dimensions[0]];
	d_Biases = new float*[4];
	d_InputWeights = new float*[4];
	d_RecurrentWeights = new float*[4];
	activation_descriptor = new cudnnActivationDescriptor_t[4];

	//Allocation of first Cell and Hidden states
	error = cudaMalloc((void **)&d_CellStates[0], Dimensions[3] * Dimensions[1] * sizeof(float));
	CheckCudaError(error, "ERR_VAR_MALLOC (CellStates0)");
	error = cudaMalloc((void **)&d_InputStates[0], Dimensions[2] * sizeof(float));
	CheckCudaError(error, "ERR_VAR_MALLOC (InputStates0)");
	error = cudaMalloc((void **)&d_HiddenStates[0], Dimensions[3] * Dimensions[1] * sizeof(float));
	CheckCudaError(error, "ERR_VAR_MALLOC (HiddenStates0)");

	error = cudaMalloc((void **)&d_Error_CellStates[0], Dimensions[3] * Dimensions[1] * sizeof(float));
	CheckCudaError(error, "ERR_VAR_MALLOC (Error_CellStates)");
	error = cudaMalloc((void **)&d_Error_HiddenStates[0], Dimensions[3] * Dimensions[1] * sizeof(float));
	CheckCudaError(error, "ERR_VAR_MALLOC (Error_HiddenStates)");

	//Allocation of Evaluation Error
	error = cudaMalloc((void **)&d_EvaluationError, sizeof(float));
	CheckCudaError(error, "ERR_VAR_MALLOC (EvaluationError)");

	//Memory Allocation for the different States
	for (int i = 0; i < Dimensions[0]; i++)
	{
		//Allocation of different States
		error = cudaMalloc((void **)&d_CellStates[i + 1], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (CellStates)");
		error = cudaMalloc((void **)&d_InputStates[i + 1], Dimensions[2] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (InputStates)");
		error = cudaMalloc((void **)&d_HiddenStates[i + 1], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (HiddenStates)");

		//Allocation of the different Gates
		error = cudaMalloc((void **)&d_ForgetGate[i], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (ForgetGate)");
		error = cudaMalloc((void **)&d_InputGate[i], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (InputGate)");
		error = cudaMalloc((void **)&d_OutputGate[i], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (OutputGate)");
		error = cudaMalloc((void **)&d_CellGate[i], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (CellGate)");

		//Allocation of the BackPropagation Errors
		error = cudaMalloc((void **)&d_Error_CellStates[i + 1], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (Error_CellStates)");
		error = cudaMalloc((void **)&d_Error_HiddenStates[i + 1], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (Error_HiddenStates)");
		error = cudaMalloc((void **)&d_Error_ForgetGate[i], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (Error_ForgetGate)");
		error = cudaMalloc((void **)&d_Error_InputGate[i], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (Error_InputGate)");
		error = cudaMalloc((void **)&d_Error_OutputGate[i], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (Error_OutputGate)");
		error = cudaMalloc((void **)&d_Error_CellGate[i], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (Error_CellGate)");
	}

	//Memset of default hidden and cellstate
	error = cudaMemset(d_CellStates[0], 0, Dimensions[3] * Dimensions[1]);
	CheckCudaError(error, "ERR_VAR_MEMSET (CellState)");
	error = cudaMemset(d_HiddenStates[0], 0, Dimensions[3] * Dimensions[1]);
	CheckCudaError(error, "ERR_VAR_MEMSET (HiddenState)");

	//Cublas and Cudnn setup
	cublasStatus = cublasCreate(&cublas);
	CheckCublasStatus(cublasStatus, "ERR_CUBLAS_CREATE");
	cudnnStatus = cudnnCreate(&cudnn);
	CheckCudnnStatus(cudnnStatus, "ERR_CUDNN_CREATE (Handler)");
	cudnnStatus = cudnnCreateTensorDescriptor(&tensor_descriptor);
	CheckCudnnStatus(cudnnStatus, "ERR_CUDNN_CREATE (Tensor)");
	cudnnStatus = cudnnSetTensor4dDescriptor(tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Dimensions[1], 1, 1);
	CheckCudnnStatus(cudnnStatus, "ERR_CUDNN_SET (Tensor)");
	cudnnActivationMode_t* activation_mode = new cudnnActivationMode_t[4]{ CUDNN_ACTIVATION_SIGMOID,CUDNN_ACTIVATION_SIGMOID,CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_TANH };

	//Memory Allocation for the Weights for each Gate
	for (int i = 0; i < 4; i++)
	{
		error = cudaMalloc((void **)&d_InputWeights[i], Dimensions[3] * Dimensions[2] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (InputWeights)");
		error = cudaMalloc((void **)&d_RecurrentWeights[i], Dimensions[3] * pow(Dimensions[1], 2) * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (RecurrentWeights)");
		error = cudaMalloc((void **)&d_Biases[i], Dimensions[3] * Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (Biases)");

		//Cudnn Activation Handler
		cudnnStatus = cudnnCreateActivationDescriptor(&activation_descriptor[i]);
		CheckCudnnStatus(cudnnStatus, "ERR_CUDNN_CREATE (Activation)");
		cudnnStatus = cudnnSetActivationDescriptor(activation_descriptor[i], activation_mode[i], CUDNN_NOT_PROPAGATE_NAN, 0);
		CheckCudnnStatus(cudnnStatus, "ERR_CUDNN_SET (Activation)");
	}

	//Allocation of the interstage variables
	d_InterstageVar = new float*[3];

	for (int i = 0; i < 3; i++)
	{
		error = cudaMalloc((void **)&d_InterstageVar[i], Dimensions[1] * sizeof(float));
		CheckCudaError(error, "ERR_VAR_MALLOC (Interstage)");
	}

	return 0;
}

int Variables::InitializeVariables(float** InputWeights, float** RecurrentWeights, float** Biases)
{
	cudaError_t error;

	for (int i = 0; i < 4; i++)
	{
		error = cudaMemcpy(d_InputWeights[i], InputWeights[i], h_Dimensions[3] * h_Dimensions[1] * h_Dimensions[2] * sizeof(float), cudaMemcpyHostToDevice);
		CheckCudaError(error, "ERR_VAR_INIT (InputWeights)");
		error = cudaMemcpy(d_RecurrentWeights[i], RecurrentWeights[i], h_Dimensions[3] * pow(h_Dimensions[1], 2) * sizeof(float), cudaMemcpyHostToDevice);
		CheckCudaError(error, "ERR_VAR_INIT (RecurrentWeights)");
		error = cudaMemcpy(d_Biases[i], Biases[i], h_Dimensions[3] * h_Dimensions[1] * sizeof(float), cudaMemcpyHostToDevice);
		CheckCudaError(error, "ERR_VAR_INIT (Biases)");
	}

	return 0;
}

int Variables::InitializeConstants(float learningrate)
{
	LearningRate = learningrate;

	return 0;
}

int Variables::UpdateWeightMatrices(float** InputWeights, float** RecurrentWeights, float** Biases)
{
	cudaError_t error;

	for (int i = 0; i < 4; i++)
	{
		error = cudaMemcpy(InputWeights[i], d_InputWeights[i], h_Dimensions[3] * h_Dimensions[2] * h_Dimensions[1] * sizeof(float), cudaMemcpyDeviceToHost);
		CheckCudaError(error, "ERR_VAR_UPDATE (InputWeights)");
		error = cudaMemcpy(RecurrentWeights[i], d_RecurrentWeights[i], h_Dimensions[3] * pow(h_Dimensions[1], 2) * sizeof(float), cudaMemcpyDeviceToHost);
		CheckCudaError(error, "ERR_VAR_UPDATE (RecurrentWeights)");
		error = cudaMemcpy(Biases[i], d_Biases[i], h_Dimensions[3] * h_Dimensions[1] * sizeof(float), cudaMemcpyDeviceToHost);
		CheckCudaError(error, "ERR_VAR_UPDATE (Biases)");
	}

	return 0;
}

int Variables::GetResults()
{
	cudaError_t error;
	int outputOffset = (h_Dimensions[3] - 1) * h_Dimensions[1];
	error = cudaMemcpy(h_Results, d_HiddenStates[h_StateCount + 1] + outputOffset, h_Dimensions[1] * sizeof(float), cudaMemcpyDeviceToHost);
	CheckCudaError(error, "ERR_VAR_RETURN (Results)");

	return 0;
}

const float Variables::getLearningRate()
{
	return LearningRate;
}

//Frees used Variables and destroys cudnn/cublas descriptions and handlers
int Variables::FreeWorkspace()
{
	cudaError_t error;
	cudnnStatus_t cudnnStatus;
	cublasStatus_t cublasStatus;

	free(h_Results);

	cudnnStatus = cudnnDestroyTensorDescriptor(tensor_descriptor);
	CheckCudnnStatus(cudnnStatus, "ERR_CUDNN_DESTROY (Tensor)");

	for (int i = 0; i < h_Dimensions[0]; i++)
	{
		error = cudaFree(d_InputStates[i]);
		CheckCudaError(error, "ERR_VAR_FREE (InputStates)");
		error = cudaFree(d_HiddenStates[i]);
		CheckCudaError(error, "ERR_VAR_FREE (HiddenStates)");
	}

	for (int i = 0; i < 4; i++)
	{
		error = cudaFree(d_InputWeights[i]);
		CheckCudaError(error, "ERR_VAR_FREE (InputWeights)");
		error = cudaFree(d_RecurrentWeights[i]);
		CheckCudaError(error, "ERR_VAR_FREE (RecurrentWeights)");
		error = cudaFree(d_Biases[i]);
		CheckCudaError(error, "ERR_VAR_FREE (Biases)");

		cudnnStatus = cudnnDestroyActivationDescriptor(activation_descriptor[i]);
		CheckCudnnStatus(cudnnStatus, "ERR_CUDNN_DESTROY (Activation)");
	}

	cudnnStatus = cudnnDestroy(cudnn);
	CheckCudnnStatus(cudnnStatus, "ERR_CUDNN_DESTROY (Handler)");
	cublasStatus = cublasDestroy(cublas);
	CheckCublasStatus(cublasStatus, "ERR_CUBLAS_DESTROY (Handler)");

	return 0;
}

