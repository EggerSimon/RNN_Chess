#pragma once
#pragma comment(lib,"cublas.lib")
#include "cublas_v2.h"
#include "cudnn.h"


class CudaErrors {
public:
	static const char* GetCublasStatus(cublasStatus_t cublasStatus);
	static const char* GetCudnnStatus(cudnnStatus_t cudnnStatus);
};