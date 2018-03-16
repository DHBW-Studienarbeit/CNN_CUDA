/*
 * cuda_kernels.cu
 *
 *  Created on: 09.03.2018
 *      Author: benjamin
 */

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>


namespace cuda {

__global__ void init(float** nodeArrayPtrs, int no_node_matrices, int* arrayLengths)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < no_node_matrices; i+=stride)
	{
		curandGenerator_t generator;
		curandStatus_t curand_state;

		curand_state = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

		curand_state = curandSetPseudoRandomGeneratorSeed(gen, (unsigned long) clock());

		curand_state = curandGenerateUniform(generator, nodeArrayPtrs[i], arrayLengths[i]);

		curand_state = curandDestroyGenerator(generator);
	}
}

__global__ void loadPicture(float* arrayPtr, float* picturePtr)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	// TODO: parrallelisation
	/* column major */
	for(int i = 0; i < 24; i++)
	{
		for(int j = 0; j < 24; j++)
		{
			for(int k = 0; k < 5; k++)
			{
				for(int l = 0; l < 5; l++)
				{
					arrayPtr[(k*5+l)*576+i*24+j] = picturePtr[(i+k)*24+j+l];
				}
			}
		}
	}
}

__global__ void convolution(float* inputPtr, float* outputPtr, float* weightPtr,
		int inputDim_x, int inputDim_y, int weightDim_x, int weightDim_y)
{
	cublasStatus_t cublasState;
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	const float alpha = 1.0f;
	const float beta = 0.0f;
	const float* alpha_ptr = &alpha;
	const float* beta_ptr = &beta;

	cublasState = cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, inputDim_x, weightDim_x,
								inputDim_y, alpha_ptr, (const float*) inputPtr, inputDim_x,
								(const float*) weightPtr, weightDim_x, beta_ptr, outputPtr, inputDim_x);

	cublasDestroy(cublasHandle);
}

__global__ void maxPooling(float* inputPtr, float* outputPtr, int x_receptive, int y_receptive,
							int inputDim_x, int inputDim_y, int outputDim_x, int outputDim_y,
							int convDim_x, int convDim_y, int nextDim_x, int nextDim_y,
							int nextReceptive_x, int nextReceptive_y)
{
	int dimSquare = nextDim_x * nextDim_y;
	int pooling_x = convDim_x / x_receptive;
	int pooling_y = convDim_y / y_receptive;
	int poolDim_y = pooling_x * pooling_y;

	float* pooling_mat; /* store pooled values and resort afterwards */
	cudaError_t cuda_error = cudaSuccess;

	cuda_error = cudaMalloc((void**)&pooling_mat, pooling_x*pooling_y*inputDim_y*sizeof(float));


	/* feature maps sorted linear in array because of column major ordering */
	for(int i = 0; i < inputDim_y; i++)
	{
		/* pooling in x and y dimension of a single feature map as a two dimensional array *
		 * of "pixels"
		 */
		for(int j = 0; j < convDim_y; j+=y_receptive)
		{
			for(int k = 0; k < convDim_x; k+=x_receptive)
			{
				float max_val = 0.0f; /* declared here to prevent using same value in different threads */
				int index = 0;
				/* finding max value out of kernel */
				for(int l = 0; l < y_receptive; l++)
				{
					for(int m = 0; m < x_receptive; m++)
					{
						index = i*inputDim_y+(j+*convDim_x+l)+(k*y_receptive+m);
						if(max_val <= inputPtr[index])
						{
							max_val = inputPtr[index];
						}
					}
				}

				pooling_mat[i*poolDim_y+(j/y_receptive)*pooling_y + (k/x_receptive)] =
						max_val;
			}
		}
	}

	/* resort matrix to fit conditions of next layer operation */
	/* column major */
	for(int i = 0; i < nextDim_y; i++)
	{
		for(int j = 0; j < nextDim_x; j++)
		{
			for(int k = 0; k < nextReceptive_y; k++)
			{
				for(int l = 0; l < nextReceptive_x; l++)
				{
					outputPtr[(k*nextReceptive_x+l)*nextDim_y*nextDim_x+i*nextDim_x+j] =
							pooling_mat[(i+k)*nextDim_x+j+l];
				}
			}
		}
	}

	cuda_error |= cudaFree(pooling_mat);
}

} /* end namespace cuda */

