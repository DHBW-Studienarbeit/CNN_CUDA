/*
 * cuda_kernels.cu
 *
 *  Created on: 09.03.2018
 *      Author: benjamin
 */

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include "mathematics.h"


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

		curand_state = curandSetPseudoRandomGeneratorSeed(generator, (unsigned long long) clock());

		curand_state = curandGenerateUniform(generator, nodeArrayPtrs[i], arrayLengths[i]);

		curand_state = curandDestroyGenerator(generator);
	}
}

/**
 * This function loads a MNIST-Picture into the Network
 * <detail>	Use this function with 80 Threads to enable 150 parallel pictures </detail>
 * <param> float * arrayPtr	-	destination array </param>
 * <param> float* picturePtr	-	source array </param>
 * <return> void </return>
 */
__global__ void loadPicture(float* arrayPtr, float* picturePtr)
{
//	/* column major */
//	for(int i = index; i < 24; i+=stride) /* rows in original row-major picture */
//	{
//		for(int j = 0; j < 24; j++) /* columns in original row-major picture */
//		{
//			for(int k = 0; k < 5; k++) /* convolutional kernel rows in original picture */
//			{
//				for(int l = 0; l < 5; l++) /* convolutional kernel columns in original picture */
//				{
//					arrayPtr[(k*5+l)*576+i*24+j] = picturePtr[(i+k)*24+j+l];
//				}
//			}
//		}
//	}
	int index = threadIdx.x;
	int stride = blockDim.x;

	int size =  576; /*24²*/

	for(int n = index; n < size; n+=stride)
	{
		int i = n/24; /* rows in original row-major picture */
		int j = n%24; /* columns in original row-major picture */

		for(int k = 0; k < 5; k++) /* convolutional kernel rows in original picture */
		{
			for(int l = 0; l < 5; l++) /* convolutional kernel columns in original picture */
			{
				arrayPtr[(k*5+l)*576+i*24+j] = picturePtr[(i+k)*24+j+l];
			}
		}
	}
	__syncthreads();
}

/**
 * This function implements the convolutional layer as a matrix multiplication
 * <detail> The function splits up this matrix operation in single vector dot products.
 * It has to be called with 80 Threads in one block to fit picture parallelism conditions </detail>
 * <param> </param>
 * <return> void </return>
 */
__global__ void convolution(float* inputPtr, float* outputPtr, float* weightPtr, float* biasPtr,
		int inputDim_x, int inputDim_y, int weightDim_x, int weightDim_y)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	cublasStatus_t cublasState = CUBLAS_STATUS_SUCCESS;
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	const float alpha = 1.0f;
	const float beta = 0.0f;
	const float* alpha_ptr = &alpha;
	const float* beta_ptr = &beta;

	int output_size = inputDim_x * weightDim_y;

	for(int i = index; i < weightDim_y; i+= stride) /* parallelize over features (columns of weight matrix) */
	{
		float result = 0.0;
		for(int j = 0; j < inputDim_x; j++) /* perform vector dot product for every point in an output feature */
		{
			/* matrices are ordered as column-major */
			cublasState |= cublasSdot(cublasHandle, weightDim_x,
										(const float*) &inputPtr[j], inputDim_x,
										(const float*) &weightPtr[i], 1,
										&result);
			outputPtr[i*weightDim_y+j] = sigmoid_once(result + biasPtr[i*weightDim_y+j]);
		}
	}
//	cublasState = cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, inputDim_x, weightDim_x,
//								inputDim_y, alpha_ptr, (const float*) inputPtr, inputDim_x,
////								(const float*) weightPtr, weightDim_x, beta_ptr, outputPtr, inputDim_x);
//	/* adding the bias */
//	cublasState |= cublasSaxpy(cublasHandle, inputDim_x*weightDim_y, alpha_ptr,
//			(const float*) biasPtr, 1, (const float*) outputPtr, 1);
//
//	/* activation function */
//	for(int i = 0; i < output_size; i++)
//	{
//		outputPtr[i] = sigmoid_once(outputPtr[i]);
//	}

	cublasDestroy(cublasHandle);
}

/**
 * This function realizes maxPooling after a convolutional layer
 * <detail> The function resorts the output array to fit the conditions of the next layer operation.
 * It has to be called with 80 Threads in one block to fit the 600 pictures parallel condition.
 * It is parallelized over the feature maps. </detail>
 * <param>
 * </param>
 * <return> void </return>
 */
__global__ void maxPooling(float* inputPtr, float* outputPtr, int x_receptive, int y_receptive,
							int inputDim_x, int inputDim_y, int convDim_x, int convDim_y,
							int nextDim_x, int nextDim_y, int nextReceptive_x,
							int nextReceptive_y, LAYER_TYPE nextLayerType)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	int dimSquare = nextDim_x * nextDim_y;
	int pooling_x = convDim_x / x_receptive;
	int pooling_y = convDim_y / y_receptive;
	int poolDim_y = pooling_x * pooling_y;

	float* pooling_mat; /* store pooled values and resort afterwards */
	cudaError_t cuda_error = cudaSuccess;

	cuda_error = cudaMalloc((void**)&pooling_mat, pooling_x*pooling_y*inputDim_y*sizeof(float));


	/* feature maps sorted linear in array because of column major ordering */
	for(int i = index; i < inputDim_y; i+=stride)
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
	if(nextLayerType == CONV_LAYER)
	{
//		for(int i = 0; i < nextDim_x; i++)
//		{
//			for(int j = 0; j < nextDim_y; j++)
//			{
//				for(int k = 0; k < inputDim_y; k++) /* number features of last layer */
//				{
//					for(int l = 0; l < nextReceptive_y; l++) /* convolutional kernel of each feature */
//					{
//						for(int m = 0; m < nextReceptive_x; m++)
//						{
//							/* pooling_mat sorted in column major */
//							outputPtr[(k*nextReceptive_x*nextReceptive_y + l*nextReceptive_x+m)*nextDim_x*nextDim_y+i*nextDim_y+j] =
//									pooling_mat[(k*poolDim_y)+(i+l)*nextDim_x+j+m];
//
//						}
//					}
//				}
//			}
//		}

		int size = nextDim_x*nextDim_y;

		for(int n = index; n < size; n+=stride)
		{
			int i = n/nextDim_y;
			int j = n%nextDim_y;
			for(int k = 0; k < inputDim_y; k++) /* number features of last layer */
			{
				for(int l = 0; l < nextReceptive_y; l++) /* convolutional kernel of each feature */
				{
					for(int m = 0; m < nextReceptive_x; m++)
					{
						/* pooling_mat sorted in column major */
						outputPtr[(k*nextReceptive_x*nextReceptive_y + l*nextReceptive_x+m)*nextDim_x*nextDim_y+i*nextDim_y+j] =
								pooling_mat[(k*poolDim_y)+(i+l)*nextDim_x+j+m];

					}
				}
			}
		}
	}
	else if(nextLayerType == FULLY_CONNECTED_LAYER)
	{
		/* no additional sorting needed
		 * data is already correctly sorted in memory
		 * that way, the user can just switch from a (m*n)x(f)-Matrix to a 1x(m*n*f)-Matrix
		 */
		int size = inputDim_y*poolingDim_y;
		for(int i = index; i < size; i+=stride)
		{
			outputPtr[i] = pooling_mat[i];
		}
//		cuda_error |= cudaMemcpy(outputPtr, pooling_mat, inputDim_y*poolingDim_y*sizeof(float), cudaMemcpyDeviceToDevice);
	}

	cuda_error |= cudaFree(pooling_mat);
}

__global__ void fullyConnected(float* inputPtr, float* outputPtr, float* weightPtr, float* biasPtr,
							int inputDim_x, int inputDim_y, int weightDim_x, int weightDim_y)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	cublasStatus_t cublasState;
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	const float alpha = 1.0f;
	const float beta = 0.0f;
	const float* alpha_ptr = &alpha;
	const float* beta_ptr = &beta;

	int output_size = inputDim_x * weightDim_y;

	for(int i = index; i < weightDim_y; i+= stride) /* parallelize over output nodes (columns of weight matrix) */
	{
		float result = 0.0;
		/* matrices are ordered as column-major */
		cublasState |= cublasSdot(cublasHandle, weightDim_x,
				(const float*) &inputPtr[j], inputDim_x,
				(const float*) &weightPtr[i], 1,
				&result);
		outputPtr[i*weightDim_y+j] = sigmoid_once(result + biasPtr[i*weightDim_y+j]);
	}


//	/* multiplication with weights */
//	cublasState = cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, inputDim_x, weightDim_x,
//			inputDim_y, alpha_ptr, (const float*) inputPtr, inputDim_x,
//			(const float*) weightPtr, weightDim_x, beta_ptr, outputPtr, inputDim_x);
//
//	/* adding the bias */
//	cublasState |= cublasSaxpy(cublasHandle, weightDim_y, alpha_ptr,
//					(const float*) biasPtr, 1, (const float*) outputPtr, 1);
//
//
//	for(int i = 0; i < weightDim_y; i++)
//	{
//		outputPtr[i] = sigmoid_once(outputPtr[i]);
//	}

	cublasDestroy(cublasHandle);
}

/**
 * This function paralellizes the training over a batch of input pictures
 * <details> The number of threads should be equal to the batch_size </details>
 * <params> </params>
 * <return> cost sum over batch </return>
 */
__global__ float train(Layer* layer_list, int no_layers, float* inputPictures, int batch_size, float* labels,
			float** nodeArrayPtrs, float** weightArrayPtrs, float** biasArrayPtrs,
			int no_node_matrices, int no_weight_matrices, int no_bias_matrices, int* nodeMatrixDims_x,
			int* nodeMatrixDims_y, int* weightMatrixDims_x, int* weightMatrixDims_y, int* biasMatrixDims_x,
			int* biasMatrixDims_y)
{
	cudaError_t cuda_error = cudaSuccess;
	int index = threadIdx.x;
	__shared__ float ret_val = 0.0;

	/* copying nodes, weights and biases for each picture-parallel thread to prevent race conditions */
	float** nodeArrays, weightArrays, biasArrays;

	for(int i = 0; i < no_node_matrices; i++)
	{
		cuda_error |= cudaMalloc((void**) &nodeArrays[i], nodeMatrixDims_x[i]*nodeMatrixDims_y[i]*sizeof(float));
		cuda_error |= cudaMemcpy((void*) nodeArrays[i], (void*) nodeArrayPtrs[i], nodeMatrixDims_x[i]*nodeMatrixDims_y[i]*sizeof(float), cudaMemcpyDeviceToDevice);
	}

	for(int i = 0; i < no_weight_matrices; i++)
	{
		cuda_error |= cudaMalloc((void**) &weightArrays[i], weightMatrixDims_x[i]*weightMatrixDims_y[i]*sizeof(float));
		cuda_error |= cudaMemcpy((void*) weightArrays[i], (void*) weightArrayPtrs[i], weightMatrixDims_x[i]*weightMatrixDims_y[i]*sizeof(float), cudaMemcpyDeviceToDevice);
		cuda_error |= cudaMalloc((void**) &biasArrays[i], biasMatrixDims_x[i]*biasMatrixDims_y[i]*sizeof(float));
		cuda_error |= cudaMemcpy((void*) biasArrays[i], (void*) biasArrayPtrs[i], biasMatrixDims_x[i]*biasMatrixDims_y[i]*sizeof(float), cudaMemcpyDeviceToDevice);
	}

	if(index < batch_size)
	{
		loadPicture<<1,80>>(nodeArrays[0], &inputPictures[index*784]);
		/* blocks until all threads finish */

		/* forward splits up into 80 parallel threads in each layer task */
		ret_val += forward<<1,1>>(layer_list, no_layers, &labels[index*10], nodeArrays, weightArrays, biasArrays);

		__syncthreads();
	}

	for(int i = 0; i < no_node_matrices; i++)
	{
		cuda_error |= cudaFree((void*)nodeArrays[i]);
	}

	for(int i = 0; i < no_weight_matrices; i++)
	{
		cuda_error |= cudaFree((void*) weightArrays[i]);
		cuda_error |= cudaFree((void*) biasArrays[i]);
	}

	return ret_val;
}


__global__ void gradient_descent(float cost, float** weightArrayPtrs, float** biasArrayPtrs, int no_weights)
{

}

/**
 * This function is the parallel implementation of forwarding in the CNN
 *
 * <params> </params>
 * <return> cost of this input </return>
 */
__global__ float forward(Layer* layer_list, int no_layers, float* labels,
					float** nodeArrayPtrs, float** weightArrayPtrs, float** biasArrayPtrs,
					int no_node_matrices, int no_weight_matrices, int no_bias_matrices, int* nodeMatrixDims_x,
					int* nodeMatrixDims_y, int* weightMatrixDims_x, int* weightMatrixDims_y)
{

	/* Indices to iterate through weight_list, node_list and bias_list */
		int weight_index = 0;
		int bias_index = 0;
		int node_index = 0;

		for(int i = 0; i < no_layers; i++)
		{
			switch (layertype_list[i])
			{
				case INPUT_LAYER:
				{
					/* nothing to do, picture already loaded */
					node_index++;
					break;
				}
				case CONV_LAYER:
				{
					cuda::convolution<<1,80>>(nodeArrayPtrs[node_index-1], nodeArrayPtrs[node_index], weightArrayPtrs[weight_index],
							biasArrayPtrs[bias_index], nodeMatrixDims_x[node_index-1], nodeMatrixDims_y[node_index-1], weightMatrixDims_x[weight_index],
							weightMatrixDims_y[weight_index]);
					weight_index++;
					bias_index++;
					node_index++;

					break;
				}
				case POOLING_LAYER:
				{
					MaxPooling_Layer* pooling_layer = (MaxPooling_Layer*) layer_list[i];
					Conv_Layer* last_layer = (Conv_Layer*) layer_list[i-1];

					int x_receptive = pooling_layer->getXReceptive();
					int y_receptive = pooling_layer->getYReceptive();
					int convDim_x = last_layer->getXSize();
					int convDim_y = last_layer->getYSize();

					if(layertype_list[i+1] == CONV_LAYER)
					{
						Conv_Layer* next_layer = (Conv_Layer*) layer_list[i+1];
						int nextDim_x = next_layer->getXSize();
						int nextDim_y = next_layer->getYSize();
						int nextReceptive_x = next_layer->getXReceptive();
						int nextReceptive_y = next_layer->getYReceptive();

						cuda::maxPooling<<1,80>>(nodeArrayPtrs[node_index-1], nodeArrayPtrs[node_index], x_receptive,
													y_receptive, nodeMatrixDims_x[node_index-1], nodeMatrixDims_y[node_index-1],
													nextDim_x, nextDim_y, nextReceptive_x, nextReceptive_y, CONV_LAYER);


					}
					else if(layertype_list[i+1] == FULLY_CONNECTED_LAYER)
					{
						int nextDim_x = 0;
						int nextDim_y = 0;
						int nextReceptive_x = 0;
						int nextReceptive_y = 0;

						cuda::maxPooling<<1,80>>(nodeArrayPtrs[node_index-1], nodeArrayPtrs[node_index], x_receptive,
																		y_receptive, nodeMatrixDims_x[node_index-1], nodeMatrixDims_y[node_index-1],
																		nextDim_x, nextDim_y, nextReceptive_x, nextReceptive_y, FULLY_CONNECTED_LAYER);
					}
					else
					{
						return -1.0f;
					}

					node_index++;

					break;
				}
				case FULLY_CONNECTED_LAYER:
				{
					cuda::fullyConnected<<1,80>>(nodeArrayPtrs[node_index-1], nodeArrayPtrs[node_index],
							weightArrayPtrs[weight_index], biasArrayPtrs[bias_index], nodeMatrixDims_x[node_index-1], nodeMatrixDims_y[node_index-1],
							weightMatrixDims_x[weight_index], weightMatrixDims_y[weight_index]);


					node_index++;
					weight_index++;
					bias_index++;

					break;
				}
				case DROPOUT_LAYER:
				{
					break;
				}
				default:
				{
					return -1.0f;
				}
			}
		}

		return mathematics::get_cost(nodeArrayPtrs[node_index-1], labels, OUTPUT_SIZE);
}

} /* end namespace cuda */

