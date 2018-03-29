/*
 * cuda_kernels.h
 *
 *  Created on: 09.03.2018
 *      Author: benjamin
 */

#ifndef CUDA_KERNELS_H_
#define CUDA_KERNELS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include "mathematics.h"

namespace cuda {

__global__ void init(float** nodeArrayPtrs, int no_node_matrices, int* arrayLengths);
__global__ void loadPicture(float* arrayPtr, float* picturePtr);
__global__ void convolution(float* inputPtr, float* outputPtr, float* weightPtr, float* biasPtr,
		int inputDim_x, int inputDim_y, int weightDim_x, int weightDim_y);
__global__ void maxPooling(float* inputPtr, float* outputPtr, int x_receptive, int y_receptive,
		int inputDim_x, int inputDim_y, int convDim_x, int convDim_y,
		int nextDim_x, int nextDim_y, int nextReceptive_x,
		int nextReceptive_y, LAYER_TYPE nextLayerType);
__global__ void fullyConnected(float* inputPtr, float* outputPtr, float* weightPtr, float* biasPtr,
		int inputDim_x, int inputDim_y, int weightDim_x, int weightDim_y);

__device__ float forward(Layer* layer_list, int no_layers, float* labels,
		float** nodeArrayPtrs, float** weightArrayPtrs, float** biasArrayPtrs,
		int no_node_matrices, int no_weight_matrices, int no_bias_matrices, int* nodeMatrixDims_x,
		int* nodeMatrixDims_y, int* weightMatrixDims_x, int* weightMatrixDims_y);

__global__ void train(Layer* layer_list, int no_layers, float* inputPictures, int batch_size, float* labels,
			float** nodeArrayPtrs, float** weightArrayPtrs, float** biasArrayPtrs,
			int no_node_matrices, int no_weight_matrices, int no_bias_matrices, int* nodeMatrixDims_x,
			int* nodeMatrixDims_y, int* weightMatrixDims_x, int* weightMatrixDims_y,
			int *biasMatrixDims_x, int* biasMatrixDims_y);
__global__ void test(Layer* layer_list, int no_layers, float* inputPictures, int batch_size, float* labels,
		float** nodeArrayPtrs, float** weightArrayPtrs, float** biasArrayPtrs,
		int no_node_matrices, int no_weight_matrices, int no_bias_matrices, int* nodeMatrixDims_x,
		int* nodeMatrixDims_y, int* weightMatrixDims_x, int* weightMatrixDims_y,
		int *biasMatrixDims_x, int* biasMatrixDims_y, int* ret_val);
__global__ void gradient_descent(float** weightArrayPtrs, float** biasArrayPtrs,
		int* weightDims_x, int* weightDims_y, int* biasDims_x, int *biasDims_y,
		int no_weights, int batch_size);


/******* backpropagation tasks *********/
__global__ void backpropagate(Layer* layer_list, int no_layers, float* labels,
		float** nodeArrayPtrs, float** weightArrayPtrs, int no_node_matrices, int no_weight_matrices,
		int* nodeMatrixDims_x, int* nodeMatrixDims_y, int* weightMatrixDims_x, int* weightMatrixDims_y);
__device__ void convolution_back(float** nodeArrayPtrs, float** weightArrayPtrs, float** nodeDerivatives, float** weightDerivates,
						int node_index, int weight_index, int prev_nodeDim_x, int weightDim_x, int weightDim_y);
__device__ void fullyConnected_back(float** nodeArrayPtrs, float** weightArrayPtrs, float** nodeDerivatives, float** weightDerivates,
						int node_index, int weight_index);
__device__ void maxPooling_back(float** nodeArrayPtrs, float** weightArrayPtrs, float** nodeDerivatives, float** weightDerivates,
		int node_index, int weight_index,
		int x_receptive, int y_receptive,
		int inputDim_x, int inputDim_y, int convDim_x, int convDim_y,
		int nextDim_x, int nextDim_y, int nextReceptive_x,
		int nextReceptive_y, LAYER_TYPE nextLayerType);

} /* end namespace cuda */

#endif /* CUDA_KERNELS_H_ */
