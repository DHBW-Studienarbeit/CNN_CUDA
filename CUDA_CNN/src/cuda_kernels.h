/*
 * cuda_kernels.h
 *
 *  Created on: 09.03.2018
 *      Author: benjamin
 */

#ifndef CUDA_KERNELS_H_
#define CUDA_KERNELS_H_


namespace cuda {

void init(float** nodeArrayPtrs, int no_node_matrices, int* arrayLengths);
void loadPicture(float* arrayPtr, float* picturePtr);
void convolution(float* inputPtr, float* outputPtr, float* weightPtr, float* biasPtr,
		int inputDim_x, int inputDim_y, int weightDim_x, int weightDim_y);
void maxPooling(float* inputPtr, float* outputPtr, int x_receptive, int y_receptive,
		int inputDim_x, int inputDim_y, int convDim_x, int convDim_y,
		int nextDim_x, int nextDim_y, int nextReceptive_x,
		int nextReceptive_y, LAYER_TYPE nextLayerType);
void fullyConnected(float* inputPtr, float* outputPtr, float* weightPtr, float* biasPtr,
		int inputDim_x, int inputDim_y, int weightDim_x, int weightDim_y);

float forward(Layer* layer_list, int no_layers, float* labels,
		float** nodeArrayPtrs, float** weightArrayPtrs, float** biasArrayPtrs,
		int no_node_matrices, int no_weight_matrices, int no_bias_matrices, int* nodeMatrixDims_x,
		int* nodeMatrixDims_y, int* weightMatrixDims_x, int* weightMatrixDims_y);

float train(Layer* layer_list, int no_layers, float* inputPictures, int batch_size, float* labels,
			float** nodeArrayPtrs, float** weightArrayPtrs, float** biasArrayPtrs,
			int no_node_matrices, int no_weight_matrices, int no_bias_matrices, int* nodeMatrixDims_x,
			int* nodeMatrixDims_y, int* weightMatrixDims_x, int* weightMatrixDims_y);
void gradient_descent(float** weightArrayPtrs, float** biasArrayPtrs,
		int* weightDims_x, int* weightDims_y, int* biasDims_x, int *biasDims_y,
		int no_weights, int batch_size);


/******* backpropagation tasks *********/
void backpropagate(Layer* layer_list, int no_layers, float* labels,
		float** nodeArrayPtrs, float** weightArrayPtrs, int no_node_matrices, int no_weight_matrices,
		int* nodeMatrixDims_x, int* nodeMatrixDims_y, int* weightMatrixDims_x, int* weightMatrixDims_y);
void convolution_back(float** nodeArrayPtrs, float** weightArrayPtrs, float** nodeDerivatives, float** weightDerivates,
						int node_index, int weight_index, int prev_nodeDim_x, int weightDim_x, int weightDim_y);
void fullyConnected_back(float** nodeArrayPtrs, float** weightArrayPtrs, float** nodeDerivatives, float** weightDerivates,
						int node_index, int weight_index);
void maxPooling_back(float** nodeArrayPtrs, float** weightArrayPtrs, float** nodeDerivatives, float** weightDerivates,
						int node_index, int weight_index);

} /* end namespace cuda */

#endif /* CUDA_KERNELS_H_ */
