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

float forward(Layer* layer_list, int no_layers, float* inputPicture, float* labels,
			float** nodeArrayPtrs, float** weightArrayPtrs, float** biasArrayPtrs);


} /* end namespace cuda */

#endif /* CUDA_KERNELS_H_ */
