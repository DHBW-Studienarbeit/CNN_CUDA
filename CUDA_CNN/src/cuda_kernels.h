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
void convolution(float* inputPtr, float* outputPtr, float* weightPtr,
		int inputDim_x, int inputDim_y, int weightDim_x, int weightDim_y);
void maxPooling(float* inputPtr, float* outputPtr, int x_receptive, int y_receptive,
		int inputDim_x, int inputDim_y, int outputDim_x, int outputDim_y,
		int convDim_x, int convDim_y, int nextDim_x, int nextDim_y,
		int nextReceptive_x, int nextReceptive_y, LAYER_TYPE nextLayerType);
void fullyConnected(float* inputPtr, float* outputPtr, float* weightPtr,
		int inputDim_x, int inputDim_y, int weightDim_x, int weightDim_y);

} /* end namespace cuda */

/* column major */
//for(int i = 0; i < 24; i++) /* rows in original row-major picture */
//{
//	for(int j = 0; j < 24; j++) /* columns in original row-major picture */
//	{
//		for(int k = 0; k < 5; k++) /* convolutional kernel rows in original picture */
//		{
//			for(int l = 0; l < 5; l++) /* convolutional kernel columns in original picture */
//			{
//				arrayPtr[(k*5+l)*576+i*24+j] = picturePtr[(i+k)*24+j+l];
//			}
//		}
//	}
//}
#endif /* CUDA_KERNELS_H_ */
