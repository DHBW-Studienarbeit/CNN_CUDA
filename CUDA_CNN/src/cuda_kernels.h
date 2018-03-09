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

} /* end namespace cuda */


#endif /* CUDA_KERNELS_H_ */
