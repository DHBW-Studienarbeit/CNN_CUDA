/*
 * mathematics.h
 *
 *  Created on: 05.12.2017
 *      Author: Florian
 */

#ifndef MATHEMATICS_H_
#define MATHEMATICS_H_



namespace mathematics {

__device__ float sigmoid_once(float in);
__device__ float sigmoid_backward_derivated_once(float activation);

__device__ void sigmoid(float *in, float *out, int size);
__device__ void sigmoid_backward_derivated(float *activation, float *derivatives, int size);


__device__ void softmax(float *in, float *out, int size);
__device__ double cross_entropy(float *calculated, float *expected, int size);

__device__ float get_cost(float *output, float *labels, int size);
__device__ void get_cost_derivatives(float *output, float *labels, float *derivatives, int size);

}


#endif /* MATHEMATICS_H_ */
