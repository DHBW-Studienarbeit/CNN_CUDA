/*
 * mathematics.cpp
 *
 *  Created on: 05.12.2017
 *      Author: Florian
 */

#include "mathematics.h"
#include "math.h"

namespace mathematics {


__device__ float sigmoid_once(float in)
{
	double temp = exp(in);
	return (float)(temp / (1+temp));
}

__device__ float sigmoid_backward_derivated_once(float activation)
{
	return activation * (1 - activation);
}

__device__ void sigmoid(float *in, float *out, int size)
{
	for(; size>0; size--, in++, out++)
	{
		*out = sigmoid_once(*in);
	}
}

__device__ void sigmoid_backward_derivated(float *activation, float *derivatives, int size)
{
	for(; size>0; size--, activation++, derivatives++)
	{
		*derivatives = sigmoid_backward_derivated_once(*activation);
	}
}


__device__ void softmax(float *in, float *out, int size)
{
	double sum=0;
	for(int i=0; i<size; i++)
	{
		sum += exp(in[i]);
	}
	for(int i=0; i<size; i++)
	{
		out[i] = (float)(exp(in[i]) / sum);
	}
}


__device__ float cross_entropy(float *calculated, float *expected, int size)
{
	double sum=0;
	for(; size>0; size--, expected++, calculated++)
	{
		sum += - (*expected) * log(*calculated);
	}
	return (float)sum;
}

__device__ float get_cost(float *output, float *labels, int size)
{
	float *normalized;
	float ret;
	normalized = new float[size];
	softmax(output, normalized, size);
	ret = cross_entropy(normalized, labels, size);
	delete[] normalized;
	return ret;
}

/* quadratic cost function derivated */
__device__ void get_cost_derivatives(float *output, float *labels, float *derivatives, int size)
{
	for(; size>0; size--, output++, labels++, derivatives++)
	{
		*derivatives = *output - *labels;
	}
}

__device__ void hadamard_product(float* a, float* b, float* out, int size)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	for(int i = index; i < size; i+=stride)
	{
		out[i] = a[i] * b[i];
	}
	__syncthreads();
}

} /* namespace mathematics */
