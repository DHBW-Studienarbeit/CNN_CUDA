/*
 * cuda_kernels.cu
 *
 *  Created on: 09.03.2018
 *      Author: benjamin
 */

#include <cuda.h>
#include <curand.h>


namespace cuda {

__global__
void init(float** nodeArrayPtrs, int no_node_matrices, int* arrayLengths)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < no_node_matrices; i+=stride)
	{
		curandGenerator_t generator;
		curandStatus_t curand_state;

		curand_state = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

		curand_state = curandSetPsudoRandomGeneratorSeed(gen, 1234ULL);

		curand_state = curandGenerateUniform(generator, nodeArrayPtrs[i], arrayLengths[i]);

		curand_state = curandDestroyGenerator(generator);
	}
}

} /* end namespace cuda */

