/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include "kernel.h"

/* Matrix size */
#define N  (275)

/* Host implementation of a simple version of sgemm */


/* Main */
int main(int argc, char **argv)
{
    cublasStatus_t status;
    float *h_A;
    float *h_B;
    float *h_C;
    float *h_C_ref;
    float *d_A = 0;
    float *d_B = 0;
    float *d_C = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;
    cublasHandle_t handle;

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n");

    status = cublasCreate(&handle);

    h_A = (float*) malloc(3*2*sizeof(float));
    h_B = (float*) malloc(2*3*sizeof(float));
    h_C = (float*) malloc(3*3*sizeof(float));

    for(int i = 0; i < 6; i++)
    {
    	h_A[i] = 1;
    	h_B[i] = 1;
    }

    h_A[0] = 2;
    h_A[5] = 2;
    h_A[3] = 3;

    for(int i = 0; i < 6; i++)
    {
    	printf("%f \t", h_A[i]);
    	if((i+1)%2 == 0)
    	{
    		printf("\n\r");
    	}
    }


    printf("\n\n\r");

    for(int i = 0; i < 6; i++)
    {
    	printf("%f \t", h_B[i]);
    	if((i+1)%3 == 0)
    	{
    		printf("\n\r");
    	}
    }

    printf("\n\n\r");

    cudaError_t cuda_error = cudaSuccess;

    cuda_error = cudaMalloc((void**)&d_A, 24);
    cuda_error = cudaMalloc((void**)&d_B, 2*3*sizeof(float));
    cuda_error = cudaMalloc((void**)&d_C, 3*3*sizeof(float));

    cudaMemcpy(d_A, h_A, 3*2* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 2*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, 3*3*sizeof(float), cudaMemcpyHostToDevice);

//    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 2,
//    		(const float*) &alpha, (const float*) d_A, 3,
//    		(const float*) d_B, 2, (const float*) &beta,
//    		d_C, 3);


    simple_sgemm<<<1,10>>>(6, alpha, d_A, d_B, beta, d_C);

    cudaMemcpy(h_C, d_C, 3*3*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 9; i++)
    {
    	printf("%f \t", h_C[i]);
    	if((i+1)%3 == 0)
    	{
    		printf("\n\r");
    	}
    }

    cublasDestroy(handle);

}
