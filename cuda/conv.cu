#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <sys/mman.h>

#define blockx 16
#define blocky 16

#include "conv.h"
#include "tictoc.h"
#include "iaca.h"


#define get(X,Y) inData[(Y*width)+X]
#define set(X,Y,V) outData[(Y*width)+X] = V

__global__ void kernel_conv3(float * inData, float *outData, unsigned long width, unsigned long height, const float * filter) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;

    float acc1, acc2, acc3;
    acc1 = 0;

    if (idx < width - 1 && idy < height - 1 && idx > 0 && idy > 0) {
        //Valid target
        acc1 = get(idx - 1, idy - 1) * filter[0];
        acc2 = get(idx, idy - 1) * filter[3];
        acc3 = get(idx + 1, idy - 1) * filter[7];

        acc1 += get(idx - 1, idy) * filter[1];
        acc2 += get(idx, idy) * filter[4];
        acc3 += get(idx + 1, idy) * filter[8];

        acc1 += get(idx - 1, idy + 1) * filter[2];
        acc2 += get(idx, idy + 1) * filter[5];
        acc3 += get(idx + 1, idy + 1) * filter[9];

        set(idx, idy, acc1);
    } else if (idx < width && idy < height) {
        if (idy > 0) {
            if (idx > 0)
                acc1 += get(idx - 1, idy - 1) * filter[0];
            acc1 += get(idx, idy - 1) * filter[3];
            if (idx < width)
                acc1 += get(idx + 1, idy - 1) * filter[7];
        }
        if (idx > 0)
            acc1 += get(idx - 1, idy) * filter[1];
        acc1 += get(idx, idy) * filter[4];
        if (idx < width)
            acc1 += get(idx + 1, idy) * filter[8];

        if (idy < height) {
            if (idx > 0)
                acc1 += get(idx - 1, idy + 1) * filter[2];
            acc1 += get(idx, idy + 1) * filter[5];
            if (idx < width)
                acc1 += get(idx + 1, idy + 1) * filter[9];
        }
        set(idx, idy, acc1);
    }
}

double conv3(float** data, unsigned long width, unsigned long height, const float* filter) {

    data_t *inData = *data;

    //defs
    float * cudaIn, *cudaOut, *cudaFilter;


    //Init cuda
    cudaMalloc(&cudaIn, 0);

    tic();
    mlockall(0);
    cudaMalloc(&cudaFilter, sizeof (float)*9);
    cudaMalloc(&cudaIn, sizeof (float)*width * height);
    cudaMalloc(&cudaOut, sizeof (float)*width * height);

    //Calcualte block
    dim3 block;
    block.x = blockx;
    block.y = blocky;
    block.z = 1;
    dim3 blocks;
    blocks.x = width / block.x + (int) ((width % block.x) > 0);
    blocks.y = height / block.y + (int) ((height % block.y) > 0);
    blocks.z = 1;
    //do
    cudaMemcpy(cudaFilter, filter, sizeof (float)*9, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaIn, inData, sizeof (float)*width*height, cudaMemcpyHostToDevice);
    kernel_conv3 << <blocks, block>>>(cudaIn, cudaOut, width, height, cudaFilter);
    cudaMemcpy(inData, cudaOut, sizeof (float)*width*height, cudaMemcpyDeviceToHost);
    cudaFree(cudaIn);
    cudaFree(cudaOut);
    cudaFree(cudaFilter);
    double time = toc();

    cudaError_t cudaError = cudaGetLastError();
    printf("CudaError:\n%s\n", cudaGetErrorString(cudaError));

    return time;
}

__global__ void kernel_conv5(float * inData, float *outData, unsigned long width, unsigned long height, const float * filter) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;

    float acc1,acc2,acc3;
    acc1 = acc2 =acc3 = 0;

    if (idx < width - 2 && idy < height - 2 && idx > 1 && idy > 1) {
        for (int sx = -2; sx <= 2; sx++) {
            acc1 += get(idx + sx, idy -2) * filter[-2 * 5 + sx];
            acc2 += get(idx + sx, idy -1) * filter[-1 * 5 + sx];
            acc3 += get(idx + sx, idy + 0) * filter[0 * 5 + sx];
            acc1 += get(idx + sx, idy + 1) * filter[1 * 5 + sx];
            acc2 += get(idx + sx, idy + 2) * filter[2 * 5 + sx];
        }
        set(idx, idy, acc1 + acc2 + acc3);
    } else if (idx < width && idy < height) {
        for (int sy = -2; sy <= 2; sy++)
            for (int sx = -2; sx <= 2; sx++) {
                if (idx + sx > 0 && idx + sx < width && idx + sy > 0 && idx + sy < height)
                    acc1 += get(idx + sx, idy + sy) * filter[sy * 5 + sx];
            }
        set(idx, idy, acc1);
    }
}

double conv5(float** data, unsigned long width, unsigned long height, const float* filter) {

    data_t *inData = *data;

    //defs
    float * cudaIn, *cudaOut, *cudaFilter;


    //Init cuda
    cudaMalloc(&cudaIn, 0);

    tic();
    mlockall(0);
    cudaMalloc(&cudaFilter, sizeof (float)*25);
    cudaMalloc(&cudaIn, sizeof (float)*width * height);
    cudaMalloc(&cudaOut, sizeof (float)*width * height);

    //Calcualte block
    dim3 block;
    block.x = blockx;
    block.y = blocky;
    block.z = 1;
    dim3 blocks;
    blocks.x = width / block.x + (int) ((width % block.x) > 0);
    blocks.y = height / block.y + (int) ((height % block.y) > 0);
    block.z = 1;
    //do
    cudaMemcpy(cudaFilter, filter, sizeof (float)*25, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaIn, inData, sizeof (float)*width*height, cudaMemcpyHostToDevice);
    kernel_conv5 << <blocks, block>>>(cudaIn, cudaOut, width, height, cudaFilter);
    cudaMemcpy(inData, cudaOut, sizeof (float)*width*height, cudaMemcpyDeviceToHost);
    cudaFree(cudaIn);
    cudaFree(cudaOut);
    cudaFree(cudaFilter);
    double time = toc();

    cudaError_t cudaError = cudaGetLastError();
    printf("CudaError:\n%s\n", cudaGetErrorString(cudaError));

    return time;
}