#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <sys/mman.h>

#include "conv.h"
#include "tictoc.h"
#include "iaca.h"


#define get(X,Y) inData[(Y*width)+X]
#define set(X,Y,V) outData[(Y*width)+X] = V

__global__ void kernel_conv3(float * inData, float *outData, unsigned long width, unsigned long height, const float * filter) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x; 
    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    float acc1;
    acc1=0;
    
    if (idx < width && idy < height) {
        //Valid target
        if (idy > 0) {
            if (idx > 0) 
                acc1 += get(idx-1, idy-1) * filter[0];
            acc1 +=     get(idx,idy-1) * filter[3];
            if (idx < width) 
                acc1 += get(idx+1, idy-1) * filter[7];
        }
        if (idx > 0) 
            acc1 += get(idx-1, idy) * filter[1];
        acc1 +=     get(idx,idy) * filter[4];
        if (idx < width) 
            acc1 += get(idx+1, idy) * filter[8];
        
        if (idy < height) {
            if (idx > 0) 
                acc1 += get(idx-1, idy+1) * filter[2];
            acc1 +=     get(idx,idy+1) * filter[5];
            if (idx < width) 
                acc1 += get(idx+1, idy+1) * filter[9];
        }
        set(idx,idy,acc1);
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
    cudaMalloc(&cudaFilter, sizeof(float)*9);
    cudaMalloc(&cudaIn, sizeof(float)*width*height);
    cudaMalloc(&cudaOut, sizeof(float)*width*height);
    
    //Calcualte block
    dim3 block;
    block.x = 8;
    block.y = 8;
    block.z = 0;
    dim3 blocks;
    blocks.x = width/8 + (int)((width%8)>0);
    blocks.y = height/8 + (int)((height%8)>0);
    blocks.z = 0;
    
    //do
    cudaMemcpy(cudaFilter, filter, sizeof(float)*9,cudaMemcpyHostToDevice);
    cudaMemcpy(cudaIn, inData, sizeof(float)*width*height, cudaMemcpyHostToDevice);
    kernel_conv3<<<block, blocks>>>(cudaIn, cudaOut, width, height, cudaFilter);
    cudaMemcpy(inData, cudaOut, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    cudaFree(cudaIn);
    cudaFree(cudaOut);
    cudaFree(cudaFilter);
    double time = toc();

    return time;
}

__global__ void kernel_conv5(float * inData, float *outData, unsigned long width, unsigned long height, const float * filter) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x; 
    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    float acc1;
    acc1=0;
    
    if (idx < width && idy < height) {
        for(int sy=-2; sy <= 2; sy++)
            for(int sx=-2; sx <= 2; sx++) {
                if (idx+sx > 0 && idx+sx < width && idx+sy>0 && idx+sy<height) 
                    acc1 += get(idx+sx, idx+sy) * filter[sy*5+sx];
            }
        set(idx,idy,acc1);
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
    cudaMalloc(&cudaFilter, sizeof(float)*25);
    cudaMalloc(&cudaIn, sizeof(float)*width*height);
    cudaMalloc(&cudaOut, sizeof(float)*width*height);
    
    //Calcualte block
    dim3 block;
    block.x = 8;
    block.y = 8;
    dim3 blocks;
    blocks.x = width/8 + (width%8)>0;
    blocks.y = height/8 + (height%8)>0;
    
    //do
    cudaMemcpy(cudaFilter, filter, sizeof(float)*25,cudaMemcpyHostToDevice);
    cudaMemcpy(cudaIn, inData, sizeof(float)*width*height, cudaMemcpyHostToDevice);
    kernel_conv5<<<block, blocks>>>(cudaIn, cudaOut, width, height, cudaFilter);
    cudaMemcpy(inData, cudaOut, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    cudaFree(cudaIn);
    cudaFree(cudaOut);
    cudaFree(cudaFilter);
    double time = toc();

    return time;
}