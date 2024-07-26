#include "Compressor.h"

#include <cuda_runtime.h>
#include <cuda.h>

__global__ void cuPressor(float* data, int size, float factor){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        data[idx] = copysignf((-1 / ((1 + factor) * abs(data[idx]) + 1) + 1) * (2 + factor) / (1 + factor), data[idx]);
    }
}

Compressor::Compressor(){
    gpuBufferSize = 0;
    d_buffer = nullptr;
    compressionFactor = 0.5;
    allocateIfNeeded(1024);
}

Compressor::~Compressor(){
    if (d_buffer != nullptr){
        cudaFree(d_buffer);
    }
}

void Compressor::compress(float* samplesIn, float* samplesOut, int size){
	static const int blockSize = 256;
	allocateIfNeeded(size);

	int numBlocks = (size + blockSize - 1) / blockSize;
	cudaMemcpy(d_buffer, samplesIn, size * sizeof(float), cudaMemcpyHostToDevice);
	cuPressor<<<numBlocks, blockSize>>>((float*) d_buffer, size, compressionFactor);
	cudaMemcpy(samplesOut, d_buffer, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}

void Compressor::allocateIfNeeded(int size){
	if (gpuBufferSize < size){
		if (d_buffer != nullptr){
			cudaFree(d_buffer);
		}
		cudaMalloc(&d_buffer, size * sizeof(float));
		gpuBufferSize = size;
	}
}

void Compressor::setCompressionFactor(double& parameter){
    static const float minValue = 0.0001;
    if (parameter < 0){
        parameter = 0;
    } else if (parameter > 1){
        parameter = 1;
    }
    float out = parameter * 2;
    for (int i = 0; i < 3; i++){
        out *= out;
    }
    compressionFactor =  out - (1 - minValue);
}