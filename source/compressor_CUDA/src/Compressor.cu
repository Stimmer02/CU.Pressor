#include "Compressor.h"


__global__ void cuPressor(float* data, int size, float factor, float volume){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        data[idx] = copysignf((-1 / ((1 + factor) * abs(data[idx]) + 1) + 1) * (2 + factor) / (1 + factor), data[idx]) * volume;
    }
}

__global__ void cuPressorComplex1(cufftComplex* data, int size, float factor, int windowSize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        data[idx].x = (-1 / ((1 + factor) * data[idx].x / windowSize + 1) + 1) * (2 + factor) / (1 + factor);
        data[idx].y = (-1 / ((1 + factor) * data[idx].y / windowSize + 1) + 1) * (2 + factor) / (1 + factor);
    }
}

__global__ void cuPressorComplex(cufftComplex* data, int size, float factor, int windowSize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        float magnitude = hypotf(data[idx].x, data[idx].y);
        float phase = atan2f(data[idx].y, data[idx].x);
        magnitude = (-1 / ((1 + factor) * magnitude / windowSize + 1) + 1) * (2 + factor) / (1 + factor);
        data[idx].x = magnitude * cosf(phase);
        data[idx].y = magnitude * sinf(phase);
    }
}

__global__ void shiftData(float* inputBuffer, float* outputBuffer, int size, int shift){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // For optimal thread usage, max idx should be size - shift
    if (idx < size - shift){
        outputBuffer[idx] = inputBuffer[idx + shift];
    }
}

Compressor::Compressor(){
    cufftR2C = 0;
    cufftC2R = 0;
    allocatedMemorySize = 0;
    workBufferIndex = 0;
    d_workBuffer[0] = nullptr;
    d_workBuffer[1] = nullptr;
    d_cufftOutput = nullptr;
    compressionFactor1 = 0.5;
    setWindowSize(1024);
}

Compressor::~Compressor(){
    if (cufftR2C != 0) {
        cufftDestroy(cufftR2C);
    }
    if (cufftC2R != 0) {
        cufftDestroy(cufftC2R);
    }
    if (d_workBuffer[0] != nullptr) {
        cudaFree(d_workBuffer[0]);
    }
    if (d_workBuffer[1] != nullptr) {
        cudaFree(d_workBuffer[1]);
    }
    if (d_cufftOutput != nullptr) {
        cudaFree(d_cufftOutput);
    }
}

void Compressor::compress(float* samplesIn, float* samplesOut, int size){
	static const int blockSize = 256;

    if (size < windowSize){
        int numBlocksFragment = (windowSize - size + blockSize - 1) / blockSize;
        shiftData<<<numBlocksFragment, blockSize>>>(d_workBuffer[workBufferIndex], d_workBuffer[!workBufferIndex], windowSize, size);
        workBufferIndex = !workBufferIndex;
        cudaMemcpy(d_workBuffer[workBufferIndex] + windowSize - size, samplesIn, size * sizeof(float), cudaMemcpyHostToDevice);
        cufftExecR2C(cufftR2C, d_workBuffer[workBufferIndex], d_cufftOutput);
        int numBlocksComplex = (windowSize / 2 + blockSize) / blockSize;
        cuPressorComplex<<<numBlocksComplex, blockSize>>>(d_cufftOutput, windowSize / 2 + 1, compressionFactor1, windowSize);
        cufftExecC2R(cufftC2R, d_cufftOutput, d_workBuffer[!workBufferIndex]);
        cuPressor<<<numBlocksFragment, blockSize>>>(d_workBuffer[!workBufferIndex] + windowSize - size, size, compressionFactor2, volume);
        cudaMemcpy(samplesOut, d_workBuffer[!workBufferIndex] + windowSize - size, size * sizeof(float), cudaMemcpyDeviceToHost);
    } else if (size == windowSize){
        cudaMemcpy(d_workBuffer[workBufferIndex], samplesIn, size * sizeof(float), cudaMemcpyHostToDevice);
        cufftExecR2C(cufftR2C, d_workBuffer[workBufferIndex], d_cufftOutput);
        int numBlocksComplex = (windowSize / 2 + blockSize) / blockSize;
        cuPressorComplex<<<numBlocksComplex, blockSize>>>(d_cufftOutput, windowSize / 2 + 1, compressionFactor1, windowSize);
        cufftExecC2R(cufftC2R, d_cufftOutput, d_workBuffer[!workBufferIndex]);
        int numBlocksAll = (windowSize + blockSize - 1) / blockSize;
        cuPressor<<<numBlocksAll, blockSize>>>(d_workBuffer[!workBufferIndex] + windowSize - size, size, compressionFactor2, volume);
        cudaMemcpy(samplesOut, d_workBuffer[!workBufferIndex], size * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        int leftToProcess = size;
        for (int i = 0; i < size - windowSize; i += windowSize){
            compress(samplesIn + i, samplesOut + i, (leftToProcess > windowSize) ? windowSize : leftToProcess);
            leftToProcess -= windowSize;
        }
    }
	cudaDeviceSynchronize();
}

void Compressor::allocateIfNeeded(int size){
	if (allocatedMemorySize < size){
		if (d_workBuffer[0] != nullptr) {
            cudaFree(d_workBuffer[0]);
        }
        if (d_workBuffer[1] != nullptr) {
            cudaFree(d_workBuffer[1]);
        }
        if (d_cufftOutput != nullptr) {
            cudaFree(d_cufftOutput);
        }
		cudaMalloc(&d_workBuffer[0], size * sizeof(cufftReal));
		cudaMalloc(&d_workBuffer[1], size * sizeof(cufftReal));
        cudaMalloc(&d_cufftOutput, (size / 2 + 1) * sizeof(cufftComplex));

        cudaMemset(d_workBuffer[0], 0, size * sizeof(cufftReal));
        cudaMemset(d_workBuffer[1], 0, size * sizeof(cufftReal));
        cudaMemset(d_cufftOutput, 0, (size / 2 + 1) * sizeof(cufftComplex));
		allocatedMemorySize = size;
	}
}

void Compressor::setWindowSize(int size){
    windowSize = size;
    allocateIfNeeded(windowSize);
    if (cufftR2C != 0) {
        cufftDestroy(cufftR2C);
    }
    if (cufftC2R != 0) {
        cufftDestroy(cufftC2R);
    }
    cufftPlan1d(&cufftR2C, windowSize, CUFFT_R2C, 1);
    cufftPlan1d(&cufftC2R, windowSize, CUFFT_C2R, 1);
}

void Compressor::setCompressionFactor1(double& parameter){
    static const float minValue = 0.001;
    if (parameter < 0){
        parameter = 0;
    } else if (parameter > 1){
        parameter = 1;
    }
    double out = parameter * 1.2;
    double multiplier = out;
    for (int i = 0; i < 2; i++){
        out *= multiplier;
    }
    compressionFactor1 = out - (1 - minValue);
}

void Compressor::setCompressionFactor2(double& parameter){
    static const float minValue = 0.001;
    if (parameter < 0){
        parameter = 0;
    } else if (parameter > 1){
        parameter = 1;
    }
    double out = parameter * 1.2;
    double multiplier = out;
    for (int i = 0; i < 2; i++){
        out *= multiplier;
    }
    compressionFactor1 = out - (1 - minValue);
}

void Compressor::setVolume(double& parameter){
    if (parameter < 0){
        parameter = 0;
    } else if (parameter > 1){
        parameter = 1;
    }
    volume = parameter;
}