#include "Compressor.h"


__global__ void volumeControl(float* data, int size, float volume){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        data[idx] *= volume;
    }
}

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

Compressor::Compressor(){
    cufftR2C = 0;
    cufftC2R = 0;
    workBuffer = new CuShiftBuffer<cufftReal>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    cufftOutput = CuBufferFactory::createBuffer<cufftComplex>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    double temp = 0.4;
    setCompressionFactor1(temp);
    setCompressionFactor2(temp);
    temp = 1;
    setVolume(temp);
    temp = 0.5;
    setPreGain(temp);
    setWindowSize(4096);
}

Compressor::~Compressor(){
    if (cufftR2C != 0) {
        cufftDestroy(cufftR2C);
    }
    if (cufftC2R != 0) {
        cufftDestroy(cufftC2R);
    }
    delete cufftOutput;
    delete workBuffer;
}

void Compressor::compress(float* samplesIn, float* samplesOut, int size){
	static const int blockSize = 256;

    CuShiftBuffer<cufftReal>& workBuffer = *this->workBuffer;
    ACuBuffer<cufftComplex>& cufftOutput = *this->cufftOutput;

    if (size < windowSize){
        int numBlocksFragment = (size + blockSize - 1) / blockSize;
        workBuffer.put(FROM_HOST, samplesIn, size);
        volumeControl<<<numBlocksFragment, blockSize>>>(workBuffer[windowSize - size], size, preGain*2);
        cufftExecR2C(cufftR2C, workBuffer, cufftOutput);
        int numBlocksComplex = (windowSize / 2 + blockSize) / blockSize;
        cuPressorComplex<<<numBlocksComplex, blockSize>>>(cufftOutput, windowSize / 2 + 1, compressionFactor1, windowSize);
        cufftExecC2R(cufftC2R, cufftOutput, workBuffer.getInactiveBuffer());
        cuPressor<<<numBlocksFragment, blockSize>>>(workBuffer.getInactiveBuffer(windowSize - size), size, compressionFactor2, volume / (preGain * 2));
        workBuffer.copyInactiveBuffer(TO_HOST, samplesOut, size, windowSize - size);
    } else if (size == windowSize){
        workBuffer.put(FROM_HOST, samplesIn, size);
        cudaMemcpy(workBuffer.getInactiveBuffer(), samplesIn, size * sizeof(float), cudaMemcpyHostToDevice);
        int numBlocksAll = (windowSize + blockSize - 1) / blockSize;
        volumeControl<<<numBlocksAll, blockSize>>>(workBuffer.getInactiveBuffer(windowSize - size), size, preGain*2);
        cufftExecR2C(cufftR2C, workBuffer, cufftOutput);
        int numBlocksComplex = (windowSize / 2 + blockSize) / blockSize;
        cuPressorComplex<<<numBlocksComplex, blockSize>>>(cufftOutput, windowSize / 2 + 1, compressionFactor1, windowSize);
        cufftExecC2R(cufftC2R, cufftOutput, workBuffer.getInactiveBuffer());
        cuPressor<<<numBlocksAll, blockSize>>>(workBuffer.getInactiveBuffer(windowSize - size), size, compressionFactor2, volume / (preGain * 2));
        workBuffer.copyInactiveBuffer(TO_HOST, samplesOut);
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
    workBuffer->resize(size);
    workBuffer->clear();
    cufftOutput->resize(size / 2 + 1);
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
    compressionFactor2 = out - (1 - minValue);
}

void Compressor::setVolume(double& parameter){
    if (parameter < 0){
        parameter = 0;
    } else if (parameter > 1){
        parameter = 1;
    }
    volume = parameter;
}

void Compressor::setPreGain(double& parameter){
    if (parameter <= 0){
        parameter = 0.001;
    } else if (parameter > 1){
        parameter = 1;
    }
    preGain = parameter;
}