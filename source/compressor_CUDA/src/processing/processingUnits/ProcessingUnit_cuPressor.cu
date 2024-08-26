#include "ProcessingUnit_cuPressor.h"

__global__ void cuPressor(float* data, int size, float factor){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        data[idx] = copysignf((-1 / ((1 + factor) * abs(data[idx]) + 1) + 1) * (2 + factor) / (1 + factor), data[idx]);
    }
}

ProcessingUnit_cuPressor::ProcessingUnit_cuPressor(float*& d_workBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize)
    : d_workBuffer(d_workBuffer), gridSize(gridSize), blockSize(blockSize), bufferSize(bufferSize){
    compressionFactor = 0.0f;
}

void ProcessingUnit_cuPressor::process(){
    cuPressor<<<gridSize, blockSize>>>(d_workBuffer, bufferSize, processedCompressionFactor);
}

void ProcessingUnit_cuPressor::setCompressionFactor(float factor){
    bool active = factor != 0.0f;
    setActive(active);
    if (active == false){
        return;
    }

    double out = factor * 2; // TODO: find a volume corelation 
    double multiplier = out;
    for (int i = 0; i < 2; i++){
        out *= multiplier;
    }
    processedCompressionFactor = out - 1;
}

float ProcessingUnit_cuPressor::getCompressionFactor() const{
    return compressionFactor;
}
