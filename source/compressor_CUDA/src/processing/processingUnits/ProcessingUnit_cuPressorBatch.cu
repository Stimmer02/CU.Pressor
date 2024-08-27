#include "ProcessingUnit_cuPressorBatch.h"

__global__ void cuPressorBath(float* data, int size, float* factors, float volume, int addressShift){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y; // do not set block y dimension, only grid y dimension
    if (idx < size){
        int index = idy * (addressShift + size) + idx + addressShift;
        data[index] = copysignf((-1 / ((1 + factors[idy]) * abs(data[index] * volume) + 1) + 1) * (2 + factors[idy]) / (1 + factors[idy]), data[index]);
    }
}

ProcessingUnit_cuPressorBatch::ProcessingUnit_cuPressorBatch(float*& d_workBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize, const uint bandCount, const uint& addressShift)
    : d_workBuffer(d_workBuffer), gridSize(gridSize), blockSize(blockSize), bufferSize(bufferSize), addressShift(addressShift), bandCount(bandCount) {
    preGain = 1;
    
    compressionFactors = new float[bandCount];
    for (int i = 0; i < bandCount; i++){
        compressionFactors[i] = 0.4;
    }
    activeFactors = bandCount;

    cudaMalloc(&d_compressionFactors, bandCount * sizeof(float));
    cudaMemcpy(d_compressionFactors, compressionFactors, bandCount * sizeof(float), cudaMemcpyHostToDevice);
}

ProcessingUnit_cuPressorBatch::~ProcessingUnit_cuPressorBatch(){
    cudaFree(d_compressionFactors);
    delete[] compressionFactors;
}

void ProcessingUnit_cuPressorBatch::process(){
    cuPressorBath<<<(gridSize, bandCount), blockSize>>>(d_workBuffer, bufferSize, d_compressionFactors, preGain, addressShift);
}

void ProcessingUnit_cuPressorBatch::setCompressionFactor(uint bandIndex, float factor){
    static const float minValue = 0.001;

    if (bandIndex >= bandCount){
        return;
    }
    if (compressionFactors[bandIndex] == 0 && factor != 0){
        activeFactors++;
    } else if (compressionFactors[bandIndex] != 0 && factor == 0){
        activeFactors--;
    }
    compressionFactors[bandIndex] = factor;

    setActive(activeFactors != 0);

    double out = factor * 2; // TODO: find a volume corelation 
    double multiplier = out;
    for (int i = 0; i < 2; i++){
        out *= multiplier;
    }
    factor = out - (1 - minValue);

    cudaMemcpy(d_compressionFactors + bandIndex, &factor, sizeof(float), cudaMemcpyHostToDevice);
}

float ProcessingUnit_cuPressorBatch::getCompressionFactor(uint bandIndex) const{
    if (bandIndex >= bandCount){
        return INFINITY;
    }
    return compressionFactors[bandIndex];
}

void ProcessingUnit_cuPressorBatch::setPreGain(float gain){
    preGain = gain;
}

float ProcessingUnit_cuPressorBatch::getPreGain() const{
    return preGain;
}

