#include "ProcessingUnit_cuPressorBatch.h"

__global__ void cuPressorBath(float* data, int size, float* factors, float* neutralPoints, int addressShift){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y; // do not set block y dimension, only grid y dimension
    if (idx < size){
        int index = idy * (addressShift + size) + idx + addressShift;
        data[index] = copysignf((-1 / ((1 + factors[idy]) * abs(data[index]) + 1) + 1) * (2 + factors[idy]) / (1 + factors[idy]), data[index]) * neutralPoints[idy];
    }
}

ProcessingUnit_cuPressorBatch::ProcessingUnit_cuPressorBatch(float*& d_workBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize, const uint bandCount, const uint& addressShift)
    : d_workBuffer(d_workBuffer), gridSize(gridSize), blockSize(blockSize), bufferSize(bufferSize), addressShift(addressShift), bandCount(bandCount) {
    
    activeFactors = bandCount;
    compressionFactors = new float[bandCount];
    neutralPoints = new float[bandCount];

    factorsBuffer = CuBufferFactory::createBuffer<float>(bandCount, CuBufferFactory::bufferType::TIME_OPTIMAL);
    neutralPointsBuffer = CuBufferFactory::createBuffer<float>(bandCount, CuBufferFactory::bufferType::TIME_OPTIMAL);

    setAllCompressionFactors(0.0);
    setAllNeutralPoints(1);
}

ProcessingUnit_cuPressorBatch::~ProcessingUnit_cuPressorBatch(){
    delete[] compressionFactors;
    delete[] neutralPoints;
    delete factorsBuffer;
    delete neutralPointsBuffer;
}

void ProcessingUnit_cuPressorBatch::process(){
    cuPressorBath<<<dim3(gridSize, bandCount), blockSize>>>(d_workBuffer, bufferSize, *factorsBuffer, *neutralPointsBuffer, addressShift);
}

float compressionFactorFunction(const float& factor){
    static const float minValue = 0.001;
    // TODO: find a volume corelation 
    return factor * 8 - (1 - minValue);
}

void ProcessingUnit_cuPressorBatch::setCompressionFactor(uint bandIndex, float factor){
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

    factor = compressionFactorFunction(factor);

    factorsBuffer->copyBuffer(FROM_HOST, &factor, 1, bandIndex);
}

void ProcessingUnit_cuPressorBatch::setAllCompressionFactors(float factor){
    setActive(factor != 0);
    for (int i = 0; i < bandCount; i++){
        compressionFactors[i] = compressionFactorFunction(factor);
    }
    factorsBuffer->copyBuffer(FROM_HOST, compressionFactors, bandCount);
}

float ProcessingUnit_cuPressorBatch::getCompressionFactor(uint bandIndex) const{
    if (bandIndex >= bandCount){
        return 0;
    }
    return compressionFactors[bandIndex];
}

void ProcessingUnit_cuPressorBatch::setNeutralPoint(uint bandIndex, float neutralPoint){
    if (bandIndex >= bandCount){
        return;
    }
    neutralPoints[bandIndex] = neutralPoint;
    neutralPointsBuffer->copyBuffer(FROM_HOST, &neutralPoint, 1, bandIndex);
}

void ProcessingUnit_cuPressorBatch::setAllNeutralPoints(float neutralPoint){
    for (int i = 0; i < bandCount; i++){
        neutralPoints[i] = neutralPoint;
    }
    neutralPointsBuffer->copyBuffer(FROM_HOST, neutralPoints, bandCount);
}

float ProcessingUnit_cuPressorBatch::getNeutralPoint(uint bandIndex) const{
    if (bandIndex >= bandCount){
        return 0;
    }
    return neutralPoints[bandIndex];
}