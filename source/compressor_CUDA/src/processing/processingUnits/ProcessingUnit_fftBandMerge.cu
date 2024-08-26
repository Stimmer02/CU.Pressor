#include "ProcessingUnit_fftBandMerge.h"

__global__ void bandMerge(float* input, float* output, int arrShift, int size, int bandCount){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        output[idx] = input[idx + arrShift];
        for (int i = 1; i < bandCount; i++){
            output[idx] += input[i * (arrShift + size) + idx + arrShift];
        }
        output[idx] /= 2.2;
    }
}

ProcessingUnit_fftBandMerge::ProcessingUnit_fftBandMerge(float*& d_inputBuffer, float*& d_outputBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize, const uint& bandCount, const uint& addressShift)
    : d_inputBuffer(d_inputBuffer), d_outputBuffer(d_outputBuffer), gridSize(gridSize), blockSize(blockSize), bufferSize(bufferSize), addressShift(addressShift), bandCount(bandCount) {}


void ProcessingUnit_fftBandMerge::process(){
    bandMerge<<<gridSize, blockSize>>>(d_inputBuffer, d_outputBuffer, addressShift, bufferSize, bandCount);
}