#include "ProcessingUnit_windowing.h"

#define PI 3.14159265358979323846

__global__ void applyWindow(const float* input, float* output, const float* window, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        output[idx] = input[idx] * window[idx];
    }
}

__global__ void generateAlaPlankTaperWindow(float* window, int size, int startLength, int endLength){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        if (idx < startLength){
            window[idx] = 0.5 * (1 - cosf(PI * idx / startLength));
        } else if (idx >= size - endLength){
            window[idx] = 0.5 * (1 - cosf(PI * (size - idx) / endLength));
        } else {
            window[idx] = 1;
        }
    }
}

ProcessingUnit_windowing::ProcessingUnit_windowing(float*& d_inputBuffer, float*& d_outputBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize)
    : d_inputBuffer(d_inputBuffer), d_outputBuffer(d_outputBuffer), gridSize(gridSize), blockSize(blockSize), bufferSize(bufferSize){
    windowBuffer = CuBufferFactory::createBuffer<float>(bufferSize, CuBufferFactory::bufferType::TIME_OPTIMAL);
}

ProcessingUnit_windowing::~ProcessingUnit_windowing(){
    delete windowBuffer;
}

void ProcessingUnit_windowing::process(){
    applyWindow<<<gridSize, blockSize>>>(d_inputBuffer, d_outputBuffer, *windowBuffer, bufferSize);
}

void ProcessingUnit_windowing::generateWindow(int startLength, int endLength){
    windowBuffer->resize(bufferSize);
    generateAlaPlankTaperWindow<<<gridSize, blockSize>>>(*windowBuffer, bufferSize, startLength, endLength);
}