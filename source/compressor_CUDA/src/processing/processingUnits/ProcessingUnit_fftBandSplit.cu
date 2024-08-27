#include "ProcessingUnit_fftBandSplit.h"

__global__ void fftBandSplit(cufftComplex* input, cufftComplex* output, float* bandMasks, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y; // do not set block y dimension, only grid y dimension
    if (idx < size){
        int index = idx + idy * size;
        output[index].x = input[idx].x * bandMasks[index] / size;
        output[index].y = input[idx].y * bandMasks[index] / size;
    }
}

ProcessingUnit_fftBandSplit::ProcessingUnit_fftBandSplit(cufftComplex*& d_input, cufftComplex*& d_output, const dim3& gridSize2D, const uint& blockSize, const uint& complexWindowSize, const uint& sampleRate) 
    : d_input(d_input), d_output(d_output), gridSize2D(gridSize2D), blockSize(blockSize), complexWindowSize(complexWindowSize), sampleRate(sampleRate){
    bandMasks = CuBufferFactory::createBuffer<float>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    // generateBandSplittingTable();
}

ProcessingUnit_fftBandSplit::~ProcessingUnit_fftBandSplit(){
    delete bandMasks;
}

void ProcessingUnit_fftBandSplit::process(){
    fftBandSplit<<<gridSize2D, blockSize>>>(d_input, d_output, *bandMasks, complexWindowSize);
}

float softSignDescend(float x, const float& minVal, const float& range){
    x = -8*(x/range - 0.5);
    return (x/(1.6*(1 + abs(x))) + 0.5) * (1 - minVal) + minVal;
}

float linearDescend(const float& x, const float& minVal, const float& range){
   return 1 - x / range * (1 - minVal);
}

void ProcessingUnit_fftBandSplit::generateBandSplittingTable(){
    uint bandCount = gridSize2D.y;
    bandMasks->resize(bandCount * complexWindowSize);
    float* masks = new float[bandCount * complexWindowSize];
    float maxFrequency = 40000.0;
    float minFrequency = 20.0;
    float sizeRatio = std::pow(maxFrequency / minFrequency, 1.0 / bandCount);
    float maskMinValue = 0.05;
    float previousbandHalfWidth = 0;
    for (int i = 0; i < bandCount; i++){
        float startFrequency = minFrequency * std::pow(sizeRatio, i);
        float endFrequency = minFrequency * std::pow(sizeRatio, i + 1);
        float bandHalfWidth = (endFrequency - startFrequency) / 2;
        printf("Band %d: %.04f - %.04f\n", i, startFrequency, endFrequency);
    
        for (int j = 1; j < complexWindowSize - 1; j++){
            float frequency = (float)j / (float)complexWindowSize * (float)sampleRate;
            if (frequency > endFrequency - bandHalfWidth){
                float valueAbove = frequency - (endFrequency - bandHalfWidth);
                if (valueAbove > bandHalfWidth * 2){
                    masks[i * complexWindowSize + j] = maskMinValue;
                } else {
                    masks[i * complexWindowSize + j] = softSignDescend(valueAbove, maskMinValue, bandHalfWidth * 2);
                }

            } else if (frequency < startFrequency + previousbandHalfWidth){
                float valueBelow = startFrequency + previousbandHalfWidth - frequency;
                if (valueBelow > previousbandHalfWidth * 2){
                    masks[i * complexWindowSize + j] = maskMinValue;
                } else {
                    masks[i * complexWindowSize + j] = softSignDescend(valueBelow, maskMinValue, previousbandHalfWidth * 2);
                }

            } else {
                masks[i * complexWindowSize + j] = 1;
            }
            if (frequency > startFrequency - previousbandHalfWidth && frequency < endFrequency + bandHalfWidth){
                printf("   %.04f, %.04f\n", frequency, masks[i * complexWindowSize + j]);
            }
        }
        masks[i * complexWindowSize] = 1;
        masks[i * complexWindowSize + complexWindowSize - 1] = 1;
        previousbandHalfWidth = bandHalfWidth;
    }

    bandMasks->copyBuffer(FROM_HOST, masks, bandCount * complexWindowSize);
    delete[] masks;
}
