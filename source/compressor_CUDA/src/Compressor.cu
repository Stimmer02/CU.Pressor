#include "Compressor.h"


__global__ void volumeControl(float* data, int size, float volume){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        data[idx] *= volume;
    }
}

__global__ void cuPressor(cufftReal* data, int size, float factor, float volume){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        data[idx] = copysignf((-1 / ((1 + factor) * abs(data[idx]) + 1) + 1) * (2 + factor) / (1 + factor), data[idx]) * volume;
    }
}
__global__ void cuPressorBath(cufftReal* data, int size, float factor, float volume, int addressShift){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y; // do not set block y dimension, only grid y dimension
    if (idx < size){
        int index = idy * (addressShift + size) + idx + addressShift;
        data[index] = copysignf((-1 / ((1 + factor) * abs(data[index] * volume) + 1) + 1) * (2 + factor) / (1 + factor), data[index]);
    }
}
__global__ void fftBandSplit(cufftComplex* input, cufftComplex* output, int size, int bandWidth){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y; // do not set block y dimension, only grid y dimension
    if (idx < size){
        bool copyBand = ((idx >= (idy * bandWidth)) && (idx < ((idy + 1) * bandWidth))) || (size - 1 == idx);
        float bandMultiplier = 1 * copyBand + 0.001 * !copyBand;
        int index = idx + idy * size;
        output[index].x = input[idx].x * bandMultiplier;
        output[index].y = input[idx].y * bandMultiplier;
    }
}

__global__ void fftBandSplit_smooth(cufftComplex* input, cufftComplex* output, int size, int bandWidth){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y; // do not set block y dimension, only grid y dimension
    if (idx < size){
        float copyBand = 1 - 0.5 * cosf(3.14159265358979323846 * (idx - idy * bandWidth) / bandWidth);
        if ((size - 1 == idx) || (idx == 0)){
            copyBand = 1;
        }
        int index = idx + idy * size;
        output[index].x = input[idx].x * copyBand;
        output[index].y = input[idx].y * copyBand;
    }
}

__global__ void bandMerge(cufftReal* input, cufftReal* output, int arrShift, int size, int bandCount){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        output[idx] = input[idx + arrShift];
        for (int i = 1; i < bandCount; i++){
            output[idx] += input[i * (arrShift + size) + idx + arrShift];
        }
        output[idx] /= bandCount;
    }
}


Compressor::Compressor(){
    bandCount = 8;
    cufftR2C = 0;
    cufftC2R = 0;
    workBuffer = new CuShiftBuffer<cufftReal>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    cufftOutput = CuBufferFactory::createBuffer<cufftComplex>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    cufftBands = CuBufferFactory::createBuffer<cufftComplex>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    bands = CuBufferFactory::createBuffer<cufftReal>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    double temp = 0.4;
    setCompressionFactor1(temp);
    setCompressionFactor2(temp);
    temp = 1;
    setVolume(temp);
    temp = 0.5;
    setPreGain(temp);
    setWindowSize(1024);
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
    delete cufftBands;
    delete bands;
}

void Compressor::compress(float* samplesIn, float* samplesOut, uint size){
	static const uint blockSize = 256;

    CuShiftBuffer<cufftReal>& workBuffer = *this->workBuffer;
    ACuBuffer<cufftComplex>& cufftOutput = *this->cufftOutput;
    ACuBuffer<cufftComplex>& cufftBands = *this->cufftBands;
    ACuBuffer<cufftReal>& bands = *this->bands;

    if (size <= windowSize){
        uint gridSizeComplex = (complexWindowSize + blockSize - 1) / blockSize;
        uint gridSizeReal = (size + blockSize - 1) / blockSize;
        dim3 gridSizeReal2D = {gridSizeReal, bandCount};
        dim3 gridSizeComplex2D = {gridSizeComplex, bandCount};

        uint addressShift = windowSize - size;

        
        workBuffer.pushBack(FROM_HOST, samplesIn, size);

        volumeControl<<<gridSizeReal, blockSize>>>(workBuffer[addressShift], size, preGain * 2);

        cufftExecR2C(cufftR2C, workBuffer, cufftOutput);
        fftBandSplit<<<gridSizeComplex2D, blockSize>>>(cufftOutput, cufftBands, complexWindowSize, (complexWindowSize - 1) / bandCount);
        cufftExecC2R(cufftC2R, cufftBands, bands);

        cuPressorBath<<<gridSizeReal2D, blockSize>>>(bands, size, compressionFactor1, (float)bandCount / windowSize, addressShift);
        bandMerge<<<gridSizeReal2D, blockSize>>>(bands, workBuffer.getInactiveBuffer(), addressShift, size, bandCount);

        volumeControl<<<gridSizeReal, blockSize>>>(workBuffer.getInactiveBuffer(), size, volume / (preGain * 2));
        // cuPressor<<<gridSizeReal, blockSize>>>(workBuffer.getInactiveBuffer(), size, compressionFactor2, volume / (preGain * 2));

        workBuffer.copyInactiveBuffer(TO_HOST, samplesOut, size);
    }  else {
        int leftToProcess = size;
        for (int i = 0; i < size - windowSize; i += windowSize){
            compress(samplesIn + i, samplesOut + i, (leftToProcess > windowSize) ? windowSize : leftToProcess);
            leftToProcess -= windowSize;
        }
    }
	cudaDeviceSynchronize();
}

void Compressor::resize(uint size){
    workBuffer->resize(size);
    workBuffer->clear();
    cufftOutput->resize(size / 2 + 1);
    cufftBands->resize((size / 2 + 1) * bandCount);
    bands->resize(size * bandCount);
}

void Compressor::setWindowSize(uint size){
    windowSize = size;
    complexWindowSize = windowSize / 2 + 1;
    resize(windowSize);
    if (cufftR2C != 0) {
        cufftDestroy(cufftR2C);
    }
    if (cufftC2R != 0) {
        cufftDestroy(cufftC2R);
    }
    cufftPlan1d(&cufftR2C, windowSize, CUFFT_R2C, 1);
    cufftPlan1d(&cufftC2R, windowSize, CUFFT_C2R, bandCount);
}

void Compressor::setBandCount(uint count){
    if (bandCount == count){
        return;
    }
    cufftBands->resize(complexWindowSize * bandCount);
    bands->resize(windowSize * bandCount);
    bandCount = count;
}

void Compressor::setSampleRate(uint rate){
    sampleRate = rate;
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