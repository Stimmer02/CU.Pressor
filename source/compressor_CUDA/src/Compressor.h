#pragma once

#ifdef _WIN32
#ifdef COMPRESSOR_EXPORTS
#define COMPRESSOR_API __declspec(dllexport)
#else
#define COMPRESSOR_API __declspec(dllimport)
#endif
#else
#define COMPRESSOR_API
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

class COMPRESSOR_API Compressor{
public:
    Compressor();
    ~Compressor();
    void compress(float* samplesIn, float* samplesOut, int size);
    void setCompressionFactor1(double& parameter);
    void setCompressionFactor2(double& parameter);
    void setVolume(double& parameter);
    void setPreGain(double& parameter);

private:
	void allocateIfNeeded(int size);
    void setWindowSize(int size);

    int windowSize;
    int allocatedMemorySize;

    bool workBufferIndex;
    cufftReal* d_workBuffer[2];
    cufftComplex* d_cufftOutput;
    cufftHandle cufftR2C;
    cufftHandle cufftC2R;

    float compressionFactor1;
    float compressionFactor2;
    float volume;
    float preGain;
};