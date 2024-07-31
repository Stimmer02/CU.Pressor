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

#include <cufft.h>

#include "buffer/CuShiftBuffer.h"

class COMPRESSOR_API Compressor{
public:
    Compressor();
    ~Compressor();
    void compress(float* samplesIn, float* samplesOut, uint size);
    void setCompressionFactor1(double& parameter);
    void setCompressionFactor2(double& parameter);
    void setVolume(double& parameter);
    void setPreGain(double& parameter);
    void setWindowSize(uint size);
    void setBandCount(uint count);
    void setSampleRate(uint rate);

private:
	void resize(uint size);
    void generateBandBoundaries();

    uint windowSize;
    uint complexWindowSize;
    uint bandCount;

    CuShiftBuffer<cufftReal>* workBuffer;
    ACuBuffer<cufftComplex>* cufftOutput;
    ACuBuffer<cufftComplex>* cufftBands;
    ACuBuffer<cufftReal>* bands;
    ACuBuffer<int>* bandBoundaries;
    cufftHandle cufftR2C;
    cufftHandle cufftC2R;

    float compressionFactor1;
    float compressionFactor2;
    float volume;
    float preGain;

    uint sampleRate;
};