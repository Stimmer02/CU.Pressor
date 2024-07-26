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

class COMPRESSOR_API Compressor{
public:
    Compressor();
    ~Compressor();
    void compress(float* samplesIn, float* samplesOut, int size);
    void setCompressionFactor(double& parameter);

private:
	void allocateIfNeeded(int size);
    int gpuBufferSize;
	float* d_buffer;

    float compressionFactor;
};