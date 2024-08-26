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

#include "buffer/CuShiftBuffer.h"
#include "processing/ProcessingQueue.h"
#include "processing/processingUnits/ProcessingUnit_volume.h"
#include "processing/processingUnits/ProcessingUnit_cuPressor.h"


/// @brief Class that performs softsign-like multiband compression using CUDA
class COMPRESSOR_API Compressor2{
public:

    /// @brief Constructor
    Compressor2(const uint& bandCount = 8, const uint& windowSize = 1024, const uint& sampleRate = 44100);

    ~Compressor2();

    /// @brief Executes compression on the input buffer
    /// @param samplesIn data to be processed
    /// @param samplesOut result of the processing
    /// @param size size of the input and output buffer
    void compress(float* samplesIn, float* samplesOut, uint size);

    /// @brief Sets size of the time window used for processing
    /// @param size size of the time window in samples
    void setWindowSize(uint size);

    // /// @brief Sets number of frequency bands used for processing
    // /// @param count number of frequency bands
    // void setBandCount(uint count);

    /// @brief Sets sample rate of the input data
    /// @param rate sample rate in Hz
    void setSampleRate(uint rate);

    /// @brief Sets volume correction factor
    /// @param volume volume correction factor
    void setVolume(float volume);

    /// @brief Sets compression factor
    /// @param factor compression factor, value between 0 and 1
    void setCompressionFactor(float factor);

private:
    /// @brief Resizes the buffers
    /// @param size new size of the buffers
	void resize(uint size);

    /// @brief Calculates size of the kernels based on settings
    void calculateKernelSize();

    /// @brief Sets the amount of samples to be processed
    /// @param size amount of samples to be processed
    void setProcessingSize(uint size);

    /// @brief Processes a single time window
    /// @param samplesIn data to be processed
    /// @param samplesOut result of the processing
    /// @param size size of the input and output buffer
    void processSingleWindow(float* samplesIn, float* samplesOut, uint size);

    /// @brief Processes multiple time windows (if size > windowSize)
    /// @param samplesIn data to be processed
    /// @param samplesOut result of the processing
    /// @param size size of the input and output buffer
    void processMultipleWindows(float* samplesIn, float* samplesOut, uint size);

    struct {
        uint bandCount;         // amount of frequency bands
        uint processingSize;    // amount of samples to be processed
        uint windowSize;        // amound of samples to be processed by fft
        uint addressShift;      // widowSize - processingSize
        uint complexWindowSize; // amount of complex samples created by fft
        uint sampleRate;        // sample rate of the input data
    } settings;

    struct {
        uint blockReal;
        uint gridReal;
    } kernelSize;

    struct {
        CuShiftBuffer<float>* workBuffer;// TODO: allow processing multiple channels
    } buffers;
    // ACuBuffer<cufftComplex>* cufftOutput;
    // ACuBuffer<cufftComplex>* cufftBands;
    // ACuBuffer<float>* bands;
    // cufftHandle cufftR2C;
    // cufftHandle cufftC2R;

    struct {
        float* d_workBuffer; 
    } bufferPointers;

    ProcessingQueue processingQueue;
    
    struct {
        ProcessingUnit_cuPressor* cuPressor;
        ProcessingUnit_volume* volume;
    } units;

};