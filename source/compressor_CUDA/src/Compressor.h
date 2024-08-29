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
#include "processing/processingUnits/ProcessingUnit_cuPressorBatch.h"
#include "processing/processingUnits/ProcessingUnit_fftR2C.h"
#include "processing/processingUnits/ProcessingUnit_fftC2R.h"
#include "processing/processingUnits/ProcessingUnit_fftBandSplit.h"
#include "processing/processingUnits/ProcessingUnit_fftBandMerge.h"
#include "processing/processingUnits/ProcessingUnit_copyBuffer.h"

#include <cufft.h>

/// @brief Class that performs softsign-like multiband compression using CUDA
class COMPRESSOR_API Compressor{
public:

    /// @brief Constructor
    Compressor(const uint& bandCount = 8, const uint& windowSize = 4096, const uint& sampleRate = 44100);

    ~Compressor();

    /// @brief Executes compression on the input buffer
    /// @param samplesIn data to be processed
    /// @param samplesOut result of the processing
    /// @param size size of the input and output buffer
    /// @param channelNumber number identifying the channel that is being processed (0 - left, 1 - right)
    void compress(const float* samplesIn, float* samplesOut, const uint& size, const uint& channelNumber);

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
    void setGlobalCompressionFactor(float factor);

    /// @brief Sets compression factor for a single band
    /// @param bandIndex index of the band to set the factor for
    /// @param factor compression factor, value between 0 and 1
    void setCompressionFactor(uint bandIndex, float factor);

    /// @brief Sets compression factor for all bands
    /// @param factor compression factor, value between 0 and 1
    void setAllCompressionFactors(float factor);

    /// @brief Sets neutral point for a single band
    /// @param bandIndex index of the band to set the neutral point for
    /// @param neutralPoint the amplitude to which the signal is compressed
    void setNeutralPoint(uint bandIndex, float neutralPoint);

    /// @brief Sets neutral point for all bands
    /// @param neutralPoint the amplitude to which the signal is compressed
    void setAllNeutralPoints(float neutralPoint);

private:
    /// @brief Resizes the buffers
    /// @param size new size of the buffers
	void resize(uint size, uint complexSize = 0);

    /// @brief Calculates size of the kernels based on settings
    void calculateKernelSize();

    /// @brief Sets the amount of samples to be processed
    /// @param size amount of samples to be processed
    void setProcessingSize(uint size);

    /// @brief Sets the amount of channels to be processed in parallel
    /// @param count amount of channels
    void setChannelCount(uint count);

    /// @brief Processes a single time window
    /// @param samplesIn data to be processed
    /// @param samplesOut result of the processing
    /// @param size size of the input and output buffer
    /// @param channelNumber number identifying the channel that is being processed
    void processSingleWindow(const float* samplesIn, float* samplesOut, const uint& size, const uint& channelNumber);

    /// @brief Processes multiple time windows (if size > windowSize)
    /// @param samplesIn data to be processed
    /// @param samplesOut result of the processing
    /// @param size size of the input and output buffer
    /// @param channelNumber number identifying the channel that is being processed
    void processMultipleWindows(const float* samplesIn, float* samplesOut, const uint& size, const uint& channelNumber);

    struct {
        uint bandCount;         // amount of frequency bands
        uint processingSize;    // amount of samples to be processed
        uint windowSize;        // amound of samples to be processed by fft
        uint addressShift;      // widowSize - processingSize
        uint complexWindowSize; // amount of complex samples created by fft
        uint sampleRate;        // sample rate of the input data
    } settings;

    struct {
        uint block;
        uint gridReal;
        dim3 gridReal2D;
        uint gridComplex;
        dim3 gridComplex2D;
    } kernelSize;

    struct {
        CuShiftBuffer<float>* workBuffer;
        ACuBuffer<cufftComplex>* cufftOutput;
        ACuBuffer<cufftComplex>* cufftBands;
        ACuBuffer<float>* bands;
    } buffers;

    struct {
        float* d_workBuffer;
        float* d_workBufferCurrentPart;
        cufftComplex* d_cufftOutput;
        cufftComplex* d_cufftBands;
        float* d_bands;
        float* d_output;
    } bufferPointers;

    struct {
        cufftHandle R2C;
        cufftHandle C2R;
    } fft;

    ProcessingQueue processingQueue;
    
    struct {
        ProcessingUnit_fftR2C* fftR2C;
        ProcessingUnit_fftBandSplit* fftBandSplit;
        ProcessingUnit_cuPressorBatch* cuPressorBatch;
        ProcessingUnit_fftC2R* fftC2R;
        ProcessingUnit_fftBandMerge* fftBandMerge;
        ProcessingUnit_copyBuffer<float>* copyBuffer;
        ProcessingUnit_cuPressor* cuPressor;
        ProcessingUnit_volume* volume;
    } units;

};