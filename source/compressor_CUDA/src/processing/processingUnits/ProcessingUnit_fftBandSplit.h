#pragma once

#include "AProcessingUnit.h"
#include "../../buffer/CuBufferFactory.h"

#include <cufft.h>
#include <cmath>

typedef unsigned int uint;

/// @brief Processing unit that splits the signal into bands by their frequency
class ProcessingUnit_fftBandSplit : public AProcessingUnit{
public:
    /// @brief Constructor
    /// @param d_input single complex signal to be split into bands (in)
    /// @param d_output complex signals representing bands (out)
    /// @param gridSize2D contains grid size in x dimension and band count in y dimension
    /// @param blockSize choosen block size
    /// @param complexWindowSize size of the complex window
    ProcessingUnit_fftBandSplit(cufftComplex*& d_input, cufftComplex*& d_output, const dim3& gridSize2D, const uint& blockSize, const uint& complexWindowSize, const uint& sampleRate);

    ~ProcessingUnit_fftBandSplit();

    /// @brief Splits the signal into bands by their frequency
    void process() override;

    /// @brief Generates a mask for band splitting, use if banc count or complex window size changes
    void generateBandSplittingTable();

private:
    cufftComplex*& d_input;
    cufftComplex*& d_output;
    const dim3& gridSize2D;
    const uint& blockSize;
    const uint& complexWindowSize;
    const uint& sampleRate;

    ACuBuffer<float>* bandMasks;
};