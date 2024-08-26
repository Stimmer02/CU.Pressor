#pragma once

#include "AProcessingUnit.h"

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

/// @brief Processing unit that merges set of bands into one
class ProcessingUnit_fftBandMerge : public AProcessingUnit{
public:

    ProcessingUnit_fftBandMerge(float*& d_inputBuffer, float*& d_outputBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize, const uint& bandCount, const uint& addressShift);

    /// @brief Applies compression to the buffer
    void process() override;


private:
    float*& d_inputBuffer;
    float*& d_outputBuffer;
    const uint& gridSize;
    const uint& blockSize;
    const uint& bufferSize;
    const uint& addressShift;
    const uint bandCount;
};