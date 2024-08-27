#pragma once

#include "AProcessingUnit.h"

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

/// @brief Processing unit that merges set of bands into one
class ProcessingUnit_fftBandMerge : public AProcessingUnit{
public:
    /// @brief Constructor
    /// @param d_inputBuffer set of signalls to be merged (in)
    /// @param d_outputBuffer merged signall (out)
    /// @param gridSize calculated grid size
    /// @param blockSize choosen block size
    /// @param bufferSize size of buffers to be processed (length of d_inputBuffer / bandCount)
    /// @param bandCount number of bands to be merged
    /// @param addressShift distance between the start of one band and the start of the next one in d_inputBuffer
    ProcessingUnit_fftBandMerge(float*& d_inputBuffer, float*& d_outputBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize, const uint& bandCount, const uint& addressShift);

    /// @brief Merges set of bands into one
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