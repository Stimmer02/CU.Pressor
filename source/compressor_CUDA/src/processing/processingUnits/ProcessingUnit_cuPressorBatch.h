#pragma once

#include "AProcessingUnit.h"

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

/// @brief Processing unit that increases or decrases momentary volume using softsigh-like function
class ProcessingUnit_cuPressorBatch : public AProcessingUnit{
public:
    /// @brief Constructor
    /// @param d_workBuffer reference to the device buffer to be processed (in/out) 
    /// @param blockSize reference to choosen block size
    /// @param bufferSize reference to the buffer lenght that is being processed
    ProcessingUnit_cuPressorBatch(float*& d_workBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize, const uint bandCount, const uint& addressShift);

    ~ProcessingUnit_cuPressorBatch();

    /// @brief Applies compression to the buffer
    void process() override;

    /// @brief Sets the compression factor
    /// @param bandIndex index of the band to set the factor for
    /// @param factor compression factor
    void setCompressionFactor(uint bandIndex, float factor);

    /// @brief Returns the compression factor
    /// @param bandIndex index of the band to get the factor for
    /// @return compression factor or inf if the band index is out of range
    float getCompressionFactor(uint bandIndex) const;

    /// @brief Sets the pre gain factor
    void setPreGain(float gain);

    /// @brief Returns the pre gain factor
    float getPreGain() const;

private:
    float*& d_workBuffer;
    const uint& gridSize;
    const uint& blockSize;
    const uint& bufferSize;
    const uint& addressShift;

    const uint bandCount;
    float* compressionFactors;
    float* d_compressionFactors;
    uint activeFactors;
    float preGain;
};