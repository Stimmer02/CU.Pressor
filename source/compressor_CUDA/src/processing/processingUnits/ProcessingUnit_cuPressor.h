#pragma once

#include "AProcessingUnit.h"

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

/// @brief Processing unit that increases or decrases momentary volume using softsigh-like function
class ProcessingUnit_cuPressor : public AProcessingUnit{
public:
    /// @brief Constructor
    /// @param d_workBuffer reference to the device buffer to be processed (in/out) 
    /// @param gridSize reference to calculated grid size
    /// @param blockSize reference to choosen block size
    /// @param bufferSize reference to the buffer lenght that is being processed
    ProcessingUnit_cuPressor(float*& d_workBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize);

    /// @brief Applies compression to the buffer
    void process() override;

    /// @brief Sets the compression factor
    void setCompressionFactor(float factor);

    /// @brief Returns the compression factor
    float getCompressionFactor() const;

private:
    float*& d_workBuffer;
    const uint& gridSize;
    const uint& blockSize;
    const uint& bufferSize;

    float compressionFactor;
    float processedCompressionFactor;
};