#pragma once

#include "AProcessingUnit.h"

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

/// @brief Processing unit that applies volume correction to the buffer
class ProcessingUnit_volume : public AProcessingUnit{
public:
    /// @brief Constructor
    /// @param d_workBuffer reference to the device buffer to be processed (in/out)
    /// @param gridSize reference to calculated grid size
    /// @param blockSize reference to choosen block size
    /// @param bufferSize reference to the buffer lenght that is being processed
    ProcessingUnit_volume(float*& d_workBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize);

    /// @brief Applies volume correction to the buffer
    void process() override;

    /// @brief Sets the volume correction factor
    void setVolume(float volume);

    /// @brief Returns the volume correction factor
    float getVolume() const;

private:
    float*& d_workBuffer;
    const uint& gridSize;
    const uint& blockSize;
    const uint& bufferSize;

    float volume;
};