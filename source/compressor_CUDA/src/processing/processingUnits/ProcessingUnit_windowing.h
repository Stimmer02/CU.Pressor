#pragma once

#include "AProcessingUnit.h"
#include "../../buffer/CuBufferFactory.h"

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

/// @brief Processing unit that applies Plank-taper window to the buffer
class ProcessingUnit_windowing : public AProcessingUnit{
public:
    /// @brief Constructor
    /// @param d_inputBuffer reference to the device buffer to be processed (in)
    /// @param d_outputBuffer reference to the device buffer to store the result (out)
    /// @param gridSize reference to calculated grid size
    /// @param blockSize reference to choosen block size
    /// @param bufferSize reference to the buffer lenght that is being processed
    ProcessingUnit_windowing(float*& d_inputBuffer, float*& d_outputBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize);

    ~ProcessingUnit_windowing();

    /// @brief Applies Plank-taper window to the buffer
    void process() override;

    /// @brief Generates Plank-taper window unig bufferSize reference passed in the constructor
    /// @param startLength how many samples from the start of the buffer should be windowed
    /// @param endLength how many samples from the end of the buffer should be windowed
    void generateWindow(int startLength, int endLength);

private:
    float*& d_inputBuffer;
    float*& d_outputBuffer;
    const uint& gridSize;
    const uint& blockSize;
    const uint& bufferSize;

    ACuBuffer<float>* windowBuffer;
};