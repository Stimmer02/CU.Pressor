#pragma once

#include "AProcessingUnit.h"
#include "../../buffer/CuBufferFactory.h"

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

/// @brief Processing unit that increases or decrases momentary volume using softsigh-like function
class ProcessingUnit_cuPressorBatch : public AProcessingUnit{
public:
    /// @brief Constructor
    /// @param d_workBuffer reference to the device buffer to be processed (in/out)
    /// @param gridSize reference to calculated grid size 
    /// @param blockSize reference to choosen block size
    /// @param bufferSize reference to the buffer lenght that is being processed
    /// @param bandCount number of bands to be processed
    /// @param addressShift how many elements should be skipped per band (bandBufferSize = bufferSize + addressShift)
    ProcessingUnit_cuPressorBatch(float*& d_workBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize, const uint bandCount, const uint& addressShift);

    ~ProcessingUnit_cuPressorBatch();

    /// @brief Applies compression to the buffer
    void process() override;

    /// @brief Sets the compression factor
    /// @param bandIndex index of the band to set the factor for
    /// @param factor compression factor
    void setCompressionFactor(uint bandIndex, float factor);

    /// @brief Sets all compression factors to the same value
    /// @param factor compression factor
    void setAllCompressionFactors(float factor);

    /// @brief Returns the compression factor
    /// @param bandIndex index of the band to get the factor for
    /// @return compression factor or 0 if the band index is out of range
    float getCompressionFactor(uint bandIndex) const;

    /// @brief Sets the neutral point of a band
    /// @param bandIndex index of the band to set the neutral point for
    /// @param neutralPoint neutral point - the amplitude to which the signal is compressed
    void setNeutralPoint(uint bandIndex, float neutralPoint);

    /// @brief Sets all neutral points to the same value
    /// @param neutralPoint neutral point - the amplitude to which the signal is compressed
    void setAllNeutralPoints(float neutralPoint);

    /// @brief Returns the neutral point of a band
    /// @param bandIndex index of the band to get the neutral point for
    /// @return neutral point or 0 if the band index is out of range
    float getNeutralPoint(uint bandIndex) const;

private:
    float*& d_workBuffer;
    const uint& gridSize;
    const uint& blockSize;
    const uint& bufferSize;
    const uint& addressShift;

    const uint bandCount;
    uint activeFactors;
    float* compressionFactors;
    ACuBuffer<float>* factorsBuffer;
    float* neutralPoints;
    ACuBuffer<float>* neutralPointsBuffer;
};