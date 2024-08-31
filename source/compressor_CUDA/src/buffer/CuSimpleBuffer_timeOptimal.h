#pragma once

#include "ACuSimpleBuffer.h"

/// @brief Simple CUDA GPU memory buffer wrapper with optimisation focused on maximal preservation of time - buffer will never be resized to a smaller size always keeping the allocated memory
template <typename TYPE>
class CuSimpleBuffer_timeOptimal : public ACuSimpleBuffer<TYPE>{
public:
    /// @brief Default constructor
    CuSimpleBuffer_timeOptimal();

    /// @brief Constructor
    /// @param size initial size of the buffer
    CuSimpleBuffer_timeOptimal(uint size);

    /// @brief Constructor setting passed buffer as the current buffer
    /// @param cuBuffer buffer to set (cuBuffer will be set to nullptr)
    CuSimpleBuffer_timeOptimal(ACuBuffer<TYPE>*& cuBuffer);

    /// @brief Resizes the buffer to the newSize only if the newSize is larger than the current size (allocatedSize will never be decreased)
    void resize(uint newSize) override;

private:
    using ACuSimpleBuffer<TYPE>::size;
    using ACuSimpleBuffer<TYPE>::allocatedSize;
    using ACuSimpleBuffer<TYPE>::d_buffer;
};

