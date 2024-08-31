#pragma once

#include "ACuSimpleBuffer.h"

/// @brief Simple CUDA GPU memory buffer wrapper with optimisation focused on maximal preservation of memory 
/// @tparam TYPE 
template <typename TYPE>
class CuSimpleBuffer_memoryOptimal : public ACuSimpleBuffer<TYPE>{
public:
    /// @brief Default constructor
    CuSimpleBuffer_memoryOptimal();

    /// @brief Constructor
    /// @param size initial size of the buffer
    CuSimpleBuffer_memoryOptimal(uint size);

    /// @brief Constructor setting passed buffer as the current buffer
    /// @param cuBuffer buffer to set (cuBuffer will be set to nullptr)
    CuSimpleBuffer_memoryOptimal(ACuBuffer<TYPE>*& cuBuffer);

    /// @brief Resizes the buffer exactly to the newSize reasulting in the smallest possible memory allocation
    /// @param newSize new size of the buffer
    void resize(uint newSize) override;

private:
    using ACuSimpleBuffer<TYPE>::size;
    using ACuSimpleBuffer<TYPE>::allocatedSize;
    using ACuSimpleBuffer<TYPE>::d_buffer;
};
