#pragma once

#include "CuBufferFactory.h"

/// @brief CUDA GPU memory buffer wrapper that allows shifting elements in the buffer
/// @tparam TYPE Data type of the buffer
/// Buffer is implemented as two buffers that are switched when shifting elements in the buffer. getInactiveBuffer() allows to still use currently unused buffer
template <typename TYPE>
class CuShiftBuffer : public ACuBuffer<TYPE>{
public:
    /// @brief Constructor
    /// @param size initial size of the buffer
    /// @param bufferType type of the buffer (specifies the buffer behavior)
    CuShiftBuffer(uint size = 0, typename CuBufferFactory::bufferType bufferType = CuBufferFactory::bufferType::TIME_OPTIMAL);

    ~CuShiftBuffer() override;
    
    /// @brief Shifts elements in the buffer by the specified amount
    /// @param shiftSize number of positions to shift the elements by
    /// (swaps the buffers)
    void shift(uint shiftSize);

    /// @brief Pushes the array to the back of the buffer shifting the elements in the buffer
    /// @param origin FROM_DEVICE or FROM_HOST - source buffer location
    /// @param arr source array
    /// @param size size of the data in the buffer to be copied
    /// (swaps the buffers)
    void pushBack(originType origin, const TYPE*& arr, uint size);

    /// @brief Returns the pointer to the inactive buffer
    /// @param index index of the buffer to return
    /// @return pointer to the inactive buffer
    /// (buffer will be used in the next shift operation)
    TYPE* getInactiveBuffer(uint index = 0) const;

    /// @brief Copies data from the inactive buffer to the destination buffer
    /// @param destination TO_DEVICE or TO_HOST - destination buffer location
    /// @param buffer destination buffer to copy the data to
    /// @param size size of the data in the buffer to be copied
    /// @param index index of the source buffer to copy the data from
    void copyInactiveBuffer(destinationType destination, TYPE* buffer, uint size = 0, uint index = 0) const;

    /// @brief Resizes the buffer
    /// @param newSize new size of the buffer
    void resize(uint newSize) override;

    /// @brief Returns the size of the buffer
    /// @return size of the buffer
    uint getSize() const override;

    /// @brief Returns the allocated size of the buffer
    /// @return maximum size of the buffer that it can be resized to without reallocation
    uint getAllocatedSize() const override;

    /// @brief Returns the pointer to the buffer
    TYPE* getBuffer() const override;

    /// @brief Deletes current buffer and sets the new buffer in its place
    /// @param cuBuffer buffer to set. The ownership of the buffer is transferred to the current buffer (cuBuffer will be deleted)
    void setBuffer(ACuBuffer<TYPE>*& cuBuffer) override;

    /// @brief Deletes current buffer and sets the new buffer in its place
    /// @param d_buffer device buffer to set. The ownership of the buffer is transferred to the current buffer (d_buffer will point to nullptr)
    /// @param size size of the data in the buffer
    /// @param allocatedSize overall size of the allocated memory
    void setBuffer(TYPE*& d_buffer, uint size, uint allocatedSize = 0) override;

private:
    using ACuBuffer<TYPE>::size;
    using ACuBuffer<TYPE>::allocatedSize;

    bool activeBuffer;
    ACuBuffer<TYPE>* buffer[2];

    /// @brief Sets the size of the buffer without any operations on the buffer
    /// @param size new size of the size variable
    void setSize(uint size) override;

    /// @brief Sets the allocated size of the buffer without any operations on the buffer
    /// @param allocatedSize new size of the allocatedSize variable
    void setAllocatedSize(uint allocatedSize) override;

    /// @brief Sets the buffer to nullptr
    void setBufferToNull() override;

    /// @brief Sets this buffer to empty one and returns prevoiusly contained buffer
    /// @return pointer to the buffer
    TYPE* getBufferOvnership() override;

    /// @brief Deallocates memory and then allocates memory for the buffer. Does not copy the data. Ignores child class resize logic
    void allocate(uint size) override;

    /// @brief Deallocates memory for the buffer setting it to empty
    void deallocate() override;
};


