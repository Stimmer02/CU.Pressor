#pragma once

#include "ACuBuffer.h"

/// @brief Base class for simple CUDA GPU memory buffer wrapper
/// @tparam TYPE Data type of the buffer
template <typename TYPE>
class ACuSimpleBuffer : public ACuBuffer<TYPE>{
public:
    /// @brief Default constructor
    ACuSimpleBuffer();

    /// @brief Constructor
    /// @param size initial size of the buffer
    ACuSimpleBuffer(uint size);

    /// @brief Constructor setting passed buffer as the current buffer
    /// @param cuBuffer buffer to set (cuBuffer will be set to nullptr)
    ACuSimpleBuffer(ACuBuffer<TYPE>*& cuBuffer);

    ~ACuSimpleBuffer() override;

    /// @brief Returns the size of the buffer
    /// @return size of the buffer
    uint getSize() const override;

    /// @brief Returns the allocated size of the buffer
    /// @return maximum size of the buffer that it can be resized to without reallocation
    uint getAllocatedSize() const override;

    /// @brief Returns the pointer to the buffer
    /// @return pointer to the buffer
    TYPE* getBuffer() const override;

    /// @brief Deletes current buffer and sets the new buffer in its place
    /// @param cuBuffer buffer to set. The ownership of the buffer is transferred to the current buffer (cuBuffer will be deleted)
    void setBuffer(ACuBuffer<TYPE>*& cuBuffer) override;

    /// @brief Deletes current buffer and sets the new buffer in its place
    /// @param d_buffer device buffer to set. The ownership of the buffer is transferred to the current buffer (d_buffer will point to nullptr)
    void setBuffer(TYPE*& d_buffer, uint size, uint allocatedSize = 0) override;

protected:
    using ACuBuffer<TYPE>::size;
    using ACuBuffer<TYPE>::allocatedSize;

    TYPE* d_buffer;

    /// @brief Sets the size of the buffer without any operations on the buffer
    /// @param size new size of the size variable
    void setSize(uint size) override;

    /// @brief Sets the allocated size of the buffer without any operations on the buffer
    /// @param allocatedSize new size of the allocatedSize variable
    void setAllocatedSize(uint allocatedSize) override;

    /// @brief Sets the buffer to nullptr
    void setBufferToNull() override;

    /// @brief Deallocates memory and then allocates memory for the buffer. Does not copy the data. Ignores child class resize logic
    /// @param size size of the buffer
    void allocate(uint size) override;

    /// @brief Deallocates memory for the buffer setting it to empty
    void deallocate() override;
};