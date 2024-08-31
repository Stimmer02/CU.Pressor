#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// #include "ACuBufferKernels.cu"

typedef unsigned int uint;

// namespace cuBuffer{
    enum destinationType{
        TO_DEVICE = cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        TO_HOST = cudaMemcpyKind::cudaMemcpyDeviceToHost,
    };

    enum originType{
        FROM_DEVICE = cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        FROM_HOST = cudaMemcpyKind::cudaMemcpyHostToDevice,
    };
// }

/// @brief Base class for CUDA GPU buffer wrapper
/// @tparam TYPE Data type of the buffer
template <typename TYPE>
class ACuBuffer{
public:
    /// @brief Constructor
    ACuBuffer() = default;
    virtual ~ACuBuffer() = default;

    /// @brief Returns the size of the buffer
    /// @return size of the buffer
    virtual uint getSize() const = 0;

    /// @brief Returns the allocated size of the buffer
    /// @return maximum size of the buffer that it can be resized to without reallocation
    virtual uint getAllocatedSize() const = 0;

    /// @brief Returns the pointer to the buffer
    /// @return pointer to the buffer
    virtual TYPE* getBuffer() const = 0;

    /// @brief Deletes current buffer and sets the new buffer in its place
    /// @param cuBuffer buffer to set. The ownership of the buffer is transferred to the current buffer (cuBuffer will be deleted)
    virtual void setBuffer(ACuBuffer<TYPE>*& cuBuffer) = 0;

    /// @brief Deletes current buffer and sets the new buffer in its place
    /// @param d_buffer device buffer to set. The ownership of the buffer is transferred to the current buffer (d_buffer will point to nullptr)
    /// @param size size of the data in the buffer
    /// @param allocatedSize overall size of the allocated memory
    virtual void setBuffer(TYPE*& d_buffer, uint size, uint allocatedSize = 0) = 0;

    /// @brief Copies data from source buffer to the current buffer resizing it to the source buffer size
    /// @param cuBuffer source buffer
    virtual void copyBuffer(const ACuBuffer<TYPE>& cuBuffer);

    /// @brief Copies data from source buffer to the current buffer resizing it to the source buffer size
    /// @param origin FROM_DEVICE or FROM_HOST - source buffer location
    /// @param buffer source buffer
    /// @param size size of the data in the buffer to be copied
    /// @param index index of the destination buffer to copy the data to
    virtual void copyBuffer(originType origin, const TYPE* buffer, uint size, uint index = 0);
    
    /// @brief Copies data from the current buffer to the destination buffer
    /// @param destination TO_DEVICE or TO_HOST - destination buffer location
    /// @param buffer destination buffer
    /// @param size size of the data in the buffer to be copied
    /// @param index index of the source buffer to copy the data from
    virtual void copyBuffer(destinationType destination, TYPE* buffer, uint size = 0, uint index = 0) const;

    /// @brief Compares the current buffer with the other buffer
    /// @param cuBuffer other buffer to compare with
    /// @return true if the buffers contain the same data, false otherwise
    virtual bool compare(const ACuBuffer<TYPE>& cuBuffer) const;

    /// @brief Sets the buffer to zeros
    virtual void clear();


    /// @brief Copies source buffer to the current buffer
    /// @param cuBuffer source buffer
    /// @return reference to the current buffer
    virtual ACuBuffer<TYPE>& operator=(const ACuBuffer<TYPE>& cuBuffer);

    /// @brief Compares the current buffer with the other buffer
    /// @param cuBuffer other buffer to compare with
    /// @return true if the buffers contain the same data, false otherwise
    virtual bool operator==(const ACuBuffer<TYPE>& cuBuffer) const;

    /// @brief Compares the current buffer with the other buffer
    /// @param cuBuffer other buffer to compare with
    /// @return true if the buffers contain different data, false otherwise
    virtual bool operator!=(const ACuBuffer<TYPE>& cuBuffer) const;

    /// @brief Returns the pointer to the buffer
    /// @param index index of the element in the buffer
    /// @return pointer to the element in the buffer
    virtual TYPE* operator[](uint index) const;

    /// @brief Returns the pointer to the buffer
    /// @return pointer to the buffer
    virtual operator TYPE*();

    /// @brief Returns the pointer to the buffer
    virtual operator const TYPE*();


    /// @brief Resizes the buffer using child class logic
    /// @param newSize new size of the buffer
    virtual void resize(uint newSize) = 0;

protected:
    uint size;
    uint allocatedSize;


    /// @brief Sets the size of the buffer without any operations on the buffer
    /// @param size new size of the size variable
    virtual void setSize(uint size) = 0;

    /// @brief Sets the allocated size of the buffer without any operations on the buffer
    /// @param allocatedSize new size of the allocatedSize variable
    virtual void setAllocatedSize(uint allocatedSize) = 0;

    /// @brief Sets the buffer to nullptr
    virtual void setBufferToNull() = 0;

    /// @brief Sets this buffer to empty one and returns prevoiusly contained buffer
    /// @return pointer to the buffer
    virtual TYPE* getBufferOvnership();


    /// @brief Deallocates memory and then allocates memory for the buffer. Does not copy the data. Ignores child class resize logic
    /// @param size size of the buffer
    virtual void allocate(uint size) = 0;

    /// @brief Deallocates memory for the buffer setting it to empty
    virtual void deallocate() = 0;
};
