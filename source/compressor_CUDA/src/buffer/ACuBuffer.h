#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// #include "ACuBufferKernels.cu"

typedef unsigned int uint;

template <typename TYPE>
class ACuBuffer{
public:
    ACuBuffer() = default;
    virtual ~ACuBuffer() = default;

    virtual uint getSize() const = 0;
    virtual uint getAllocatedSize() const = 0;
    virtual TYPE* getBuffer() const = 0;
    virtual void setBuffer(ACuBuffer<TYPE>*& cuBuffer) = 0;
    virtual void setBuffer(TYPE*& d_buffer, uint size, uint allocatedSize = 0) = 0;
    virtual void copyBuffer(const ACuBuffer<TYPE>& cuBuffer);
    virtual void copyBuffer(const TYPE*& d_buffer, uint size);
    virtual bool compare(const ACuBuffer<TYPE>& cuBuffer) const;
    virtual void clear();

    virtual ACuBuffer<TYPE>& operator=(const ACuBuffer<TYPE>& cuBuffer);
    virtual bool operator==(const ACuBuffer<TYPE>& cuBuffer) const;
    virtual bool operator!=(const ACuBuffer<TYPE>& cuBuffer) const;
    virtual TYPE* operator[](uint index) const;
    virtual operator TYPE*();
    virtual operator const TYPE*();

    virtual void resize(uint newSize) = 0;

// protected:
    uint size;
    uint allocatedSize;

    virtual void setSize(uint size) = 0;
    virtual void setAllocatedSize(uint allocatedSize) = 0;
    virtual void setBufferToNull() = 0;
    virtual TYPE* getBufferOvnership();

    virtual void allocate(uint size) = 0;
    virtual void deallocate() = 0;
};
