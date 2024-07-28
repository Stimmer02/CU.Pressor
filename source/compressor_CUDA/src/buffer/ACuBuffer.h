#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

template <typename TYPE>
class ACuBuffer{
public:
    ACuBuffer();
    ACuBuffer(uint size);
    ACuBuffer(ACuBuffer*& cuBuffer);
    virtual ~ACuBuffer();

    virtual uint getSize();
    virtual TYPE* getBuffer();
    virtual void clear();
    virtual void setBuffer(ACuBuffer<TYPE>*& cuBuffer);
    virtual void setBuffer(TYPE*& buffer, uint size, uint allocatedSize = 0);
    virtual void copyBuffer(const ACuBuffer<TYPE>*& cuBuffer);
    virtual void copyBuffer(const TYPE*& buffer, uint size);
    virtual bool compare(const ACuBuffer<TYPE>*& cuBuffer);

    virtual ACuBuffer<TYPE>& operator=(const ACuBuffer<TYPE>& cuBuffer);
    virtual bool operator==(const ACuBuffer<TYPE>& cuBuffer);
    virtual bool operator!=(const ACuBuffer<TYPE>& cuBuffer);
    virtual TYPE* operator[](uint index);
    virtual TYPE* operator()();

    virtual void resize(uint newSize) = 0;

protected:
    friend class ACuBuffer;

    uint size;
    uint allocatedSize;
    TYPE* d_buffer;

    virtual void allocate(uint size);
    virtual void deallocate();
};