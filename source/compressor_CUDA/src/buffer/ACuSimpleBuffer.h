#pragma once

#include "ACuBuffer.h"

template <typename TYPE>
class ACuSimpleBuffer : public ACuBuffer<TYPE>{
public:
    ACuSimpleBuffer();
    ACuSimpleBuffer(uint size);
    ACuSimpleBuffer(ACuBuffer<TYPE>*& cuBuffer);
    ~ACuSimpleBuffer() override;

    uint getSize() const override;
    uint getAllocatedSize() const override;
    TYPE* getBuffer() const override;
    void setBuffer(ACuBuffer<TYPE>*& cuBuffer) override;
    void setBuffer(TYPE*& d_buffer, uint size, uint allocatedSize = 0) override;

protected:
    using ACuBuffer<TYPE>::size;
    using ACuBuffer<TYPE>::allocatedSize;

    TYPE* d_buffer;

    void setSize(uint size) override;
    void setAllocatedSize(uint allocatedSize) override;
    void setBufferToNull() override;

    void deallocate() override;
    void allocate(uint size) override;
};