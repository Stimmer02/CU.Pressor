#pragma once

#include "CuBufferFactory.h"

template <typename TYPE>
class CuShiftBuffer : public ACuBuffer<TYPE>{
public:
    CuShiftBuffer(uint size = 0, typename CuBufferFactory::bufferType bufferType = CuBufferFactory::bufferType::TIME_OPTIMAL);
    ~CuShiftBuffer() override;
    
    void shift(uint shiftSize);
    void put(TYPE* d_arr, uint size);
    TYPE* getInactiveBuffer() const;

    void resize(uint newSize) override;
    uint getSize() const override;
    uint getAllocatedSize() const override;
    TYPE* getBuffer() const override;

    void setBuffer(ACuBuffer<TYPE>*& cuBuffer) override;
    void setBuffer(TYPE*& d_buffer, uint size, uint allocatedSize = 0) override;

private:
    bool activeBuffer;
    ACuBuffer* buffer[2];

    void setSize(uint size) override;
    void setAllocatedSize(uint allocatedSize) override;
    void setBufferToNull() override;
    TYPE* getBufferOvnership();

    void deallocate() override;
    void allocate(uint size) override;
};


