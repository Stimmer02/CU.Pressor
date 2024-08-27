#pragma once

#include "ACuSimpleBuffer.h"

template <typename TYPE>
class CuSimpleBuffer_memoryOptimal : public ACuSimpleBuffer<TYPE>{
public:
    CuSimpleBuffer_memoryOptimal();
    CuSimpleBuffer_memoryOptimal(uint size);
    CuSimpleBuffer_memoryOptimal(ACuBuffer<TYPE>*& cuBuffer);

    void resize(uint newSize) override;

private:
    using ACuSimpleBuffer<TYPE>::size;
    using ACuSimpleBuffer<TYPE>::allocatedSize;
    using ACuSimpleBuffer<TYPE>::d_buffer;
};
