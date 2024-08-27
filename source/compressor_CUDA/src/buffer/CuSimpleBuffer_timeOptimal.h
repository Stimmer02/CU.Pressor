#pragma once

#include "ACuSimpleBuffer.h"

template <typename TYPE>
class CuSimpleBuffer_timeOptimal : public ACuSimpleBuffer<TYPE>{
public:
    CuSimpleBuffer_timeOptimal();
    CuSimpleBuffer_timeOptimal(uint size);
    CuSimpleBuffer_timeOptimal(ACuBuffer<TYPE>*& cuBuffer);

    void resize(uint newSize) override;

private:
    using ACuSimpleBuffer<TYPE>::size;
    using ACuSimpleBuffer<TYPE>::allocatedSize;
    using ACuSimpleBuffer<TYPE>::d_buffer;
};

