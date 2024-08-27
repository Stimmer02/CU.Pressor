#pragma once

#include "CuBufferFactory.h"

// UNTESTED - I got better idea of what I am trying to do in place of this class
template <typename TYPE>
class CuBufferArray {
public:
    CuBufferArray(uint bufferCount, uint bufferSize = 0);
    ~CuBufferArray();

    void resize(uint bufferSize);
    TYPE* getBuffer(uint index);
    void clear();
    void clear(uint index);
    TYPE** getBuffers();
    TYPE** getBuffers(uint addressOffset);

private:
    const uint bufferCount;
    ACuBuffer<TYPE>** buffers;

    TYPE** d_bufferPointerArr;
    uint arrOffset;
    TYPE** d_bufferPointerArrOffset;

    void createBufferPointerArray();
    void createBufferPointerArrayOffset(unsigned int addressOffset);
};