#pragma once

#include "CuSimpleBuffer_memoryOptimal.h"
#include "CuSimpleBuffer_timeOptimal.h"

namespace CuBufferFactory{

    enum bufferType{
        MEMORY_OPTIMAL,
        TIME_OPTIMAL
    };
    template <typename TYPE>
    ACuBuffer<TYPE>* createBuffer(uint size = 0, typename bufferType bufferType = bufferType::TIME_OPTIMAL);
    template <typename TYPE>
    void fillBufferArray(ACuBuffer<TYPE>** cuBuffers, uint count, uint size = 0, typename bufferType bufferType = bufferType::TIME_OPTIMAL);
    template <typename TYPE>
    void createBufferArray(ACuBuffer<TYPE>** cuBuffers, uint count, uint size = 0, typename bufferType bufferType = bufferType::TIME_OPTIMAL);
    template <typename TYPE>
    void switchBufferType(ACuBuffer<TYPE>*& cuBuffer, typename bufferType bufferType);
};  


using namespace CuBufferFactory;

template <typename TYPE>
static ACuBuffer<TYPE>* CuBufferFactory::createBuffer<TYPE>(uint size, typename bufferType bufferType){
    switch(bufferType){
        case bufferType::MEMORY_OPTIMAL:
            return new CuSimpleBuffer_memoryOptimal<TYPE>(size);
        case bufferType::TIME_OPTIMAL:
            return new CuSimpleBuffer_timeOptimal<TYPE>(size);
        default:
            return nullptr;
    }
}

template <typename TYPE>
void CuBufferFactory::fillBufferArray<TYPE>(ACuBuffer<TYPE>** cuBuffers, uint count, uint size, typename bufferType bufferType){
    for (uint i = 0; i < count; i++){
        cuBuffers[i] = createBuffer<TYPE>(size, bufferType);
    }
}

template <typename TYPE>
void CuBufferFactory::createBufferArray<TYPE>(ACuBuffer<TYPE>** cuBuffers, uint count, uint size, typename bufferType bufferType){
    cuBuffers = new ACuBuffer<TYPE>*[count];
    fillBufferArray<TYPE>(cuBuffers, count, size, bufferType);
}

template <typename TYPE>
void CuBufferFactory::switchBufferType<TYPE>(ACuBuffer<TYPE>*& cuBuffer, typename bufferType bufferType){
    if (cuBuffer == nullptr){
        return;
    }

    ACuBuffer<TYPE>* newCuBuffer = createBuffer(bufferType, cuBuffer->getSize());
    newCuBuffer->setBuffer(cuBuffer);
}
