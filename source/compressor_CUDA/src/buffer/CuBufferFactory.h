#pragma once

#include "ACuBuffer.h"
#include "CuBuffer_memoryOptimal.h"
#include "CuBuffer_timeOptimal.h"

class CuBufferFactory{
public:
    enum class BufferType{
        MEMORY_OPTIMAL,
        TIME_OPTIMAL
    };

    template <typename TYPE>
    static ACuBuffer<TYPE>* createBuffer(BufferType bufferType, uint size = 0){
        switch(bufferType){
            case BufferType::MEMORY_OPTIMAL:
                return new CuBuffer_memoryOptimal<TYPE>(size);
            case BufferType::TIME_OPTIMAL:
                return new CuBuffer_timeOptimal<TYPE>(size);
            default:
                return nullptr;
        }
    }

    template <typename TYPE>
    static void switchBufferType(ACuBuffer<TYPE>*& cuBuffer, BufferType bufferType){
        if(cuBuffer == nullptr){
            return;
        }

        ACuBuffer<TYPE>* newCuBuffer = createBuffer(bufferType, cuBuffer->getSize());
        newCuBuffer->setBuffer(cuBuffer);
    }
};  