#pragma once

#include "CuSimpleBuffer_memoryOptimal.h"
#include "CuSimpleBuffer_timeOptimal.h"

/// @brief Factory class for creating CUDA buffers
namespace CuBufferFactory{

    enum bufferType{
        MEMORY_OPTIMAL,
        TIME_OPTIMAL
    };

    /// @brief Creates a new buffer of the specified type
    /// @tparam TYPE data type of the buffer
    /// @param size initial size of the buffer
    /// @param bufferType type of the buffer (specifies the buffer behavior)
    /// @return pointer to the created buffer
    template <typename TYPE>
    ACuBuffer<TYPE>* createBuffer(uint size = 0, CuBufferFactory::bufferType bufferType = bufferType::TIME_OPTIMAL);

    /// @brief Fills the array of buffers with buffers of the specified type
    /// @tparam TYPE data type of the buffer
    /// @param cuBuffers array of buffers to fill
    /// @param count number of buffers to fill
    /// @param size initial size of the buffers
    /// @param bufferType type of the buffers (specifies the buffer behavior)
    template <typename TYPE>
    void fillBufferArray(ACuBuffer<TYPE>** cuBuffers, uint count, uint size = 0, CuBufferFactory::bufferType bufferType = bufferType::TIME_OPTIMAL);

    /// @brief Creates an array of buffers of the specified type
    /// @tparam TYPE data type of the buffer
    /// @param cuBuffers array of buffers to create
    /// @param count number of buffers to create
    /// @param size initial size of the buffers
    /// @param bufferType type of the buffers (specifies the buffer behavior)
    template <typename TYPE>
    void createBufferArray(ACuBuffer<TYPE>** cuBuffers, uint count, uint size = 0, CuBufferFactory::bufferType bufferType = bufferType::TIME_OPTIMAL);

    /// @brief Switches the buffer type of the buffer
    /// @tparam TYPE data type of the buffer
    /// @param cuBuffer buffer to switch
    /// @param bufferType type of the buffer (specifies the buffer behavior)
    template <typename TYPE>
    void switchBufferType(ACuBuffer<TYPE>*& cuBuffer, CuBufferFactory::bufferType bufferType);
};  


using namespace CuBufferFactory;

template <typename TYPE>
static ACuBuffer<TYPE>* CuBufferFactory::createBuffer(uint size, CuBufferFactory::bufferType bufferType){
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
void CuBufferFactory::fillBufferArray(ACuBuffer<TYPE>** cuBuffers, uint count, uint size, CuBufferFactory::bufferType bufferType){
    for (uint i = 0; i < count; i++){
        cuBuffers[i] = createBuffer<TYPE>(size, bufferType);
    }
}

template <typename TYPE>
void CuBufferFactory::createBufferArray(ACuBuffer<TYPE>** cuBuffers, uint count, uint size, CuBufferFactory::bufferType bufferType){
    cuBuffers = new ACuBuffer<TYPE>*[count];
    fillBufferArray<TYPE>(cuBuffers, count, size, bufferType);
}

template <typename TYPE>
void CuBufferFactory::switchBufferType(ACuBuffer<TYPE>*& cuBuffer, CuBufferFactory::bufferType bufferType){
    if (cuBuffer == nullptr){
        return;
    }

    ACuBuffer<TYPE>* newCuBuffer = createBuffer(bufferType, cuBuffer->getSize());
    newCuBuffer->setBuffer(cuBuffer);
}
