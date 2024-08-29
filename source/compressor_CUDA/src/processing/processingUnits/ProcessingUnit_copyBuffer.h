#pragma once

#include "AProcessingUnit.h"

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

/// @brief Processing unit that copies data from one buffer to another
template <typename TYPE>
class ProcessingUnit_copyBuffer : public AProcessingUnit{
public:


    /// @brief Constructor
    /// @param source source buffer
    /// @param destination destination buffer
    /// @param size size of the buffer
    /// @param kind type of the copy operation
    ProcessingUnit_copyBuffer(const TYPE*& source, TYPE*& destination, uint& size, const enum cudaMemcpyKind kind);

    /// @brief Copies the data from source to destination
    void process() override;
    
private:
    const TYPE*& source;
    TYPE*& destination;
    uint& size;

    const enum cudaMemcpyKind kind;
};

template <typename TYPE>
ProcessingUnit_copyBuffer<TYPE>::ProcessingUnit_copyBuffer(const TYPE*& source, TYPE*& destination, uint& size, const enum cudaMemcpyKind kind) : source(source), destination(destination), size(size), kind(kind){}

template <typename TYPE>
void ProcessingUnit_copyBuffer<TYPE>::process(){
    cudaMemcpy(destination, source, size * sizeof(TYPE), kind);
}
