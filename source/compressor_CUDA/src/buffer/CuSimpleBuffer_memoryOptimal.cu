#include "CuSimpleBuffer_memoryOptimal.h"

template <typename TYPE>
CuSimpleBuffer_memoryOptimal<TYPE>::CuSimpleBuffer_memoryOptimal() : ACuSimpleBuffer<TYPE>(){}

template <typename TYPE>
CuSimpleBuffer_memoryOptimal<TYPE>::CuSimpleBuffer_memoryOptimal(uint size) : ACuSimpleBuffer<TYPE>(size){}

template <typename TYPE>
CuSimpleBuffer_memoryOptimal<TYPE>::CuSimpleBuffer_memoryOptimal(ACuBuffer<TYPE>*& cuBuffer) : ACuSimpleBuffer<TYPE>(cuBuffer){}

template <typename TYPE>
void CuSimpleBuffer_memoryOptimal<TYPE>::resize(uint size){
    if (this->size == size){
        return;
    }

    TYPE* d_newBuffer;
    TYPE* d_oldBuffer = d_buffer;
    cudaMalloc((void**)&d_newBuffer, size * sizeof(TYPE));
    if (this->d_buffer != nullptr){
        if (size < this->size){
            cudaMemcpy(d_newBuffer, d_buffer, size * sizeof(TYPE), cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy(d_newBuffer, d_buffer, this->size * sizeof(TYPE), cudaMemcpyDeviceToDevice);
        }
    }
    d_buffer = d_newBuffer;
    this->size = size;
    allocatedSize = size;
    
    if (d_oldBuffer != nullptr){
        cudaFree(d_oldBuffer);
    }
}