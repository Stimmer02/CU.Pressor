#include "CuBuffer_memoryOptimal.h"

template <typename TYPE>
CuBuffer_memoryOptimal<TYPE>::CuBuffer_memoryOptimal() : ACuBuffer<TYPE>(){}

template <typename TYPE>
CuBuffer_memoryOptimal<TYPE>::CuBuffer_memoryOptimal(uint size) : ACuBuffer<TYPE>(size){}

template <typename TYPE>
CuBuffer_memoryOptimal<TYPE>::CuBuffer_memoryOptimal(CuBuffer_memoryOptimal*& cuBuffer) : ACuBuffer<TYPE>(cuBuffer){}

template <typename TYPE>
void CuBuffer_memoryOptimal<TYPE>::resize(uint size){
    if (this->size == size){
        return;
    }

    TYPE* d_newBuffer, d_oldBuffer = d_buffer;
    cudaMalloc(&d_newBuffer, size * sizeof(TYPE));
    if (size < this->size){
        cudaMemcpy(d_newBuffer, d_buffer, size * sizeof(TYPE), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemcpy(d_newBuffer, d_buffer, this->size * sizeof(TYPE), cudaMemcpyDeviceToDevice);
    }
    d_buffer = d_newBuffer;
    this->size = size;
    allocatedSize = size;
    cudaFree(d_oldBuffer);
}
