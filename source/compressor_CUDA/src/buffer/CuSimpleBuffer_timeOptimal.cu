#include "CuSimpleBuffer_timeOptimal.h"

template <typename TYPE>
CuSimpleBuffer_timeOptimal<TYPE>::CuSimpleBuffer_timeOptimal() : ACuSimpleBuffer<TYPE>(){}

template <typename TYPE>
CuSimpleBuffer_timeOptimal<TYPE>::CuSimpleBuffer_timeOptimal(uint size) : ACuSimpleBuffer<TYPE>(size){}

template <typename TYPE>
CuSimpleBuffer_timeOptimal<TYPE>::CuSimpleBuffer_timeOptimal(ACuBuffer<TYPE>*& cuBuffer) : ACuSimpleBuffer<TYPE>(cuBuffer){}

template <typename TYPE>
void CuSimpleBuffer_timeOptimal<TYPE>::resize(uint size){
    if (allocatedSize >= size){
        this->size = size;
        return;
    }
    TYPE* d_newBuffer;
    TYPE* d_oldBuffer = d_buffer;
    cudaMalloc((void**)&d_newBuffer, size * sizeof(TYPE));
    if (d_buffer != nullptr){
        cudaMemcpy(d_newBuffer, d_buffer, this->size * sizeof(TYPE), cudaMemcpyDeviceToDevice);
    }
    d_buffer = d_newBuffer;
    this->size = size;
    allocatedSize = size;
    if (d_oldBuffer != nullptr){
        cudaFree(d_oldBuffer);
    }
}