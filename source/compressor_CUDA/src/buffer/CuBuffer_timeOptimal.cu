#include "CuBuffer_timeOptimal.h"

template <typename TYPE>
CuBuffer_timeOptimal<TYPE>::CuBuffer_timeOptimal() : ACuBuffer<TYPE>(){}

template <typename TYPE>
CuBuffer_timeOptimal<TYPE>::CuBuffer_timeOptimal(uint size) : ACuBuffer<TYPE>(size){}

template <typename TYPE>
CuBuffer_timeOptimal<TYPE>::CuBuffer_timeOptimal(CuBuffer_timeOptimal*& cuBuffer) : ACuBuffer<TYPE>(cuBuffer){}

template <typename TYPE>
void CuBuffer_timeOptimal<TYPE>::resize(uint size){
    if (allocatedSize >= size){
        this->size = size;
        return;
    }
    TYPE* d_newBuffer, d_oldBuffer = d_buffer;
    cudaMalloc(&d_newBuffer, size * sizeof(TYPE));
    cudaMemcpy(d_newBuffer, d_buffer, this->size * sizeof(TYPE), cudaMemcpyDeviceToDevice);
    d_buffer = d_newBuffer;
    this->size = size;
    allocatedSize = size;
    cudaFree(d_oldBuffer);
}
