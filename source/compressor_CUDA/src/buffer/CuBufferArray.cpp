#include "CuBufferArray.h"

template <typename TYPE>
__global__ void compyArrayPointer(TYPE** d_bufferPointerArr, TYPE* d_buffer, uint index, uint offset){
    d_bufferPointerArr[index] = d_buffer + offset;
}


template <typename TYPE>
CuBufferArray<TYPE>::CuBufferArray(unsigned int bufferCount, unsigned int bufferSize) : bufferCount(bufferCount){
    CuBufferFactory::createBufferArray<TYPE>(buffers, bufferCount, bufferSize);
    d_bufferPointerArr = nullptr;
    d_bufferPointerArrOffset = nullptr;
    createBufferPointerArray();
    createBufferPointerArrayOffset(0);
}

template <typename TYPE>
CuBufferArray<TYPE>::~CuBufferArray(){
    for (unsigned int i = 0; i < bufferCount; i++){
        delete buffers[i];
    }
    delete[] buffers;
}

template <typename TYPE>
TYPE* CuBufferArray<TYPE>::getBuffer(unsigned int index){
    return buffers[index]->getBuffer();
}

template <typename TYPE>
void CuBufferArray<TYPE>::resize(unsigned int bufferSize){
    for (unsigned int i = 0; i < bufferCount; i++){
        buffers[i]->resize(bufferSize);
    }
    createBufferPointerArray();
    createBufferPointerArrayOffset(0);
}

template <typename TYPE>
void CuBufferArray<TYPE>::clear(){
    for (unsigned int i = 0; i < bufferCount; i++){
        buffers[i]->clear();
    }
}

template <typename TYPE>
void CuBufferArray<TYPE>::clear(unsigned int index){
    buffers[index]->clear();
}

template <typename TYPE>
TYPE** CuBufferArray<TYPE>::getBuffers(){
    return d_bufferPointerArr;
}

template <typename TYPE>
TYPE** CuBufferArray<TYPE>::getBuffers(unsigned int addressOffset){
    if (addressOffset == 0){
        return d_bufferPointerArr;
    }
    if (arrOffset != addressOffset){
        createBufferPointerArrayOffset(addressOffset);
    }
    return d_bufferPointerArrOffset;
}

template <typename TYPE>
void CuBufferArray<TYPE>::createBufferPointerArray(){
    if (d_bufferPointerArr == nullptr){
        cudaMalloc(&d_bufferPointerArr, bufferCount * sizeof(TYPE*));
    }
    for (unsigned int i = 0; i < bufferCount; i++){
        compyArrayPointer<TYPE><<<1, 1>>>(d_bufferPointerArr, buffers[i]->getBuffer(), i, 0);
    }
    cudaDeviceSynchronize();
}   

template <typename TYPE>
void CuBufferArray<TYPE>::createBufferPointerArrayOffset(unsigned int addressOffset){
    if (d_bufferPointerArrOffset == nullptr){
        cudaMalloc(&d_bufferPointerArrOffset, bufferCount * sizeof(TYPE*));
    }
    for (unsigned int i = 0; i < bufferCount; i++){
        compyArrayPointer<TYPE><<<1, 1>>>(d_bufferPointerArrOffset, buffers[i]->getBuffer(), i, addressOffset);
    }
    arrOffset = addressOffset;
    cudaDeviceSynchronize();
}