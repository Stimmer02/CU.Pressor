#include "ACuBuffer.h"

template <typename TYPE>
__global__ void kernel_compare(TYPE* buffer1, TYPE* buffer2, uint size, bool* result){
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        if (buffer1[idx] != buffer2[idx]){
            *result = false;
        }
    }
}

template <typename TYPE>
ACuBuffer<TYPE>::ACuBuffer(){
    size = 0;
    allocatedSize = 0;
    d_buffer = nullptr;
}

template <typename TYPE>
ACuBuffer<TYPE>::ACuBuffer(uint size) : ACuBuffer(){
    allocate(size);
}

template <typename TYPE>
ACuBuffer<TYPE>::ACuBuffer(ACuBuffer*& cuBuffer){
    setBuffer(cuBuffer);
}

template <typename TYPE>
ACuBuffer<TYPE>::~ACuBuffer(){
    deallocate();
}

template <typename TYPE>
uint ACuBuffer<TYPE>::getSize(){
    return size;
}

template <typename TYPE>
TYPE* ACuBuffer<TYPE>::getBuffer(){
    return d_buffer;
}

template <typename TYPE>
void ACuBuffer<TYPE>::clear(){
    cudaMemset(d_buffer, 0, size * sizeof(TYPE));
}

template <typename TYPE>
void ACuBuffer<TYPE>::setBuffer(ACuBuffer*& cuBuffer){
    deallocate();
    size = cuBuffer->size;
    allocatedSize = cuBuffer->allocatedSize;
    d_buffer = cuBuffer->d_buffer;

    cuBuffer->allocatedSize = 0;
    cuBuffer->size = 0;
    cuBuffer->d_buffer = nullptr;

    delete cuBuffer;
    cuBuffer = nullptr;
}

template <typename TYPE>
void ACuBuffer<TYPE>::setBuffer(TYPE*& buffer, uint size, uint allocatedSize = 0){
    deallocate();
    this->size = size;
    this->allocatedSize = allocatedSize != 0 ? allocatedSize : size;
    d_buffer = buffer;
    buffer = nullptr;
}

template <typename TYPE>
void ACuBuffer<TYPE>::copyBuffer(const ACuBuffer<TYPE>*& cuBuffer){
    if (this != &cuBuffer) {
        if (cuBuffer.d_buffer == nullptr){
            deallocate();
        }
        resize(cuBuffer.size);
        cudaMemcpy(d_buffer, cuBuffer.d_buffer, size * sizeof(TYPE), cudaMemcpyDeviceToDevice);
    }
}

template <typename TYPE>
void ACuBuffer<TYPE>::copyBuffer(const TYPE*& buffer, uint size){
    resize(size);
    cudaMemcpy(d_buffer, buffer, size * sizeof(TYPE), cudaMemcpyHostToDevice);
}

template <typename TYPE>
bool ACuBuffer<TYPE>::compare(const ACuBuffer<TYPE>*& cuBuffer){
    if (size != cuBuffer.size){
        return false;
    }

    bool* d_result;
    cudaMalloc(&d_result, sizeof(bool));
    kernel_compare<<<(size + 255) / 256, 256>>>(d_buffer, cuBuffer.d_buffer, size, d_result);

    bool result;
    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}

template <typename TYPE>
ACuBuffer<TYPE>& ACuBuffer<TYPE>::operator=(const ACuBuffer<TYPE>& cuBuffer){
    copyBuffer(cuBuffer);
    return *this;
}

template <typename TYPE>
bool ACuBuffer<TYPE>::operator==(const ACuBuffer<TYPE>& cuBuffer){
    return compare(cuBuffer);
}

template <typename TYPE>
bool ACuBuffer<TYPE>::operator!=(const ACuBuffer<TYPE>& cuBuffer){
    return !compare(cuBuffer);
}

template <typename TYPE>
TYPE* ACuBuffer<TYPE>::operator[](uint index){
    return d_buffer + index;
}

template <typename TYPE>
TYPE* ACuBuffer<TYPE>::operator()(){
    return d_buffer;
}

template <typename TYPE>
void ACuBuffer<TYPE>::allocate(uint size){
    deallocate();
    cudaMalloc(&d_buffer, size * sizeof(TYPE));
    this->size = size;
    allocatedSize = size;
}

template <typename TYPE>
void ACuBuffer<TYPE>::deallocate(){
    if (d_buffer != nullptr){
        cudaFree(d_buffer);
        d_buffer = nullptr;
    }
    size = 0;
    allocatedSize = 0;
}