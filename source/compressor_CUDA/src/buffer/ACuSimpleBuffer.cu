#include "ACuSimpleBuffer.h"

template <typename TYPE>
ACuSimpleBuffer<TYPE>::ACuSimpleBuffer(){
    setSize(0);
    setAllocatedSize(0);
    setBufferToNull();
}

template <typename TYPE>
ACuSimpleBuffer<TYPE>::ACuSimpleBuffer(uint size) : ACuSimpleBuffer(){
    allocate(size);
}

template <typename TYPE>
ACuSimpleBuffer<TYPE>::ACuSimpleBuffer(ACuBuffer<TYPE>*& cuBuffer) : ACuSimpleBuffer(){
    setBuffer(cuBuffer);
}

template <typename TYPE>
ACuSimpleBuffer<TYPE>::~ACuSimpleBuffer(){
    deallocate();
}

template <typename TYPE>
void ACuSimpleBuffer<TYPE>::allocate(uint size){
    deallocate();
    if (size == 0){
        return;
    }
    cudaMalloc(&d_buffer, size * sizeof(TYPE));
    setSize(size);
    setAllocatedSize(size);
}

template <typename TYPE>
void ACuSimpleBuffer<TYPE>::deallocate(){
    if (d_buffer != nullptr){
        cudaFree(d_buffer);
        d_buffer = nullptr;
    }
    setSize(0);
    setAllocatedSize(0);
}

template <typename TYPE>
void ACuSimpleBuffer<TYPE>::setBuffer(ACuBuffer<TYPE>*& cuBuffer){
    deallocate();
    setSize(cuBuffer->getSize());
    setAllocatedSize(cuBuffer->getAllocatedSize());
    d_buffer = static_cast<ACuSimpleBuffer<TYPE>*>(cuBuffer)->getBufferOvnership();

    delete cuBuffer;
    cuBuffer = nullptr;
}

template <typename TYPE>
void ACuSimpleBuffer<TYPE>::setBuffer(TYPE*& d_buffer, uint size, uint allocatedSize){
    deallocate();
    setSize(size);
    setSize(allocatedSize != 0 ? allocatedSize : size);
    this->d_buffer = d_buffer;
    d_buffer = nullptr;
}

template <typename TYPE>
void ACuSimpleBuffer<TYPE>::setSize(uint size){
    this->size = size;
}

template <typename TYPE>
void ACuSimpleBuffer<TYPE>::setAllocatedSize(uint allocatedSize){
    this->allocatedSize = allocatedSize;
}

template <typename TYPE>
uint ACuSimpleBuffer<TYPE>::getSize() const {
    return size;
}

template <typename TYPE>
uint ACuSimpleBuffer<TYPE>::getAllocatedSize() const {
    return allocatedSize;
}

template <typename TYPE>
TYPE* ACuSimpleBuffer<TYPE>::getBuffer() const {
    return d_buffer;
}

template <typename TYPE>
void ACuSimpleBuffer<TYPE>::setBufferToNull(){
    d_buffer = nullptr;
}