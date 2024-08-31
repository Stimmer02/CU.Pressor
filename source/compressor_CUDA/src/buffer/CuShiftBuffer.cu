#include "CuShiftBuffer.h"

template <typename TYPE>
__global__ void kernel_shiftData(TYPE* inputBuffer, TYPE* outputBuffer, int size, int shift){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // For optimal thread usage, max idx should be size - shift
    if (idx < size - shift){
        outputBuffer[idx] = inputBuffer[idx + shift];
    }
}


template <typename TYPE>
CuShiftBuffer<TYPE>::CuShiftBuffer(uint size, typename CuBufferFactory::bufferType bufferType){
    CuBufferFactory::fillBufferArray<TYPE>(buffer, 2, size, bufferType);
    setSize(buffer[0]->getSize());
    setAllocatedSize(buffer[0]->getAllocatedSize());
    activeBuffer = 0;
}

template <typename TYPE>
CuShiftBuffer<TYPE>::~CuShiftBuffer(){
    deallocate();
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::shift(uint shiftSize){
    static const int blockSize = 256;
    int gridSize = (getSize() - shiftSize + blockSize - 1) / blockSize;
    kernel_shiftData<TYPE><<<gridSize, blockSize>>>(buffer[activeBuffer]->getBuffer(), buffer[!activeBuffer]->getBuffer(), getSize(), shiftSize);
    activeBuffer = !activeBuffer;
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::pushBack(originType origin, const TYPE*& arr, uint size){
    if (size >= getSize()){
        cudaMemcpy(buffer[activeBuffer]->getBuffer(), arr, getSize() * sizeof(TYPE), (cudaMemcpyKind)origin);
    } else {
        shift(size);
        cudaMemcpy(buffer[activeBuffer]->getBuffer() + buffer[activeBuffer]->getSize() - size, arr, size * sizeof(TYPE), (cudaMemcpyKind)origin); 
    }
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::setBuffer(ACuBuffer<TYPE>*& cuBuffer){
    static_cast<CuShiftBuffer<TYPE>*>(buffer[activeBuffer])->deallocate();
    static_cast<CuShiftBuffer<TYPE>*>(buffer[!activeBuffer])->allocate(cuBuffer->getAllocatedSize());

    setSize(cuBuffer->getSize());
    setAllocatedSize(cuBuffer->getAllocatedSize());
    static_cast<CuShiftBuffer<TYPE>*>(buffer[activeBuffer])->setBuffer(cuBuffer);
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::setBuffer(TYPE*& d_buffer, uint size, uint allocatedSize){
    allocatedSize = allocatedSize != 0 ? allocatedSize : size;
    static_cast<CuShiftBuffer<TYPE>*>(buffer[activeBuffer])->deallocate();
    static_cast<CuShiftBuffer<TYPE>*>(buffer[!activeBuffer])->allocate(allocatedSize);

    setSize(size);
    setAllocatedSize(allocatedSize);
    static_cast<CuShiftBuffer<TYPE>*>(buffer[activeBuffer])->setBuffer(d_buffer, size, allocatedSize);
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::resize(uint newSize){
    buffer[0]->resize(newSize);
    buffer[1]->resize(newSize);
    setSize(buffer[0]->getSize());
    setAllocatedSize(buffer[0]->getAllocatedSize());
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::deallocate(){
    static_cast<CuShiftBuffer<TYPE>*>(buffer[0])->deallocate();
    static_cast<CuShiftBuffer<TYPE>*>(buffer[1])->deallocate();
    setSize(0);
    setAllocatedSize(0);
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::allocate(uint size){
    static_cast<CuShiftBuffer<TYPE>*>(buffer[0])->allocate(size);
    static_cast<CuShiftBuffer<TYPE>*>(buffer[1])->allocate(size);
    setSize(size);
    setAllocatedSize(size);
}

template <typename TYPE>
uint CuShiftBuffer<TYPE>::getSize() const {
    return size;
}

template <typename TYPE>
uint CuShiftBuffer<TYPE>::getAllocatedSize() const {
    return allocatedSize;
}

template <typename TYPE>
TYPE* CuShiftBuffer<TYPE>::getBuffer() const {
    return buffer[activeBuffer]->getBuffer();
}

template <typename TYPE>
TYPE* CuShiftBuffer<TYPE>::getInactiveBuffer(uint index) const {
    return buffer[!activeBuffer]->getBuffer() + index;
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::copyInactiveBuffer(destinationType destination, TYPE* buffer, uint size, uint index) const {
    this->buffer[!activeBuffer]->copyBuffer(destination, buffer, size, index);
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::setSize(uint size){
    this->size = size;
    static_cast<CuShiftBuffer<TYPE>*>(buffer[0])->setSize(size);
    static_cast<CuShiftBuffer<TYPE>*>(buffer[1])->setSize(size);
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::setAllocatedSize(uint allocatedSize){
    this->allocatedSize = allocatedSize;
    static_cast<CuShiftBuffer<TYPE>*>(buffer[0])->setAllocatedSize(allocatedSize);
    static_cast<CuShiftBuffer<TYPE>*>(buffer[1])->setAllocatedSize(allocatedSize);
}

template <typename TYPE>
void CuShiftBuffer<TYPE>::setBufferToNull(){
    static_cast<CuShiftBuffer<TYPE>*>(buffer[0])->setBufferToNull();
    static_cast<CuShiftBuffer<TYPE>*>(buffer[1])->setBufferToNull();
}

template <typename TYPE>
TYPE* CuShiftBuffer<TYPE>::getBufferOvnership(){
    TYPE* d_buffer = buffer[activeBuffer]->getBuffer();
    static_cast<CuShiftBuffer<TYPE>*>(buffer[activeBuffer])->setBufferToNull();
    deallocate();
    return d_buffer;
}