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
struct has_not_equal_operator {
    template <typename U>
    static auto test(U* u) -> decltype(*u != *u, std::true_type());

    template <typename>
    static std::false_type test(...);

    static constexpr bool value = decltype(test<TYPE>(nullptr))::value;
};


template <typename TYPE>
typename std::enable_if<has_not_equal_operator<TYPE>::value, bool>::type
compareKernelWrapper(TYPE* d_buffer1, TYPE* d_buffer2, uint size){
    bool* d_result;
    cudaMalloc(&d_result, sizeof(bool));
    kernel_compare<TYPE><<<(size + 255) / 256, 256>>>(d_buffer1, d_buffer2, size, d_result);

    bool result;
    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}

template <typename TYPE>
typename std::enable_if<!has_not_equal_operator<TYPE>::value, bool>::type
compareKernelWrapper(TYPE* d_buffer1, TYPE* d_buffer2, uint size){
    bool* d_result;
    cudaMalloc(&d_result, sizeof(bool));
    size *= sizeof(TYPE);
    kernel_compare<unsigned char><<<(size + 255) / 256, 256>>>((unsigned char*)d_buffer1, (unsigned char*)d_buffer2, size, d_result);

    bool result;
    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}

template <typename TYPE>
void ACuBuffer<TYPE>::clear(){
    cudaMemset(getBuffer(), 0, getSize() * sizeof(TYPE));
}

template <typename TYPE>
void ACuBuffer<TYPE>::copyBuffer(const ACuBuffer<TYPE>& cuBuffer){
    if (this != &cuBuffer) {
        if (cuBuffer.getBuffer() == nullptr){
            deallocate();
        }
        resize(cuBuffer.getSize());
        cudaMemcpy(getBuffer(), cuBuffer.getBuffer(), getSize() * sizeof(TYPE), cudaMemcpyDeviceToDevice);
    }
}

template <typename TYPE>
void ACuBuffer<TYPE>::copyBuffer(const TYPE*& d_buffer, uint size){
    resize(size);
    cudaMemcpy(getBuffer(), d_buffer, size * sizeof(TYPE), cudaMemcpyHostToDevice);
}

template <typename TYPE>
bool ACuBuffer<TYPE>::compare(const ACuBuffer<TYPE>& cuBuffer) const {
    if (this == &cuBuffer){
        return true;
    }
    if (getSize() != cuBuffer.getSize()){
        return false;
    }

    return compareKernelWrapper<TYPE>(getBuffer(), cuBuffer.getBuffer(), getSize());
}

template <typename TYPE>
ACuBuffer<TYPE>& ACuBuffer<TYPE>::operator=(const ACuBuffer<TYPE>& cuBuffer){
    copyBuffer(cuBuffer);
    return *this;
}

template <typename TYPE>
bool ACuBuffer<TYPE>::operator==(const ACuBuffer<TYPE>& cuBuffer) const {
    return compare(cuBuffer);
}

template <typename TYPE>
bool ACuBuffer<TYPE>::operator!=(const ACuBuffer<TYPE>& cuBuffer) const {
    return !compare(cuBuffer);
}

template <typename TYPE>
TYPE* ACuBuffer<TYPE>::operator[](uint index) const {
    return getBuffer() + index;
}

template <typename TYPE>
ACuBuffer<TYPE>::operator TYPE*(){
    return getBuffer();
}

template <typename TYPE>
ACuBuffer<TYPE>::operator const TYPE*(){
    return getBuffer();
}

template <typename TYPE>
TYPE* ACuBuffer<TYPE>::getBufferOvnership(){
    TYPE* d_buffer = getBuffer();
    setBufferToNull();
    setSize(0);
    setAllocatedSize(0);
    return d_buffer; 
}