#include "ACuBuffer.h"

template <typename TYPE>
class CuBuffer_memoryOptimal : public ACuBuffer<TYPE>{
public:
    CuBuffer_memoryOptimal();
    CuBuffer_memoryOptimal(uint size);
    CuBuffer_memoryOptimal(CuBuffer_memoryOptimal*& cuBuffer);

    void resize(uint newSize) override;
};