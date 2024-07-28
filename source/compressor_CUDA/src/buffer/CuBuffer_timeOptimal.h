#include "ACuBuffer.h"

template <typename TYPE>
class CuBuffer_timeOptimal : public ACuBuffer<TYPE>{
public:
    CuBuffer_timeOptimal();
    CuBuffer_timeOptimal(uint size);
    CuBuffer_timeOptimal(CuBuffer_timeOptimal*& cuBuffer);

    void resize(uint newSize) override;
};