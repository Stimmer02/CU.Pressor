#include "CuShiftBuffer.cu"
#include "CuSimpleBuffer_timeOptimal.cu"
#include "CuSimpleBuffer_memoryOptimal.cu"
#include "ACuSimpleBuffer.cu"
#include "ACuBuffer.cu"


#define INSTANTIATE_TEMPLATE_CLASS(TEMPLATE_CLASS, TYPE) \
    template class TEMPLATE_CLASS<TYPE>;

#define INSTANTIATE_TEMPLATE_CLASSES(TYPE) \
    INSTANTIATE_TEMPLATE_CLASS(CuShiftBuffer, TYPE) \
    INSTANTIATE_TEMPLATE_CLASS(CuSimpleBuffer_timeOptimal, TYPE) \
    INSTANTIATE_TEMPLATE_CLASS(CuSimpleBuffer_memoryOptimal, TYPE) \
    INSTANTIATE_TEMPLATE_CLASS(ACuSimpleBuffer, TYPE) \
    INSTANTIATE_TEMPLATE_CLASS(ACuBuffer, TYPE)

INSTANTIATE_TEMPLATE_CLASSES(float)
INSTANTIATE_TEMPLATE_CLASSES(float2)