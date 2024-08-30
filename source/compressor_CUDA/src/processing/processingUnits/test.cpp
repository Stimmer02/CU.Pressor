#include <cuda.h>

template<typename TYPE, unsigned short SIZE>
struct A {
    __constant__ TYPE[SIZE] a;
};