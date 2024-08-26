#include "ProcessingUnit_fftR2C.h"

ProcessingUnit_fftR2C::ProcessingUnit_fftR2C(cufftReal*& d_input, cufftComplex*& d_output, cufftHandle& plan)
    : d_input(d_input), d_output(d_output), plan(plan){}

void ProcessingUnit_fftR2C::process(){
    cufftExecR2C(plan, d_input, d_output);
}
