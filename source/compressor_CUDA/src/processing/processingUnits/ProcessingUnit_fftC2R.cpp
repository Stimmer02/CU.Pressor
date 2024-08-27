#include "ProcessingUnit_fftC2R.h"

ProcessingUnit_fftC2R::ProcessingUnit_fftC2R(cufftComplex*& d_input, cufftReal*& d_output, cufftHandle& plan)
    : d_input(d_input), d_output(d_output), plan(plan){}

void ProcessingUnit_fftC2R::process(){
    cufftExecC2R(plan, d_input, d_output);
}