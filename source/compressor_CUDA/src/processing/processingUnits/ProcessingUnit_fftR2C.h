#pragma once

#include "AProcessingUnit.h"

#include <cufft.h>

typedef unsigned int uint;

/// @brief Processing unit that transforms real data to complex data using FFT
class ProcessingUnit_fftR2C : public AProcessingUnit{
public:
    /// @brief Constructor
    /// @param d_input reference to the device buffer to be processed (in)
    /// @param d_output reference to the device buffer to store the result (out)
    /// @param plan reference to the cufft plan to be used
    ProcessingUnit_fftR2C(cufftReal*& d_input, cufftComplex*& d_output, cufftHandle& plan);

    /// @brief Transforms real data to complex data using FFT
    void process() override;

private:
    cufftReal*& d_input;
    cufftComplex*& d_output;
    cufftHandle& plan;
};