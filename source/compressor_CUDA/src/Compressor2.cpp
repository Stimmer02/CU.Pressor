#include "Compressor2.h"

Compressor2::Compressor2(const uint& bandCount, const uint& windowSize, const uint& sampleRate){
    settings.bandCount = bandCount;
    // settings.bandCount = 2;
    settings.windowSize = 0;
    settings.sampleRate = 0;
    settings.processingSize = 0;
    settings.complexWindowSize = 0;
    settings.addressShift = 0;

    fft.C2R = 0;
    fft.R2C = 0;

    buffers.workBuffer = new CuShiftBuffer<float>[2];
    for (int i = 0; i < 2; i++){
        buffers.workBuffer[i] = CuShiftBuffer<float>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    }
    buffers.cufftOutput = CuBufferFactory::createBuffer<cufftComplex>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    buffers.cufftBands = CuBufferFactory::createBuffer<cufftComplex>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    buffers.bands = CuBufferFactory::createBuffer<float>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);

    units.fftR2C = new ProcessingUnit_fftR2C(bufferPointers.d_workBuffer, bufferPointers.d_cufftOutput, fft.R2C);
    units.fftBandSplit = new ProcessingUnit_fftBandSplit(bufferPointers.d_cufftOutput, bufferPointers.d_cufftBands, kernelSize.gridComplex2D, kernelSize.block, settings.complexWindowSize, settings.sampleRate);
    units.fftC2R = new ProcessingUnit_fftC2R(bufferPointers.d_cufftBands, bufferPointers.d_bands, fft.C2R);
    units.fftBandMerge = new ProcessingUnit_fftBandMerge(bufferPointers.d_bands, bufferPointers.d_output, kernelSize.gridReal, kernelSize.block, settings.processingSize, settings.bandCount, settings.addressShift);
    units.cuPressor = new ProcessingUnit_cuPressor(bufferPointers.d_output, kernelSize.gridReal, kernelSize.block, settings.processingSize);
    units.volume = new ProcessingUnit_volume(bufferPointers.d_output, kernelSize.gridReal, kernelSize.block, settings.processingSize);

    processingQueue.appendQueue(units.fftR2C);
    processingQueue.appendQueue(units.fftBandSplit);
    processingQueue.appendQueue(units.fftC2R);
    processingQueue.appendQueue(units.fftBandMerge);
    processingQueue.appendQueue(units.cuPressor);
    processingQueue.appendQueue(units.volume);

    // setSampleRate(sampleRate); // commented to avoid repeating execution of generateBandSplittingTable()
    settings.sampleRate = sampleRate;
    setWindowSize(windowSize);
}

Compressor2::~Compressor2(){
    delete[] buffers.workBuffer;
    delete buffers.cufftOutput;
    delete buffers.cufftBands;
    delete buffers.bands;

    delete units.fftR2C;
    delete units.fftBandSplit;
    delete units.fftC2R;
    delete units.fftBandMerge;
    delete units.cuPressor;
    delete units.volume;

    if (fft.R2C != 0) {
        cufftDestroy(fft.R2C);
    }
    if (fft.C2R != 0) {
        cufftDestroy(fft.C2R);
    }
}

void Compressor2::compress(const float* samplesIn, float* samplesOut, const uint& size, const uint& channelNumber){
    if (size <= settings.windowSize){
        processSingleWindow(samplesIn, samplesOut, size, channelNumber);
    } else {
        processMultipleWindows(samplesIn, samplesOut, size, channelNumber);
    }
    cudaDeviceSynchronize();
}

void Compressor2::processSingleWindow(const float* samplesIn, float* samplesOut, const uint& size, const uint& channelNumber){
    setProcessingSize(size);
    buffers.workBuffer[channelNumber].pushBack(FROM_HOST, samplesIn, settings.processingSize);
    // those two pointers change every time pushBack is called
    bufferPointers.d_workBuffer = buffers.workBuffer[channelNumber];
    bufferPointers.d_output = buffers.workBuffer[channelNumber].getInactiveBuffer();

    processingQueue.execute();
    buffers.workBuffer[channelNumber].copyInactiveBuffer(TO_HOST, samplesOut, settings.processingSize);
}

void Compressor2::processMultipleWindows(const float* samplesIn, float* samplesOut, const uint& size, const uint& channelNumber){
    int leftToProcess = size;
    for (int i = 0; i < size - settings.windowSize; i += settings.windowSize){
        processSingleWindow(samplesIn + i, samplesOut + i, (leftToProcess > settings.windowSize) ? settings.windowSize : leftToProcess, channelNumber);
        leftToProcess -= settings.windowSize;
    }
}

void Compressor2::setWindowSize(uint size){
    settings.windowSize = size;
    settings.complexWindowSize = size / 2 + 1;
    settings.addressShift = settings.windowSize - settings.processingSize;
    calculateKernelSize();
    resize(size);
    units.fftBandSplit->generateBandSplittingTable();

    if (fft.R2C != 0) {
        cufftDestroy(fft.R2C);
    }
    cufftPlan1d(&fft.R2C, size, CUFFT_R2C, 1);
    if (fft.C2R != 0) {
        cufftDestroy(fft.C2R);
    }
    cufftPlan1d(&fft.C2R, size, CUFFT_C2R, settings.bandCount);
}

// void Compressor2::setBandCount(uint count){
//     settings.bandCount = count;
//     calculateKernelSize();
//     resize(settings.processingSize, settings.complexWindowSize, settings.bandCount);
//     units.fftBandSplit->generateBandSplittingTable();
//     if (fft.C2R != 0) {
//         cufftDestroy(fft.C2R);
//     }
//     cufftPlan1d(&fft.C2R, size, CUFFT_C2R, settings.bandCount);
// }

void Compressor2::setSampleRate(uint rate){
    settings.sampleRate = rate;
    units.fftBandSplit->generateBandSplittingTable();
}

void Compressor2::resize(uint size, uint complexSize){
    complexSize = (complexSize == 0) ? size / 2 + 1 : complexSize;

    buffers.workBuffer[0].resize(size);
    buffers.workBuffer[1].resize(size);
    bufferPointers.d_workBuffer = buffers.workBuffer[0];
    bufferPointers.d_output = buffers.workBuffer[0].getInactiveBuffer();

    buffers.cufftOutput->resize(complexSize);
    bufferPointers.d_cufftOutput = *buffers.cufftOutput;

    buffers.cufftBands->resize(complexSize * settings.bandCount);
    bufferPointers.d_cufftBands = *buffers.cufftBands;

    buffers.bands->resize(size * settings.bandCount);
    bufferPointers.d_bands = *buffers.bands;
}

void Compressor2::calculateKernelSize(){
    kernelSize.block = 256;
    kernelSize.gridReal = (settings.processingSize + kernelSize.block - 1) / kernelSize.block;
    kernelSize.gridReal2D = dim3(kernelSize.gridReal, settings.bandCount);
    kernelSize.gridComplex = (settings.complexWindowSize + kernelSize.block - 1) / kernelSize.block;
    kernelSize.gridComplex2D = dim3(kernelSize.gridComplex, settings.bandCount);
}

void Compressor2::setProcessingSize(uint size){
    if (size == settings.processingSize){
        return;
    }
    settings.processingSize = size;
    settings.addressShift = settings.windowSize - settings.processingSize;
    calculateKernelSize();
}

void Compressor2::setVolume(float volume){
    units.volume->setVolume(volume);
}

void Compressor2::setCompressionFactor(float factor){
    units.cuPressor->setCompressionFactor(factor);
}