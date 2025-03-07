#include "Compressor.h"

Compressor::Compressor(const uint& bandCount, const uint& windowSize, const uint& sampleRate){
    settings.bandCount = bandCount;
    settings.sampleRate = sampleRate;
    settings.windowSize = 0;
    settings.processingSize = 0;
    settings.complexWindowSize = 0;
    settings.addressShift = 0;

    fft.C2R = 0;
    fft.R2C = 0;

    buffers.shiftBuffer = new CuShiftBuffer<float>[2];
    for (int i = 0; i < 2; i++){
        buffers.shiftBuffer[i] = CuShiftBuffer<float>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    }
    buffers.cufftOutput = CuBufferFactory::createBuffer<cufftComplex>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    buffers.cufftBands = CuBufferFactory::createBuffer<cufftComplex>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    buffers.bands = CuBufferFactory::createBuffer<float>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);

    units.windowing = new ProcessingUnit_windowing(bufferPointers.d_shiftBuffer, bufferPointers.d_workBuffer, kernelSize.gridFullWindow, kernelSize.block, settings.windowSize);
    units.fftR2C = new ProcessingUnit_fftR2C(bufferPointers.d_workBuffer, bufferPointers.d_cufftOutput, fft.R2C);
    units.fftBandSplit = new ProcessingUnit_fftBandSplit(bufferPointers.d_cufftOutput, bufferPointers.d_cufftBands, kernelSize.gridComplex2D, kernelSize.block, settings.complexWindowSize, settings.sampleRate);
    units.fftC2R = new ProcessingUnit_fftC2R(bufferPointers.d_cufftBands, bufferPointers.d_bands, fft.C2R);
    units.cuPressorBatch = new ProcessingUnit_cuPressorBatch(bufferPointers.d_bandsMoved, kernelSize.gridReal, kernelSize.block, settings.processingSize, settings.bandCount, settings.addressShift);
    units.fftBandMerge = new ProcessingUnit_fftBandMerge(bufferPointers.d_bandsMoved, bufferPointers.d_workBuffer, kernelSize.gridReal, kernelSize.block, settings.processingSize, settings.bandCount, settings.addressShift);

    units.fftBypass = new ProcessingUnit_copyBuffer<float>((const float*&)bufferPointers.d_shiftBufferCurrentPart, bufferPointers.d_workBuffer, settings.processingSize, cudaMemcpyDeviceToDevice);

    units.cuPressor = new ProcessingUnit_cuPressor(bufferPointers.d_workBuffer, kernelSize.gridReal, kernelSize.block, settings.processingSize);
    units.volume = new ProcessingUnit_volume(bufferPointers.d_workBuffer, kernelSize.gridReal, kernelSize.block, settings.processingSize);
    units.copyToHost = new ProcessingUnit_copyBuffer<float>((const float*&)bufferPointers.d_workBuffer, bufferPointers.output, settings.processingSize, cudaMemcpyDeviceToHost);

    units.systemBypass = new ProcessingUnit_copyBuffer<float>(bufferPointers.input, bufferPointers.output, settings.processingSize, cudaMemcpyHostToHost);
    units.GPUProcessing = new SoftDependencyGroup(3);

    processingQueue.appendQueue(units.windowing);
    processingQueue.appendQueue(units.fftR2C);
    processingQueue.appendQueue(units.fftBandSplit);
    processingQueue.appendQueue(units.fftC2R);
    processingQueue.appendQueue(units.cuPressorBatch);
    processingQueue.appendQueue(units.fftBandMerge);
    processingQueue.appendQueue(units.fftBypass);
    processingQueue.appendQueue(units.cuPressor);
    processingQueue.appendQueue(units.volume);
    processingQueue.appendQueue(units.copyToHost);
    processingQueue.appendQueue(units.systemBypass);

    units.windowing->registerDependency(units.cuPressorBatch);
    units.fftR2C->registerDependency(units.cuPressorBatch);
    units.fftBandSplit->registerDependency(units.cuPressorBatch);
    units.fftC2R->registerDependency(units.cuPressorBatch);
    units.fftBandMerge->registerDependency(units.cuPressorBatch);

    units.fftBypass->registerExclusion(units.cuPressorBatch);

    units.GPUProcessing->registerUnit(units.cuPressorBatch);
    units.GPUProcessing->registerUnit(units.cuPressor);
    units.GPUProcessing->registerUnit(units.volume);
    units.copyToHost->registerDependency(units.GPUProcessing);

    units.systemBypass->registerExclusion(units.GPUProcessing);

    setWindowSize(windowSize);
}

Compressor::~Compressor(){
    delete[] buffers.shiftBuffer;
    delete buffers.cufftOutput;
    delete buffers.cufftBands;
    delete buffers.bands;

    delete units.windowing;
    delete units.fftR2C;
    delete units.fftBandSplit;
    delete units.fftC2R;
    delete units.cuPressorBatch;
    delete units.fftBandMerge;
    delete units.fftBypass;
    delete units.cuPressor;
    delete units.volume;
    delete units.copyToHost;
    delete units.systemBypass;
    delete units.GPUProcessing;

    if (fft.R2C != 0) {
        cufftDestroy(fft.R2C);
    }
    if (fft.C2R != 0) {
        cufftDestroy(fft.C2R);
    }
}

void Compressor::compress(const float* samplesIn, float* samplesOut, const uint& size, const uint& channelNumber){
    if (size <= settings.windowSize){
        processSingleWindow(samplesIn, samplesOut, size, channelNumber);
    } else {
        processMultipleWindows(samplesIn, samplesOut, size, channelNumber);
    }
    cudaDeviceSynchronize();
}

void Compressor::processSingleWindow(const float* samplesIn, float* samplesOut, const uint& size, const uint& channelNumber){
    setProcessingSize(size);
    bufferPointers.input = samplesIn;
    bufferPointers.output = samplesOut;
    buffers.shiftBuffer[channelNumber].pushBack(FROM_HOST, samplesIn, settings.processingSize);
    bufferPointers.d_shiftBuffer = buffers.shiftBuffer[channelNumber];
    bufferPointers.d_shiftBufferCurrentPart = buffers.shiftBuffer[channelNumber][settings.addressShift];
    bufferPointers.d_workBuffer = buffers.shiftBuffer[channelNumber].getInactiveBuffer();

    processingQueue.execute();
}

void Compressor::processMultipleWindows(const float* samplesIn, float* samplesOut, const uint& size, const uint& channelNumber){
    int leftToProcess = size;
    for (int i = 0; i < size - settings.windowSize; i += settings.windowSize){
        processSingleWindow(samplesIn + i, samplesOut + i, (leftToProcess > settings.windowSize) ? settings.windowSize : leftToProcess, channelNumber);
        leftToProcess -= settings.windowSize;
    }
}

void Compressor::setWindowSize(uint size){
    settings.windowSize = size;
    settings.complexWindowSize = size / 2 + 1;
    settings.addressShift = settings.windowSize - settings.processingSize;
    calculateKernelSize();
    resize(size);
    units.fftBandSplit->generateBandSplittingTable();
    units.windowing->generateWindow(settings.windowSize / 5, settings.processingSize / 5);

    if (fft.R2C != 0) {
        cufftDestroy(fft.R2C);
    }
    cufftPlan1d(&fft.R2C, size, CUFFT_R2C, 1);
    if (fft.C2R != 0) {
        cufftDestroy(fft.C2R);
    }
    cufftPlan1d(&fft.C2R, size, CUFFT_C2R, settings.bandCount);
}

// void Compressor::setBandCount(uint count){
//     settings.bandCount = count;
//     calculateKernelSize();
//     resize(settings.processingSize, settings.complexWindowSize, settings.bandCount);
//     units.fftBandSplit->generateBandSplittingTable();
//     if (fft.C2R != 0) {
//         cufftDestroy(fft.C2R);
//     }
//     cufftPlan1d(&fft.C2R, size, CUFFT_C2R, settings.bandCount);
// }

void Compressor::setSampleRate(uint rate){
    settings.sampleRate = rate;
    units.fftBandSplit->generateBandSplittingTable();
}

void Compressor::resize(uint size, uint complexSize){
    complexSize = (complexSize == 0) ? size / 2 + 1 : complexSize;

    buffers.shiftBuffer[0].resize(size);
    buffers.shiftBuffer[1].resize(size);
    bufferPointers.d_shiftBuffer = buffers.shiftBuffer[0];
    bufferPointers.d_shiftBufferCurrentPart = buffers.shiftBuffer[0][settings.addressShift];
    bufferPointers.d_workBuffer = buffers.shiftBuffer[0].getInactiveBuffer();

    buffers.cufftOutput->resize(complexSize);
    bufferPointers.d_cufftOutput = *buffers.cufftOutput;

    buffers.cufftBands->resize(complexSize * settings.bandCount);
    bufferPointers.d_cufftBands = *buffers.cufftBands;

    buffers.bands->resize(size * settings.bandCount);
    bufferPointers.d_bands = *buffers.bands;
    bufferPointers.d_bandsMoved = bufferPointers.d_bands - settings.processingSize;
}

void Compressor::calculateKernelSize(){
    kernelSize.block = 256;
    kernelSize.gridReal = (settings.processingSize + kernelSize.block - 1) / kernelSize.block;
    kernelSize.gridReal2D = dim3(kernelSize.gridReal, settings.bandCount);
    kernelSize.gridFullWindow = (settings.windowSize + kernelSize.block - 1) / kernelSize.block;
    kernelSize.gridComplex = (settings.complexWindowSize + kernelSize.block - 1) / kernelSize.block;
    kernelSize.gridComplex2D = dim3(kernelSize.gridComplex, settings.bandCount);
}

void Compressor::setProcessingSize(uint size){
    if (size == settings.processingSize){
        return;
    }
    settings.processingSize = size;
    settings.addressShift = settings.windowSize - settings.processingSize;
    bufferPointers.d_bandsMoved = bufferPointers.d_bands - size;
    calculateKernelSize();
    units.windowing->generateWindow(settings.windowSize / 5, settings.processingSize / 5);
}

void Compressor::setVolume(float volume){
    units.volume->setVolume(volume);
}

void Compressor::setGlobalCompressionFactor(float factor){
    units.cuPressor->setCompressionFactor(factor);
}

void Compressor::setCompressionFactor(uint band, float factor){
    units.cuPressorBatch->setCompressionFactor(band, factor);
}

void Compressor::setAllCompressionFactors(float factor){
    units.cuPressorBatch->setAllCompressionFactors(factor);
}

void Compressor::setNeutralPoint(uint band, float neutralPoint){
    units.cuPressorBatch->setNeutralPoint(band, neutralPoint);
}

void Compressor::setAllNeutralPoints(float neutralPoint){
    units.cuPressorBatch->setAllNeutralPoints(neutralPoint);
}

float Compressor::getlowerFrequencyBandBound(const uint &bandIndex, const uint &bandCount){
    return ProcessingUnit_fftBandSplit::getlowerFrequencyBandBound(bandIndex, bandCount);
}
