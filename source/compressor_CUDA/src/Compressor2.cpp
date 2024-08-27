#include "Compressor2.h"

Compressor2::Compressor2(const uint& bandCount, const uint& windowSize, const uint& sampleRate){
    settings.bandCount = bandCount;
    settings.windowSize = 0;
    settings.sampleRate = 0;
    settings.processingSize = 0;
    settings.complexWindowSize = 0;

    buffers.workBuffer = new CuShiftBuffer<float>[2];
    for (int i = 0; i < 2; i++){
        buffers.workBuffer[i] = CuShiftBuffer<float>(0, CuBufferFactory::bufferType::TIME_OPTIMAL);
    }

    units.cuPressor = new ProcessingUnit_cuPressor(bufferPointers.d_workBuffer, kernelSize.gridReal, kernelSize.blockReal, settings.processingSize);
    units.volume = new ProcessingUnit_volume(bufferPointers.d_workBuffer, kernelSize.gridReal, kernelSize.blockReal, settings.processingSize);

    processingQueue.appendQueue(units.cuPressor);
    processingQueue.appendQueue(units.volume);

    setWindowSize(windowSize);
    setSampleRate(sampleRate);
}

Compressor2::~Compressor2(){
    delete[] buffers.workBuffer;
    delete units.cuPressor;
    delete units.volume;
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
    bufferPointers.d_workBuffer = buffers.workBuffer[channelNumber][settings.addressShift];

    processingQueue.execute();

    buffers.workBuffer[channelNumber].copyBuffer(TO_HOST, samplesOut, settings.processingSize, settings.addressShift);
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
    resize(size);
}

// void Compressor2::setBandCount(uint count){
//     settings.bandCount = count;
// }

void Compressor2::setSampleRate(uint rate){
    settings.sampleRate = rate;
}

void Compressor2::resize(uint size){
    buffers.workBuffer[0].resize(size);
    buffers.workBuffer[1].resize(size);
    bufferPointers.d_workBuffer = buffers.workBuffer[0];
}

void Compressor2::calculateKernelSize(){
    kernelSize.blockReal = 256;
    kernelSize.gridReal = (settings.processingSize + kernelSize.blockReal - 1) / kernelSize.blockReal;
}

void Compressor2::setProcessingSize(uint size){
    if (size == settings.processingSize){
        return;
    }
    settings.processingSize = size;
    calculateKernelSize();
}

void Compressor2::setVolume(float volume){
    units.volume->setVolume(volume);
}

void Compressor2::setCompressionFactor(float factor){
    units.cuPressor->setCompressionFactor(factor);
}