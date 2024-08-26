#include "ProcessingUnit_volume.h"

__global__ void volumeControl(float* data, int size, float volume){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        data[idx] *= volume;
    }
}

ProcessingUnit_volume::ProcessingUnit_volume(float*& d_workBuffer, const uint& gridSize, const uint& blockSize, const uint& bufferSize) 
    : d_workBuffer(d_workBuffer), gridSize(gridSize), blockSize(blockSize), bufferSize(bufferSize){
    volume = 1.0f;
}

void ProcessingUnit_volume::process(){
    volumeControl<<<gridSize, blockSize>>>(d_workBuffer, bufferSize, volume);
}

void ProcessingUnit_volume::setVolume(float volume){
    this->volume = volume;
    setActive(volume != 1.0f);
}

float ProcessingUnit_volume::getVolume() const{
    return volume;
}