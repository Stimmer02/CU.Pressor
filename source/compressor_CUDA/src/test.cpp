#include "Compressor.h"
#include <stdio.h>

int main(){
    Compressor compressor;
    double compressionInitialValue = 0.5;
    printf("Setting parameters to %f\n", compressionInitialValue);
    compressor.setCompressionFactor1(compressionInitialValue);
    compressor.setCompressionFactor2(compressionInitialValue);
    compressor.setVolume(compressionInitialValue);
    compressor.setPreGain(compressionInitialValue);
    printf("Setting window size to 2048\n");
    compressor.setWindowSize(2048);
    float samplesIn[1024*5];
    float samplesOut[1024*5];
    printf("Compressing samples 1\n");
    compressor.compress(samplesIn, samplesOut, 1024);
    printf("Compressing samples 2\n");
    compressor.compress(samplesIn, samplesOut, 1024*2);
    printf("Compressing samples 3\n");
    compressor.compress(samplesIn, samplesOut, 1024*5);
    printf("Compressor created successfully\n");
    return 0;
}