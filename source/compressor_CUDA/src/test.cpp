#include "Compressor.h"
#include <stdio.h>

int main(){
    Compressor compressor;
    double value = 0.5;
    printf("Setting parameters to %f\n", value);
    compressor.setGlobalCompressionFactor(value);
    compressor.setVolume(value);
    compressor.setAllCompressionFactors(value);
    compressor.setAllNeutralPoints(value);
    printf("Setting window size to 2048\n");
    compressor.setWindowSize(2048);
    float samplesIn[1024*5];
    float samplesOut[1024*5];
    printf("Compressing samples 1\n");
    compressor.compress(samplesIn, samplesOut, 1024, 0);
    printf("Compressing samples 2\n");
    compressor.compress(samplesIn, samplesOut, 1024*2, 1);
    printf("Compressing samples 3\n");
    compressor.compress(samplesIn, samplesOut, 1024*5, 0);
    printf("Compressor created successfully\n");
    return 0;
}