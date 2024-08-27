#include "Compressor2.h"
#include <stdio.h>

int main(){
    Compressor2 compressor;
    double compressionInitialValue = 0.5;
    printf("Setting parameters to %f\n", compressionInitialValue);
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