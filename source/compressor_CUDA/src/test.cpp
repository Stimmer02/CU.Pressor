#include "Compressor.h"
#include <stdio.h>

int main(){
    Compressor compressor;
    double compressionInitialValue = 0.5;
    compressor.setCompressionFactor1(compressionInitialValue);
    printf("Compressor created successfully\n");
    return 0;
}