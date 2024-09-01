# CU.Pressor
CUDA dynamic compressor VST3 plugin.

## About this project
This is a try to harness power of the GPU in audio processing. This time in a more shareable environment.
The whole idea is based on my previous project - CU.Syntch where I have been researching possibilities of GPU in audio processing. From konledge I have gathered I can say that there is much potential in this idea. GPU opens a world of possibilities that would not be even considered because of how expensive (in computation time) some algorithms would be on CPU. 

## What this plugin does
This plugin splits signal into bands and then compresses them into specified value using sigmoid-like function. While properly setup this plugin won't increase the maximum amplitude of any band but still increase it's volume by rising increasing volume of quiet parts of the singal while decreasing it in case of luder parts (after the bands are summed at the output overall signall will have increased maximal amplitude). 

## How to run 
To run this you need:
1. CUDA capable device
2. [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) with properly set environment paths 
3. a DAW (I use Reaper)

Plugin comes in two parts (it is a long story):
- __.vst__ folder 
- __.dll__ file

__.vst__ goes where you keep your plugins and .dll has to go where you DAW will see it. I have struggled a lot to make REAPER see this __.dll__ file and the only place that it consistently works is in C:\Windows\System32 folder. If you have better idea where to put this file then please share it with me. Anyway, after placing the __.dll__ file you will have to reboot your system so it will be able to recognize it.

## To do:
- [x] processing route optimization - flexible processing route modified by demand
- [ ] easily modifiable band count - right now band count is set during compilation
- [ ] separate bass processing route - for higher bass quality
- [ ] proper user interface - for now it just utilizes parameters
- [ ] automatic volume adjustments - allowing to preserve average aplitude of the oryginall signall
- [ ] quality/performance slider - way to choose the output quality
- [ ] performance/latency slider - way to choose latency induced by processing
- [ ] multiple channel processing - for now it is fixed on stereo
- [ ] parallel channel processing - possibly for massive performance gain
- [ ] GPU constant memory utilization - could turn beneficial in some cases
