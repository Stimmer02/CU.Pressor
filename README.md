# CU.Pressor
CUDA compressor-like VST3 plugin.

A try to harness power of the GPU in audio processing. This time in a more shareable environment.
The whole idea is based on my previous project - CU.Syntch where I have been researching possibilities of GPU in audio processing. From konledge I have gathered I can say that there is much potential in this idea. GPU opens a world of possibilities that would not be even considered because of how expensive (in computation time) some algorithms would be on CPU. 

This plugin uses simple compression formula on many frequency bands making whole spectrum more even in volume and presence. At least that is the idea - for now everything is work in progress...

To run this you need:
1. a DAW (I use Reaper)
2. CUDA toolkit with properly set environment paths
