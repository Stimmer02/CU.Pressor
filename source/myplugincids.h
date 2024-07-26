//------------------------------------------------------------------------
// Copyright(c) 2024 stimmer02.
//------------------------------------------------------------------------

#pragma once

#include "pluginterfaces/base/funknown.h"
#include "pluginterfaces/vst/vsttypes.h"

namespace cudaCompressor {
//------------------------------------------------------------------------
static const Steinberg::FUID kCuPressorProcessorUID (0x0E00A561, 0x2C6458BE, 0xB7608BAB, 0x42192C36);
static const Steinberg::FUID kCuPressorControllerUID (0x78116FF1, 0xC0BF54C4, 0x9FDB009A, 0xCD4E0759);

#define CuPressorVST3Category "Fx"

//------------------------------------------------------------------------
} // namespace cudaCompressor
