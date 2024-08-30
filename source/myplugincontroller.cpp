//------------------------------------------------------------------------
// Copyright(c) 2024 stimmer02.
//------------------------------------------------------------------------

#include "myplugincontroller.h"
#include "myplugincids.h"
// #include "vstgui/plugin-bindings/vst3editor.h"

using namespace Steinberg;


#include <string>
#include <stdio.h>
#include <vector>
#include <codecvt>
#include <locale>
#include "compressor_CUDA/src/Compressor.h"

// Function to convert std::string to Steinberg::Vst::TChar*
Steinberg::Vst::TChar* toTChar(const std::string& str) {
    std::u16string u16str = std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>{}.from_bytes(str);
    Steinberg::Vst::TChar* tcharStr = new Steinberg::Vst::TChar[u16str.size() + 1];
    std::copy(u16str.begin(), u16str.end(), tcharStr);
    tcharStr[u16str.size()] = 0; // Null-terminate the string
    return tcharStr;
}
namespace cudaCompressor {

//------------------------------------------------------------------------
// CuPressorController Implementation
//------------------------------------------------------------------------

tresult PLUGIN_API CuPressorController::initialize (FUnknown* context)
{
	// Here the Plug-in will be instantiated

	//---do not forget to call parent ------
	tresult result = EditControllerEx1::initialize (context);
	if (result != kResultOk)
	{
		return result;
	}

	parameters.addParameter(
		STR16("Global Compression"),    // Parameter title
		nullptr,          // Parameter units (optional)
		0,                // Step count (0 means continuous)
		0.1,              // Default value (in normalized range [0,1])
		Vst::ParameterInfo::kCanAutomate, // Flags (this one makes it automatable)
		0,                // Parameter ID
		0,                // Parameter group (optional)
		STR16("GComp"));   // Short title (optional)
	parameters.addParameter(
		STR16("Volume"),   
		nullptr,          
		0,               
		1.0,              
		Vst::ParameterInfo::kCanAutomate, 
		1,                
		0,              
		STR16("Vol"));
	// parameters.addParameter(
	// 	STR16("All Compression"),   
	// 	nullptr,          
	// 	0,               
	// 	0,              
	// 	Vst::ParameterInfo::kCanAutomate, 
	// 	2,                
	// 	0,              
	// 	STR16("AComp"));
	// parameters.addParameter(
	// 	STR16("All Neutral Point"),   
	// 	nullptr,          
	// 	0,               
	// 	1.0,              
	// 	Vst::ParameterInfo::kCanAutomate, 
	// 	3,                
	// 	0,              
	// 	STR16("ANP"));

	const int startFrom = 2;
	for (int i = 0; i < COMPRESSOR_BANDS; i++){
		char buffer[100];
		snprintf(buffer, sizeof(buffer), " (%.0f - %.0fHz)", 
				Compressor::getlowerFrequencyBandBound(i, COMPRESSOR_BANDS), 
				Compressor::getlowerFrequencyBandBound(i+1, COMPRESSOR_BANDS));
    
   		std::string bandCompression = "Band " + std::to_string(i+1) + " Compression" + buffer;
		parameters.addParameter(
            toTChar(bandCompression),
            nullptr,
            0,
            0.0,
            Vst::ParameterInfo::kCanAutomate,
            i*2+startFrom,
            0,
            toTChar("B" + std::to_string(i+1) + "Comp"));

        parameters.addParameter(
            toTChar("Band " + std::to_string(i+1) + " Neutral Point"),   
            nullptr,          
            0,               
            1.0,              
            Vst::ParameterInfo::kCanAutomate, 
            i*2+1+startFrom,                
            0,              
            toTChar("B" + std::to_string(i+1) + "NP"));
	}


	return result;
}

//------------------------------------------------------------------------
tresult PLUGIN_API CuPressorController::terminate ()
{
	// Here the Plug-in will be de-instantiated, last possibility to remove some memory!

	//---do not forget to call parent ------
	return EditControllerEx1::terminate ();
}

//------------------------------------------------------------------------
tresult PLUGIN_API CuPressorController::setComponentState (IBStream* state)
{
	// Here you get the state of the component (Processor part)
	if (!state)
		return kResultFalse;

	return kResultOk;
}

//------------------------------------------------------------------------
tresult PLUGIN_API CuPressorController::setState (IBStream* state)
{
	// Here you get the state of the controller

	return kResultTrue;
}

//------------------------------------------------------------------------
tresult PLUGIN_API CuPressorController::getState (IBStream* state)
{
	// Here you are asked to deliver the state of the controller (if needed)
	// Note: the real state of your plug-in is saved in the processor

	return kResultTrue;
}

//------------------------------------------------------------------------
IPlugView* PLUGIN_API CuPressorController::createView (FIDString name)
{
	// Here the Host wants to open your editor (if you have one)
	// if (FIDStringsEqual (name, Vst::ViewType::kEditor))
	// {
	// 	// create your editor here and return a IPlugView ptr of it
	// 	auto* view = new VSTGUI::VST3Editor (this, "view", "myplugineditor.uidesc");
	// 	return view;
	// }
	return nullptr;
}

//------------------------------------------------------------------------
} // namespace cudaCompressor
