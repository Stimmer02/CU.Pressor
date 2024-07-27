//------------------------------------------------------------------------
// Copyright(c) 2024 stimmer02.
//------------------------------------------------------------------------

#include "myplugincontroller.h"
#include "myplugincids.h"
#include "vstgui/plugin-bindings/vst3editor.h"

using namespace Steinberg;

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
		STR16("Compression1"),    // Parameter title
		nullptr,          // Parameter units (optional)
		0,                // Step count (0 means continuous)
		0.4,              // Default value (in normalized range [0,1])
		Vst::ParameterInfo::kCanAutomate, // Flags (this one makes it automatable)
		0,                // Parameter ID
		0,                // Parameter group (optional)
		STR16("Comp1"));   // Short title (optional)
	parameters.addParameter(
		STR16("Compression2"),
		nullptr,          
		0,                
		0.4,              
		Vst::ParameterInfo::kCanAutomate, 
		1,                
		0,                
		STR16("Comp2"));   
	parameters.addParameter(
		STR16("Volum"),   
		nullptr,          
		0,               
		1.0,              
		Vst::ParameterInfo::kCanAutomate, 
		2,                
		0,              
		STR16("Vol"));   

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
	if (FIDStringsEqual (name, Vst::ViewType::kEditor))
	{
		// create your editor here and return a IPlugView ptr of it
		auto* view = new VSTGUI::VST3Editor (this, "view", "myplugineditor.uidesc");
		return view;
	}
	return nullptr;
}

//------------------------------------------------------------------------
} // namespace cudaCompressor
