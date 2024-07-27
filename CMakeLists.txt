cmake_minimum_required(VERSION 3.14.0)

option(SMTG_ENABLE_VST3_PLUGIN_EXAMPLES "Enable VST 3 Plug-in Examples" OFF)
option(SMTG_ENABLE_VST3_HOSTING_EXAMPLES "Enable VST 3 Hosting Examples" OFF)

set(CMAKE_OSX_DEPLOYMENT_TARGET 10.13 CACHE STRING "")

set(vst3sdk_SOURCE_DIR "C:/Program Files/vst3sdk")
if(NOT vst3sdk_SOURCE_DIR)
    message(FATAL_ERROR "Path to VST3 SDK is empty!")
endif()

project(CuPressor
    # This is your plug-in version number. Change it here only.
    # Version number symbols usable in C++ can be found in
    # source/version.h and ${PROJECT_BINARY_DIR}/projectversion.h.
    VERSION 1.1.0.0 
    DESCRIPTION "CuPressor VST 3 Plug-in"
)

set(SMTG_VSTGUI_ROOT "${vst3sdk_SOURCE_DIR}")

add_subdirectory(${vst3sdk_SOURCE_DIR} ${PROJECT_BINARY_DIR}/vst3sdk)
smtg_enable_vst3_sdk()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
file(COPY "${CMAKE_SOURCE_DIR}/source/compressor_CUDA/lib/Compressor.dll"
     DESTINATION "${CMAKE_BINARY_DIR}/bin/Release")


smtg_add_vst3plugin(CuPressor
source/version.h
source/myplugincids.h
source/mypluginprocessor.h
source/mypluginprocessor.cpp
source/myplugincontroller.h
source/myplugincontroller.cpp
source/mypluginentry.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/source/compressor_CUDA/src)
target_link_libraries(CuPressor PRIVATE ${CMAKE_SOURCE_DIR}/source/compressor_CUDA/lib/Compressor.lib)


# include_directories(
#     "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include"
#     ${CMAKE_SOURCE_DIR}/source/compressor_CUDA/src
# )
# target_link_libraries(CuPressor PRIVATE
#     ${CMAKE_SOURCE_DIR}/source/compressor_CUDA/lib/Compressor.lib
#     "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/lib/x64/cudart.lib"
# )

#- VSTGUI Wanted ----
if(SMTG_ENABLE_VSTGUI_SUPPORT)
    target_sources(CuPressor
        PRIVATE
            resource/myplugineditor.uidesc
    )
    target_link_libraries(CuPressor
        PRIVATE
            vstgui_support
    )
    smtg_target_add_plugin_resources(CuPressor
        RESOURCES
            "resource/myplugineditor.uidesc"
    )
endif(SMTG_ENABLE_VSTGUI_SUPPORT)
# -------------------

smtg_target_add_plugin_snapshots (CuPressor
    RESOURCES
        resource/0E00A5612C6458BEB7608BAB42192C36_snapshot.png
        resource/0E00A5612C6458BEB7608BAB42192C36_snapshot_2.0x.png
)

# add_custom_command(TARGET CuPressor POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${CMAKE_SOURCE_DIR}/source/compressor_CUDA/lib/Compressor.dll"
#     "$<TARGET_FILE_DIR:CuPressor>/../../Compressor.dll"
# )

target_link_libraries(CuPressor
    PRIVATE
        sdk
)

smtg_target_configure_version_file(CuPressor)

if(SMTG_MAC)
    smtg_target_set_bundle(CuPressor
        BUNDLE_IDENTIFIER com.stimmer.cudacupressor
        COMPANY_NAME "stimmer02"
    )
    smtg_target_set_debug_executable(CuPressor
        "/Applications/VST3PluginTestHost.app"
        "--pluginfolder;$(BUILT_PRODUCTS_DIR)"
    )
elseif(SMTG_WIN)
    target_sources(CuPressor PRIVATE 
        resource/win32resource.rc
    )
    if(MSVC)
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT CuPressor)

        smtg_target_set_debug_executable(CuPressor
            "$(ProgramW6432)/Steinberg/VST3PluginTestHost/VST3PluginTestHost.exe"
            "--pluginfolder \"$(OutDir)/\""
        )
    endif()
endif(SMTG_MAC)