@echo off

goto main


:handleError
echo An error occurred. Exiting script.
pause
exit 1


:main
echo.
echo Setting up environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if %ERRORLEVEL% neq 0 call :handleError

echo.
echo Preparing directories...
if exist lib ( rmdir /s /q lib )
if %ERRORLEVEL% neq 0 call :handleError

mkdir lib
if %ERRORLEVEL% neq 0 call :handleError

if exist build ( rmdir /s /q build )
if %ERRORLEVEL% neq 0 call :handleError

mkdir build
if %ERRORLEVEL% neq 0 call :handleError

cd build
if %ERRORLEVEL% neq 0 call :handleError

echo.
echo Creating dynamic library...
@REM nvcc -shared -lcudart -lcufft -o Compressor.dll -D COMPRESSOR_EXPORTS ..\src\test.cpp ..\src\Compressor.cu ..\src\buffer\CuShiftBuffer.cu ..\src\buffer\ACuBuffer.cu ..\src\buffer\ACuSimpleBuffer.cu ..\src\buffer\CuSimpleBuffer_memoryOptimal.cu ..\src\buffer\CuSimpleBuffer_timeOptimal.cu ..\src\buffer\templateInstantiation.cu
nvcc -shared -lcudart -lcufft -o Compressor.dll -D COMPRESSOR_EXPORTS ..\src\test.cpp ..\src\Compressor2.cpp ..\src\buffer\CuShiftBuffer.cu ..\src\buffer\ACuBuffer.cu ..\src\buffer\ACuSimpleBuffer.cu ..\src\buffer\CuSimpleBuffer_memoryOptimal.cu ..\src\buffer\CuSimpleBuffer_timeOptimal.cu ..\src\buffer\templateInstantiation.cu ..\src\processing\ProcessingQueue.cpp ..\src\processing\processingUnits\AProcessingUnit.cpp ..\src\processing\processingUnits\ProcessingUnit_cuPressor.cu ..\src\processing\processingUnits\ProcessingUnit_volume.cu
if %ERRORLEVEL% neq 0 call :handleError


@REM echo.
@REM echo Compiling CUDA source files...
@REM nvcc -c -o Compressor.obj ..\src\Compressor.cu
@REM if %ERRORLEVEL% neq 0 call :handleError

@REM echo.
@REM echo Creating static library...
@REM nvcc -lib -o Compressor.lib Compressor.obj
@REM if %ERRORLEVEL% neq 0 call :handleError

echo.
echo Building test...
cl ..\src\test.cpp /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\include" /link /out:test.exe /LIBPATH:".\" Compressor.lib
@REM cl ..\src\test.cpp /link /out:test.exe /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\lib\x64" Compressor.lib cudart.lib
if %ERRORLEVEL% neq 0 call :handleError

cd ..
if %ERRORLEVEL% neq 0 call :handleError

echo.
echo Running test...
call ".\build\test.exe"
if %ERRORLEVEL% neq 0 call :handleError

echo.
echo Moving files...
move build\Compressor.dll lib\Compressor.dll
if %ERRORLEVEL% neq 0 call :handleError

@REM move build\Compressor.obj lib\Compressor.obj
@REM if %ERRORLEVEL% neq 0 call :handleError

move build\Compressor.lib lib\Compressor.lib
if %ERRORLEVEL% neq 0 call :handleError

@REM if exist ..\..\build\VST3\Release\CuPressor.vst3\ ( cp lib\Compressor.dll ..\..\build\VST3\Release\CuPressor.vst3\ )
cp lib\Compressor.dll C:\Windows\System32\
if %ERRORLEVEL% neq 0 call :handleError

echo.
echo Done.
pause