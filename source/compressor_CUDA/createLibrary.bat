@echo off

goto main


:handleError
echo An error occurred. Exiting script.
pause
exit /b 1


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
echo Building library...
nvcc -shared -lcudart -o Compressor.dll ..\src\Compressor.cu -D COMPRESSOR_EXPORTS
if %ERRORLEVEL% neq 0 call :handleError

echo.
echo Building test...
cl ..\src\test.cpp /link /out:test.exe /LIBPATH:".\" Compressor.lib
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

move build\Compressor.lib lib\Compressor.lib
if %ERRORLEVEL% neq 0 call :handleError

echo.
echo Done.
pause