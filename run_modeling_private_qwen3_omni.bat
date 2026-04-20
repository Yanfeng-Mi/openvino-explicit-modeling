@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

for %%I in ("%SCRIPT_DIR%\..") do set "ROOT_DIR=%%~fI"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "OPENVINO_BIN=%ROOT_DIR%\openvino\bin\intel64\Release"
set "TBB_BIN=%ROOT_DIR%\openvino\temp\Windows_AMD64\tbb\bin"
set "GENAI_DLL_DIR=%ROOT_DIR%\openvino.genai\build\openvino_genai"
set "GENAI_RUNTIME_BIN_DIR=%ROOT_DIR%\openvino.genai\build\bin"
set "GENAI_RELEASE_BIN_DIR=%GENAI_RUNTIME_BIN_DIR%\Release"
set "BRIDGE_DIR=%ROOT_DIR%\modeling_private\models\qwen3_omni"
set "VENV_SCRIPTS=%ROOT_DIR%\.venv\Scripts"

if "%~1"=="" goto :usage

set "MODE=%~1"
shift

set "REPEAT_COUNT=1"
set "FORWARD_ARGS="
:collect_args
if "%~1"=="" goto :after_collect
if /i "%~1"=="--repeat" (
    if "%~2"=="" (
        echo [ERROR] --repeat requires a positive integer value.
        goto :usage_fail
    )
    set "REPEAT_COUNT=%~2"
    shift
    shift
    goto :collect_args
)
set "FORWARD_ARGS=!FORWARD_ARGS! %1"
shift
goto :collect_args

:after_collect
for /f "delims=0123456789" %%A in ("%REPEAT_COUNT%") do (
    echo [ERROR] Invalid --repeat value: %REPEAT_COUNT%
    goto :usage_fail
)
if "%REPEAT_COUNT%"=="" (
    echo [ERROR] Invalid --repeat value.
    goto :usage_fail
)
if "%REPEAT_COUNT%"=="0" (
    echo [ERROR] --repeat must be greater than 0.
    goto :usage_fail
)

if /i "%MODE%"=="image" (
    set "TARGET_NAME=modeling_private_qwen3_omni_modeling_qwen3_omni.exe"
) else if /i "%MODE%"=="tts" (
    set "TARGET_NAME=modeling_private_qwen3_omni_modeling_qwen3_omni_tts_min.exe"
) else (
    echo [ERROR] Unknown mode: %MODE%
    goto :usage_fail
)

if not exist "%OPENVINO_BIN%" (
    echo [ERROR] OpenVINO bin directory not found: %OPENVINO_BIN%
    exit /b 1
)

if not exist "%TBB_BIN%" (
    echo [ERROR] TBB bin directory not found: %TBB_BIN%
    exit /b 1
)

if not exist "%GENAI_DLL_DIR%" (
    echo [ERROR] OpenVINO GenAI DLL directory not found: %GENAI_DLL_DIR%
    exit /b 1
)

if not exist "%GENAI_RUNTIME_BIN_DIR%" (
    echo [ERROR] OpenVINO GenAI runtime bin directory not found: %GENAI_RUNTIME_BIN_DIR%
    exit /b 1
)

if not exist "%BRIDGE_DIR%\processing_qwen3_omni_bridge.py" (
    echo [ERROR] Qwen3 Omni bridge script not found: %BRIDGE_DIR%\processing_qwen3_omni_bridge.py
    exit /b 1
)

set "TARGET_EXE=%GENAI_RELEASE_BIN_DIR%\%TARGET_NAME%"
set "TARGET_DIR=%GENAI_RELEASE_BIN_DIR%"
if not exist "%TARGET_EXE%" (
    set "TARGET_EXE=%GENAI_RUNTIME_BIN_DIR%\%TARGET_NAME%"
    set "TARGET_DIR=%GENAI_RUNTIME_BIN_DIR%"
)

if not exist "%TARGET_EXE%" (
    echo [ERROR] Executable not found: %TARGET_NAME%
    echo         Reconfigure and rebuild openvino.genai after the modeling_private CMake hook is applied.
    exit /b 1
)

if exist "%VENV_SCRIPTS%\python.exe" (
    set "PYTHON_PATH_PREPEND=%VENV_SCRIPTS%;"
) else (
    set "PYTHON_PATH_PREPEND="
    echo [WARN] %VENV_SCRIPTS%\python.exe not found. Falling back to whatever 'python' resolves from PATH.
)

set "QWEN3_OMNI_BRIDGE_DIR=%BRIDGE_DIR%"
set "OV_GENAI_USE_MODELING_API=1"
set "PATH=%PYTHON_PATH_PREPEND%%OPENVINO_BIN%;%TBB_BIN%;%GENAI_DLL_DIR%;%GENAI_RUNTIME_BIN_DIR%;%TARGET_DIR%;%PATH%"

if not "%REPEAT_COUNT%"=="1" (
    set "FORWARD_ARGS=%FORWARD_ARGS% --benchmark-runs %REPEAT_COUNT%"
)

cd /d "%TARGET_DIR%"
echo [RUN] "%TARGET_EXE%"%FORWARD_ARGS%
call "%TARGET_EXE%"%FORWARD_ARGS%
exit /b %ERRORLEVEL%

:usage
echo Usage:
echo   %~nx0 image [--repeat N] --model-dir PATH --image PATH [--prompt TEXT] [--device NAME] [--output-tokens N] [--cache-model] [--vision-quant-mode MODE] [--vision-group-size N] [--vision-backup-mode MODE] [--text-quant-mode MODE] [--text-group-size N] [--text-backup-mode MODE]
echo   %~nx0 tts [--repeat N] MODEL_DIR CASE_ID TEXT_PROMPT WAV_OUT [IMAGE_PATH] [AUDIO_PATH] [DEVICE] [MAX_NEW_TOKENS] [PRECISION] [VIDEO_FRAMES_DIR] [--cache-model]
echo.
echo Examples:
echo   %~nx0 image --repeat 5 --model-dir D:\data\models\Huggingface\Qwen3-Omni --image D:\data\images\cat.jpg --prompt "What can you see" --device CPU --cache-model --vision-quant-mode int8_asym --vision-group-size 128 --text-quant-mode int4_asym --text-group-size 128 --text-backup-mode int8_asym
echo   %~nx0 tts D:\data\models\Huggingface\Qwen3-Omni demo "Describe this scene" D:\temp\omni.wav D:\data\images\cat.jpg "" CPU 64 fp32 --cache-model
exit /b 0

:usage_fail
call :usage
exit /b 1