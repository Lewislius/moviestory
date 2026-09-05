@echo off
REM =============================================================================
REM MetaQuery + Qwen3-VL: Windows Training Launch Script
REM =============================================================================
REM Usage:
REM   scripts\train_qwen3vl.bat <run_name> <config_file> <base_dir> [num_gpus]
REM
REM Examples:
REM   scripts\train_qwen3vl.bat qwen3vl2b_t2i qwen3vl2b_sana.yaml E:\data\metaquery
REM   scripts\train_qwen3vl.bat qwen3vl2b_inst qwen3vl2b_sana_inst.yaml E:\data\metaquery 4
REM =============================================================================

setlocal

set RUN_NAME=%1
set CONFIG_FILE=%2
set BASE_DIR=%3
set NUM_GPUS=%4

if "%RUN_NAME%"=="" set RUN_NAME=qwen3vl2b_t2i
if "%CONFIG_FILE%"=="" set CONFIG_FILE=qwen3vl2b_sana.yaml
if "%BASE_DIR%"=="" set BASE_DIR=E:\data\metaquery
if "%NUM_GPUS%"=="" set NUM_GPUS=1

echo =============================================
echo MetaQuery + Qwen3-VL Training (Windows)
echo =============================================
echo Run Name:    %RUN_NAME%
echo Config:      %CONFIG_FILE%
echo Base Dir:    %BASE_DIR%
echo Num GPUs:    %NUM_GPUS%
echo =============================================

set OMP_NUM_THREADS=8

torchrun ^
    --nproc-per-node=%NUM_GPUS% ^
    train.py ^
    --run_name %RUN_NAME% ^
    --config_file %CONFIG_FILE% ^
    --base_dir %BASE_DIR%

endlocal
