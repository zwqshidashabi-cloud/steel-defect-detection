@echo off
chcp 65001 >nul
echo ============================================
echo   NEU-DET 钢材缺陷数据集准备
echo ============================================
echo.
D:\Anaconda\python.exe D:\projects\steel_detection\scripts\download_neu_det.py
echo.
pause
