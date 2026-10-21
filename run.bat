@echo off
title K3 Helmet Detection System - Deployment Runner
color 0A
echo ======================================================================
echo       SISTEM DETEKSI HELM PROYEK K3 (DEPLOYMENT READY)
echo       Integrasi YOLO11 + Pengolahan Citra Digital (Slide BSI P01-P11)
echo ======================================================================
echo.
echo Memilih Mode Peluncuran:
echo [1] Buka Dashboard Web (Rekomendasi untuk Presentasi & CCTV Webcam)
echo [2] Buka OpenCV Desktop Webcam Langsung
echo [3] Jalankan Deteksi pada Citra Sampel
echo [4] Keluar
echo.
set /p choice="Masukkan pilihan (1/2/3/4): "

if "%choice%"=="1" (
    echo.
    echo [*] Membuka Web Dashboard di http://127.0.0.1:8000 ...
    start http://127.0.0.1:8000
    python run_web.py
) else if "%choice%"=="2" (
    echo.
    echo [*] Membuka Webcam Desktop OpenCV...
    python detect_cam.py
) else if "%choice%"=="3" (
    echo.
    echo [*] Menjalankan deteksi pada samples/sample_construction_team.jpg...
    python detect_image.py -i samples/sample_construction_team.jpg -o output_demo.jpg
    start output_demo.jpg
) else (
    echo Keluar...
)
pause
