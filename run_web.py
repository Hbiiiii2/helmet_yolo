"""
Script Runner untuk Meluncurkan Web Dashboard K3 Helmet Detection
Jalankan: python run_web.py
Akses di browser: http://127.0.0.1:8000
"""

import uvicorn
import sys
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print("=" * 65)
    print("  SISTEM MONITORING K3 - DETEKSI HELM PROYEK & PENUTUP KEPALA")
    print("  Integrasi YOLO11 + Digital Image Processing (Slide BSI P01-P11)")
    print("=" * 65)
    print(f"[*] Server berjalan di: http://{host}:{port}")
    print("[*] Tekan CTRL + C di terminal untuk menghentikan server.")
    print("=" * 65)
    uvicorn.run("web.app:app", host=host, port=port, reload=False)
