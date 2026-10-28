"""
FastAPI Backend Application
Sistem Deteksi Helm Proyek K3 (Real Helm vs Topi vs Kerudung vs Tanpa Helm)
Ready-to-Deploy dengan Dashboard Interaktif, Streaming Kamera, Upload Citra/Video, dan REST API.
"""

import os
import cv2
import base64
import time
import json
from pathlib import Path
from typing import Optional, List
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.yolo_detector import SafetyHelmetDetector
from core.config import (
    BASE_DIR,
    LABEL_REAL_HELMET,
    LABEL_TOPI,
    LABEL_KERUDUNG,
    LABEL_NO_HELMET,
    STATUS_META,
    K3_COLOR_RANGES
)

# Inisialisasi FastAPI
app = FastAPI(
    title="K3 Helmet Detection & Verification System",
    description="Sistem Cerdas Deteksi K3 Helm Proyek Menggunakan YOLO11 dan Pengolahan Citra Digital (Slide BSI P01-P11)",
    version="2.0.0"
)

# Folder paths
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
UPLOADS_DIR = BASE_DIR / "uploads"
SAMPLES_DIR = BASE_DIR / "samples"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

# Mount static files & templates
if SAMPLES_DIR.exists():
    app.mount("/static/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global detector instance
print("[*] Menginisialisasi SafetyHelmetDetector...")
detector = SafetyHelmetDetector()
print("[+] SafetyHelmetDetector Siap Digunakan.")

# Session violation history
audit_history = []


def encode_image_base64(img: np.ndarray) -> str:
    """Mengonversi citra OpenCV BGR ke string Base64 Data URL."""
    if img is None or img.size == 0:
        return ""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def decode_base64_image(b64_data: str) -> np.ndarray:
    """Mengonversi Base64 string ke citra OpenCV BGR."""
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    img_bytes = base64.b64decode(b64_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Halaman Dashboard Utama."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "K3 Smart Helmet Surveillance System"}
    )


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204, content={})


@app.get("/health")
async def health_check():
    """Health check endpoint untuk Railway dan Docker container orchestration."""
    return {
        "status": "healthy",
        "service": "k3-helmet-yolo",
        "detector_ready": detector is not None
    }


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.35)
):
    """
    Endpoint deteksi citra lengkap dengan visualisasi tahapan Pengolahan Citra:
    1. Original ROI
    2. Image Enhancement (CLAHE - Slide P03)
    3. Segmentasi Warna HSV (Slide P04)
    4. Deteksi Tepi Canny & Sobel (Slide P07)
    5. Morfologi Citra (Slide P11)
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="File gambar tidak valid.")

        results = detector.detect(img, conf=conf)
        annotated_b64 = encode_image_base64(results["annotated_frame"])

        # Kemas detail deteksi dan potong citra diagnostik untuk setiap kepala
        detections_detail = []
        for d in results["detections"]:
            cls_info = d["classification"]
            features = cls_info["features"]
            
            # Format visualisasi diagnostik per materi slide BSI
            crop_b64 = encode_image_base64(d.get("crop"))
            enhanced_b64 = encode_image_base64(features.get("enhanced_roi"))
            color_mask_b64 = encode_image_base64(cv2.cvtColor(features.get("color_mask"), cv2.COLOR_GRAY2BGR) if features.get("color_mask") is not None else None)
            canny_b64 = encode_image_base64(cv2.cvtColor(features.get("canny_img"), cv2.COLOR_GRAY2BGR) if features.get("canny_img") is not None else None)

            det_item = {
                "bbox": d["bbox"],
                "label": cls_info["label"],
                "title": cls_info["title"],
                "badge": cls_info["badge"],
                "confidence": round(cls_info["confidence"] * 100, 1),
                "color_hex": cls_info["color_hex"],
                "description": cls_info["description"],
                "detected_k3_color": cls_info["detected_k3_color"],
                "k3_color_ratio": round(cls_info["k3_color_ratio"] * 100, 1),
                "edge_density": round(cls_info["edge_density"] * 100, 2),
                "drapery_ratio": round(cls_info["drapery_ratio"], 2),
                "solidity": round(features.get("solidity", 0.0), 2),
                "symmetry_score": round(features.get("symmetry_score", 0.0), 2),
                "neck_coverage": round(features.get("neck_coverage", 0.0) * 100, 2),
                "vent_count": int(cls_info.get("vent_count", 0)),
                "visor_shadow": round(float(cls_info.get("visor_shadow", 0.0)) * 100, 1),
                "dark_visor_ratio": round(float(cls_info.get("dark_visor_ratio", 0.0)) * 100, 1),
                "hair_ratio": round(float(cls_info.get("hair_ratio", 0.0)) * 100, 1),
                "lighting_condition": cls_info.get("lighting_condition", "Normal"),
                "diagnostics": {
                    "original_crop": crop_b64,
                    "enhanced_roi": enhanced_b64,
                    "color_mask": color_mask_b64,
                    "canny_edges": canny_b64
                }
            }
            detections_detail.append(det_item)

            # Rekam ke audit log jika ada pelanggaran
            if cls_info["label"] != LABEL_REAL_HELMET:
                audit_history.insert(0, {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "violation_type": cls_info["title"],
                    "badge": cls_info["badge"],
                    "color_hex": cls_info["color_hex"],
                    "thumbnail": crop_b64,
                    "details": f"Warna K3: {cls_info['detected_k3_color']}, Densitas Tepi: {det_item['edge_density']}%"
                })

        return {
            "status": "success",
            "annotated_image": annotated_b64,
            "stats": results["stats"],
            "detections": detections_detail
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/frame")
async def detect_frame(payload: dict):
    """
    Endpoint real-time frame webcam (dikirim dari browser via Canvas/Fetch).
    Sangat responsif dan ringan untuk video streaming interaktif.
    """
    try:
        frame_b64 = payload.get("image")
        conf = float(payload.get("conf", 0.35))
        if not frame_b64:
            raise HTTPException(status_code=400, detail="Data gambar tidak ada.")

        frame = decode_base64_image(frame_b64)
        if frame is None:
            raise HTTPException(status_code=400, detail="Gagal decode frame.")

        results = detector.detect(frame, conf=conf)
        annotated_b64 = encode_image_base64(results["annotated_frame"])

        # Cek apakah ada pelanggaran aktif di frame ini
        has_violation = results["stats"].get("violations", 0) > 0
        violation_types = []
        for d in results["detections"]:
            lbl = d["classification"]["label"]
            if lbl != LABEL_REAL_HELMET:
                violation_types.append(d["classification"]["title"])

        return {
            "status": "success",
            "annotated_frame": annotated_b64,
            "stats": results["stats"],
            "has_violation": has_violation,
            "violation_types": list(set(violation_types))
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/audit/logs")
async def get_audit_logs():
    """Mengambil riwayat log pelanggaran K3 terbaru."""
    return {"logs": audit_history[:50]}


@app.post("/api/audit/clear")
async def clear_audit_logs():
    """Mengosongkan log audit pelanggaran."""
    global audit_history
    audit_history.clear()
    return {"status": "success", "message": "Log audit berhasil dibersihkan."}


@app.get("/api/samples")
async def get_sample_list():
    """Daftar sampel gambar untuk demonstrasi dan evaluasi 1-klik."""
    samples = []
    for p in SAMPLES_DIR.glob("*.*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            samples.append({
                "filename": p.name,
                "url": f"/static/samples/{p.name}"
            })
    return {"samples": samples}


@app.get("/api/theory")
async def get_theory_mapping():
    """
    Korelasi Ilmiah Sistem dengan Materi Perkuliahan Pengolahan Citra BSI:
    Memberikan bahan argumentasi akademis lengkap untuk presentasi kelompok (Pertemuan 12-15).
    """
    return {
        "course": "Pengolahan Citra Digital (Project Based Learning)",
        "modules": [
            {
                "slide": "Slide P02: Spatial Filtering",
                "concept": "Konvolusi Kernel, Smoothing & Sharpening",
                "implementation": "Menggunakan Gaussian Blur (5x5, sigma=1.0) untuk mereduksi noise kamera sebelum ekstraksi gradien tepi, serta Laplacian filtering untuk memisahkan detail kain halus dari pantulan plastik."
            },
            {
                "slide": "Slide P03: Peningkatan Kualitas Citra (Image Enhancement)",
                "concept": "Perbaikan Kontras & Histogram Equalization",
                "implementation": "Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization) pada channel Luminance LAB (tile 8x8, clip limit 2.5) untuk menormalkan pencahayaan ekstrem di lapangan proyek (terik matahari/bayangan)."
            },
            {
                "slide": "Slide P04: Model Warna (Color Models)",
                "concept": "Ruang Warna RGB vs HSV & Color Slicing",
                "implementation": "Mengonversi ROI ke HSV untuk segmentasi warna helm standar K3 (Kuning, Putih, Oranye, Merah, Biru, Hijau). Membedakan helm K3 dari topi kasual (hitam/jeans/pola) dan kerudung."
            },
            {
                "slide": "Slide P07: Pendeteksian Tepi (Edge Detection)",
                "concept": "Operator Sobel & Deteksi Tepi Canny",
                "implementation": "Menghitung densitas tepi interior dan rasio gradien Sobel-Y terhadap Sobel-X. Helm proyek memiliki permukaan plastik halus (edge density rendah). Kerudung memiliki lipatan kain vertikal (Sobel-Y > Sobel-X)."
            },
            {
                "slide": "Slide P09: Segmentasi Citra & P11: Morfologi Citra",
                "concept": "Thresholding Otsu, Erosi, Dilatasi, Opening, Closing, Convex Hull",
                "implementation": "Morfologi Opening untuk membersihkan noise rambut, Closing untuk menutup lubang pantulan kilau. Analisis kekompakan bentuk (Solidity) dan simetri horizontal untuk mendeteksi lidah topi (visor)."
            }
        ]
    }
