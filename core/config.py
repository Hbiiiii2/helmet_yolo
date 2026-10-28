import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Model Configuration
DEFAULT_MODEL_PATH = str(BASE_DIR / "content" / "runs" / "detect" / "runs_colab" / "helmet" / "helmet_yolo-2" / "weights" / "best.pt")
FALLBACK_MODEL_PATH = str(BASE_DIR / "models" / "best.pt")

def get_model_path():
    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    if os.path.exists(FALLBACK_MODEL_PATH):
        return FALLBACK_MODEL_PATH
    return DEFAULT_MODEL_PATH

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.35
OUTDOOR_CANDIDATE_CONF = 0.22
IOU_THRESHOLD = 0.45

# K3 Helmet Standard Strict Color Definitions in HSV (Slide P04: Model Warna)
# Format: [H_min, S_min, V_min], [H_max, S_max, V_max]
# Dikalibrasi agar tahan terhadap Backlight (bayangan) & Glare Matahari (outdoor)
# namun tetap mengecualikan cokelat/khaki topi dan warna kulit wajah
K3_COLOR_RANGES = {
    "Kuning K3 (Pekerja/Kontraktor)": {
        "lower": (18, 82, 80),
        "upper": (38, 255, 255),
        "bgr_preview": (0, 215, 255)  # Gold/Yellow
    },
    "Putih K3 (Supervisor/Engineer/Tamu)": {
        "is_rgb_balanced": True,
        "min_brightness": 145,
        "max_diff": 32,
        "max_saturation": 46,
        "bgr_preview": (245, 245, 245)  # Pure White
    },
    "Merah K3 (Safety Officer/K3/Damkar)": [
        {"lower": (0, 115, 75), "upper": (8, 255, 255)},
        {"lower": (166, 105, 75), "upper": (180, 255, 255)}
    ],
    "Biru K3 (Teknisi Listrik/Mekanik)": {
        "lower": (85, 55, 50),
        "upper": (138, 255, 255),
        "bgr_preview": (230, 115, 0)  # Blue
    },
    "Hijau K3 (Inspektur Lingkungan)": {
        "lower": (38, 55, 55),
        "upper": (85, 255, 255),
        "bgr_preview": (50, 205, 50)  # Lime/Green
    },
    "Oranye K3 (Operator Alat Berat/Surveyor)": {
        "lower": (8, 135, 85),
        "upper": (19, 255, 255),
        "bgr_preview": (0, 140, 255)  # Orange
    }
}

# Classification Labels
LABEL_REAL_HELMET = "REAL_HELMET"
LABEL_TOPI = "TOPI"
LABEL_HELM_MOTOR = "HELM_MOTOR"
LABEL_HELM_SEPEDA = "HELM_SEPEDA"
LABEL_KERUDUNG = "KERUDUNG"
LABEL_NO_HELMET = "NO_HELMET"

STATUS_META = {
    LABEL_REAL_HELMET: {
        "title": "HELM PROYEK RESMI (K3)",
        "badge": "COMPLIANT",
        "color": "#10b981", # Green
        "icon": "shield-check",
        "description": "Memenuhi standar K3 (kubah plastik kaku HDPE/ABS, warna standar, telinga terbuka)."
    },
    LABEL_TOPI: {
        "title": "PERINGATAN: HANYA TOPI (BUKAN HELM K3)",
        "badge": "VIOLATION",
        "color": "#f59e0b", # Amber/Orange
        "icon": "exclamation-triangle",
        "description": "Menggunakan topi biasa/baseball cap berlidah. Dilarang di area proyek tanpa helm keselamatan!"
    },
    LABEL_HELM_MOTOR: {
        "title": "PERINGATAN: HELM MOTOR (BUKAN HELM K3)",
        "badge": "VIOLATION",
        "color": "#06b6d4", # Cyan
        "icon": "shield-slash",
        "description": "Menggunakan helm motor (full face / half face). Dilarang di proyek karena membatasi visibilitas & pendengaran pekerja!"
    },
    LABEL_HELM_SEPEDA: {
        "title": "PERINGATAN: HELM SEPEDA (BUKAN HELM K3)",
        "badge": "VIOLATION",
        "color": "#eab308", # Yellow
        "icon": "bicycle",
        "description": "Menggunakan helm sepeda berpori ventilasi aerodinamis. Tidak tahan terhadap benturan benda jatuh vertikal K3!"
    },
    LABEL_KERUDUNG: {
        "title": "PERINGATAN: KERUDUNG / HIJAB (WAJIB PAKAI HELM)",
        "badge": "VIOLATION",
        "color": "#8b5cf6", # Purple
        "icon": "exclamation-circle",
        "description": "Menggunakan hijab/kerudung tanpa helm safety di luarnya. Wajib memakai helm proyek K3!"
    },
    LABEL_NO_HELMET: {
        "title": "BAHAYA: TANPA HELM KESELAMATAN",
        "badge": "DANGER",
        "color": "#ef4444", # Red
        "icon": "x-circle",
        "description": "Kepala tanpa pelindung sama sekali di area wajib K3!"
    }
}
