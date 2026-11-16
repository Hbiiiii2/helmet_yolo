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
IOU_THRESHOLD = 0.45

# Definisi rentang warna helm proyek dihapus agar semua warna helm proyek
# terdeteksi seragam sebagai satu kelas tunggal (HELM PROYEK K3).
K3_COLOR_RANGES = {}

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
        "description": "Memenuhi standar K3 (kubah plastik kaku HDPE/ABS, pelindung kepala utuh, telinga terbuka)."
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
