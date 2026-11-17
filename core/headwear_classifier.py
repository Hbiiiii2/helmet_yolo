"""
Modul HeadwearClassifier (Ensemble Decision Engine)
Menggabungkan output deep learning YOLO dengan ekstraksi fitur Pengolahan Citra Digital
untuk mendeteksi secara presisi:
1. REAL_HELMET : Helm Proyek K3 Resmi (Kubah Plastik Kaku, Warna K3 Murni, Telinga Terbuka)
2. TOPI        : Topi Biasa / Baseball Cap / Beanie (Lidah Visor, Kain Khaki/Camo)
3. HELM_MOTOR  : Helm Motor Full/Half Face (Kaca Visor, Cangkang Tebal Menutupi Rahang)
4. HELM_SEPEDA : Helm Sepeda Aerodinamis (Pori-Pori Ventilasi Berulang)
5. KERUDUNG    : Hijab / Kerudung (Kain Drapery Jatuh Menutupi Leher)
6. NO_HELMET   : Tanpa Pelindung Kepala (Bare Head)
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .config import (
    LABEL_REAL_HELMET,
    LABEL_TOPI,
    LABEL_HELM_MOTOR,
    LABEL_HELM_SEPEDA,
    LABEL_KERUDUNG,
    LABEL_NO_HELMET,
    STATUS_META
)
from .image_processing import ImageProcessingPipeline


class HeadwearClassifier:
    """
    Sistem verifikasi multi-tahap (Multi-stage Verification System)
    untuk membedakan penutup kepala asli K3 vs aksesoris non-standar.
    """

    def __init__(self):
        self.ip_pipeline = ImageProcessingPipeline()

    def classify_headwear(
        self,
        yolo_class: str,
        yolo_conf: float,
        head_roi: np.ndarray,
        person_crop: np.ndarray = None,
        neck_roi: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Mengklasifikasikan status penutup kepala menggunakan fusi YOLO + Pengolahan Citra.
        """
        # Ekstraksi fitur citra digital mendalam
        features = self.ip_pipeline.inspect_patch(head_roi, person_crop, neck_roi=neck_roi)
        
        k3_ratio = features.get("k3_color_ratio", 0.0)
        edge_density = features.get("edge_density", 0.0)
        drapery_ratio = features.get("drapery_ratio", 1.0)
        solidity = features.get("solidity", 0.8)
        symmetry = features.get("symmetry_score", 0.5)
        neck_coverage = features.get("neck_coverage", 0.0)
        vent_count = features.get("vent_count", 0)
        dome_edge_density = features.get("dome_edge_density", 0.0)
        visor_shadow = features.get("visor_shadow", 0.0)
        is_khaki = features.get("is_khaki_tone", False)
        h_std = features.get("h_std", 0.0)
        dark_visor_ratio = features.get("dark_visor_ratio", 0.0)
        visor_horizontal_edge = features.get("visor_horizontal_edge", 0.0)
        visor_band_ratio = features.get("visor_band_ratio", 1.0)
        ear_skin_ratio = features.get("ear_skin_ratio", 0.0)
        hair_ratio = features.get("hair_ratio", 0.0)
        skin_ratio = features.get("skin_ratio", 0.0)
        r_minus_b = features.get("r_minus_b", 0.0)
        is_hijab_structural = features.get("is_hijab", False)
        color_continuity = features.get("color_continuity", False)

        # -------------------------------------------------------------
        # 1. EVALUASI CIRI FISIK (Slide P02-P11) TANPA SYARAT WARNA HELM
        # -------------------------------------------------------------
        # Kerudung / Hijab: Kain penutup kepala berlanjut ke samping & leher tanpa kulit leher terbuka
        is_kerudung = is_hijab_structural and (color_continuity or edge_density > 0.10 or drapery_ratio > 1.15)

        # Helm Sepeda: Pori ventilasi berulang (>= 3) pada kubah atas DAN bukan rambut keriting
        is_helm_sepeda = (hair_ratio < 0.30) and ((vent_count >= 3 and dome_edge_density > 0.10) or (vent_count >= 4 and dome_edge_density > 0.07))
        
        # Helm Motor: Kaca visor gelap menutupi area wajah (>= 28% atau >= 20% + tepi horizontal tegas)
        is_helm_motor = (
            (dark_visor_ratio >= 0.28) or
            (dark_visor_ratio >= 0.20 and visor_horizontal_edge > 25.0)
        )

        # Topi Biasa / Baseball Cap:
        # Karakteristik lidah visor memproyeksikan bayangan gelap di dahi bawah visor,
        # tekstur/corak kain non-plastik
        has_visor_shadow = visor_shadow >= 0.18
        is_topi = (
            (has_visor_shadow and (is_khaki or r_minus_b > 22.0 or visor_horizontal_edge > 25.0)) or
            (is_khaki and h_std > 38.0 and edge_density > 0.18)
        )

        # Helm Proyek Resmi K3:
        # Tidak membatasi warna helm; pastikan bukan aksesoris non-K3 dan tanpa rambut/kulit ubun-ubun terbuka
        is_real_helmet = (hair_ratio < 0.25) and (skin_ratio < 0.25) and (not is_helm_sepeda) and (not is_helm_motor) and (not is_topi) and (not is_kerudung)

        # -------------------------------------------------------------
        # 2. ENSEMBLE DECISION DENGAN PRIORITAS KESELAMATAN K3
        # -------------------------------------------------------------
        if yolo_class == "helmet":
            # Jika kubah atas jelas didominasi rambut + kulit dahi terbuka (bukan kerudung),
            # maka itu kepala tanpa helm meskipun YOLO salah deteksi sebagai helmet
            if not is_kerudung and ((hair_ratio >= 0.50 and skin_ratio >= 0.05) or (yolo_conf < 0.55 and hair_ratio >= 0.45 and not is_helm_motor)):
                final_label = LABEL_NO_HELMET
                confidence = min(0.92, max(0.65, hair_ratio))
            elif is_helm_sepeda:
                final_label = LABEL_HELM_SEPEDA
                confidence = min(0.96, max(yolo_conf, 0.72))
            elif is_helm_motor:
                final_label = LABEL_HELM_MOTOR
                confidence = min(0.96, max(yolo_conf, 0.74))
            elif is_kerudung:
                final_label = LABEL_KERUDUNG
                confidence = min(0.96, max(yolo_conf, 0.75))
            elif is_topi:
                final_label = LABEL_TOPI
                confidence = min(0.95, max(yolo_conf, 0.70))
            else:
                final_label = LABEL_REAL_HELMET
                confidence = min(0.98, yolo_conf + 0.08)

        elif yolo_class == "no-helmet":
            # Cek kulit kepala/dahi terbuka dulu, lalu hijab (agar hijab gelap tidak dikira rambut), lalu rambut
            if skin_ratio >= 0.35:
                final_label = LABEL_NO_HELMET
                confidence = min(0.96, max(yolo_conf, 0.75))
            elif is_kerudung:
                final_label = LABEL_KERUDUNG
                confidence = min(0.95, max(yolo_conf, 0.75))
            elif hair_ratio >= 0.15:
                final_label = LABEL_NO_HELMET
                confidence = min(0.96, max(yolo_conf, 0.75))
            elif is_topi:
                final_label = LABEL_TOPI
                confidence = min(0.92, max(yolo_conf, 0.68))
            elif is_helm_sepeda and hair_ratio < 0.10 and skin_ratio < 0.20 and vent_count >= 4:
                final_label = LABEL_HELM_SEPEDA
                confidence = min(0.92, max(yolo_conf, 0.68))
            elif is_helm_motor and hair_ratio < 0.15 and skin_ratio < 0.25:
                final_label = LABEL_HELM_MOTOR
                confidence = min(0.92, max(yolo_conf, 0.68))
            else:
                final_label = LABEL_NO_HELMET
                confidence = min(0.95, yolo_conf + 0.05)
        else:
            # Fallback jika yolo_class tidak terdefinisi
            if hair_ratio >= 0.15:
                final_label = LABEL_NO_HELMET
                confidence = 0.88
            elif is_real_helmet:
                final_label = LABEL_REAL_HELMET
                confidence = 0.85
            elif is_topi:
                final_label = LABEL_TOPI
                confidence = 0.80
            else:
                final_label = LABEL_NO_HELMET
                confidence = 0.75

        meta = STATUS_META[final_label]

        return {
            "label": final_label,
            "confidence": float(confidence),
            "title": meta["title"],
            "badge": meta["badge"],
            "color_hex": meta["color"],
            "description": meta["description"],
            "detected_k3_color": "Helm Proyek K3",
            "k3_color_ratio": float(k3_ratio),
            "edge_density": float(edge_density),
            "drapery_ratio": float(drapery_ratio),
            "vent_count": int(vent_count),
            "visor_shadow": float(visor_shadow),
            "dark_visor_ratio": float(dark_visor_ratio),
            "hair_ratio": float(hair_ratio),
            "skin_ratio": float(skin_ratio),
            "features": features
        }
