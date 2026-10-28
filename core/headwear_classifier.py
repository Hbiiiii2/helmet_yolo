"""
Modul HeadwearClassifier (Ensemble Decision Engine)
Menggabungkan output deep learning YOLO dengan ekstraksi fitur Pengolahan Citra Digital
(Tahan Backlight & Noise Cahaya Matahari Outdoor) untuk mendeteksi secara presisi:
1. REAL_HELMET : Helm Proyek K3 Resmi (Kubah Plastik Kaku, Warna K3 Murni, Telinga Terbuka)
2. TOPI        : Topi Biasa / Baseball Cap / Beanie (Lidah Visor, Kain Khaki/Camo)
3. HELM_MOTOR  : Helm Motor Full/Half Face (Kaca Visor, Cangkang Tebal Menutupi Rahang)
4. HELM_SEPEDA : Helm Sepeda Aerodinamis (Pori-Pori Ventilasi Berulang)
5. KERUDUNG    : Hijab / Kerudung (Kain Drapery Jatuh Menutupi Leher)
6. NO_HELMET   : Tanpa Pelindung Kepala (Bare Head)
"""

import cv2
import numpy as np
from typing import Dict, Any
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
    untuk membedakan penutup kepala asli K3 vs aksesoris non-standar
    baik di dalam ruangan maupun kondisi ekstrem outdoor (backlight & terik matahari).
    """

    def __init__(self):
        self.ip_pipeline = ImageProcessingPipeline()

    def classify_headwear(
        self,
        yolo_class: str,
        yolo_conf: float,
        head_roi: np.ndarray,
        person_crop: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Mengklasifikasikan status penutup kepala menggunakan fusi YOLO + Pengolahan Citra.
        """
        features = self.ip_pipeline.inspect_patch(head_roi, person_crop)

        k3_ratio = features.get("k3_color_ratio", 0.0)
        best_color = features.get("best_color_name", "Bukan Warna K3")
        edge_density = features.get("edge_density", 0.0)
        drapery_ratio = features.get("drapery_ratio", 1.0)
        vent_count = features.get("vent_count", 0)
        dome_edge_density = features.get("dome_edge_density", 0.0)
        visor_shadow = features.get("visor_shadow", 0.0)
        is_khaki = features.get("is_khaki_tone", False)
        h_std = features.get("h_std", 0.0)
        dark_visor_ratio = features.get("dark_visor_ratio", 0.0)
        visor_horizontal_edge = features.get("visor_horizontal_edge", 0.0)
        hair_ratio = features.get("hair_ratio", 0.0)
        skin_ratio = features.get("skin_ratio", 0.0)
        r_minus_b = features.get("r_minus_b", 0.0)
        is_hijab_structural = features.get("is_hijab", False)
        lighting_condition = features.get("lighting_condition", "Normal")

        # -------------------------------------------------------------
        # 1. EVALUASI CIRI FISIK (Slide P02-P11, Tahan Backlight/Outdoor)
        # -------------------------------------------------------------
        # Kerudung / Hijab: Kain penutup kepala berlanjut ke leher tanpa kulit terbuka, bukan warna K3 kaku
        is_kerudung = is_hijab_structural and (best_color == "Bukan Warna K3" or k3_ratio < 0.20)

        # Helm Sepeda: Pori ventilasi berulang (>= 2) pada kubah atas
        is_helm_sepeda = (
            (vent_count >= 3 and dome_edge_density > 0.11) or
            (vent_count >= 2 and dome_edge_density > 0.16)
        )

        # Helm Motor: Kaca visor gelap mengilap menutupi area wajah + tepi horizontal tegas
        # Proteksi Backlight: Jika kubah atas memiliki warna K3 kuat (k3_ratio >= 0.25),
        # bayangan wajah di bawah helm tidak boleh dianggap kaca helm motor kecuali visor sangat pekat & tajam.
        if k3_ratio >= 0.25 and best_color != "Bukan Warna K3" and hair_ratio < 0.12:
            is_helm_motor = (dark_visor_ratio >= 0.38 and visor_horizontal_edge >= 28.0)
        else:
            is_helm_motor = (
                (dark_visor_ratio >= 0.28 and visor_horizontal_edge >= 22.0) or
                (dark_visor_ratio >= 0.40 and visor_horizontal_edge >= 16.0)
            )

        # Topi Biasa / Baseball Cap:
        has_visor_shadow = visor_shadow >= 0.15
        is_non_k3_tone = (best_color == "Bukan Warna K3") or (k3_ratio < 0.15)
        is_topi = (
            (has_visor_shadow and (is_khaki or r_minus_b > 22.0 or visor_horizontal_edge > 25.0) and k3_ratio < 0.30) or
            (is_non_k3_tone and has_visor_shadow) or
            (is_non_k3_tone and is_khaki and h_std > 38.0)
        )

        # Helm Proyek Resmi K3:
        # a. Tempurung kubah plastik kaku memiliki warna standar K3 solid (>= 15%)
        # b. Tidak ada rambut asli manusia yang mendominasi ubun-ubun kepala
        #    (toleransi bayangan tepi bawah helm pada ROI kecil/jauh jika k3_ratio >= 0.35 dan tanpa kulit dahi terbuka)
        # c. Area kubah tidak didominasi oleh kulit dahi terbuka (skin_ratio < 0.42)
        # d. Bebas dari ciri aksesoris lain (bukan pori sepeda, bukan kaca visor motor)
        max_allowed_hair = 0.36 if (k3_ratio >= 0.28 and skin_ratio < 0.15) else 0.16
        is_solid_k3 = (
            (k3_ratio >= 0.15) and
            (best_color != "Bukan Warna K3") and
            (hair_ratio < max_allowed_hair) and
            (skin_ratio < 0.42)
        )
        is_real_helmet = is_solid_k3 and (not is_helm_sepeda) and (not is_helm_motor)

        # -------------------------------------------------------------
        # 2. ENSEMBLE DECISION DENGAN PRIORITAS KESELAMATAN K3
        # -------------------------------------------------------------
        if yolo_class == "helmet":
            if (
                hair_ratio >= 0.38 or
                (hair_ratio >= 0.18 and skin_ratio >= 0.22) or
                (hair_ratio >= 0.20 and k3_ratio < 0.32)
            ):
                # Proteksi terhadap false-positive YOLO pada kepala terbuka (rambut ubun-ubun terlihat jelas)
                final_label = LABEL_NO_HELMET
                confidence = 0.90
            elif is_helm_sepeda:
                final_label = LABEL_HELM_SEPEDA
                confidence = max(yolo_conf, 0.85)
            elif is_helm_motor:
                final_label = LABEL_HELM_MOTOR
                confidence = max(yolo_conf, 0.88)
            elif is_real_helmet:
                final_label = LABEL_REAL_HELMET
                confidence = max(yolo_conf, 0.85 + min(0.12, k3_ratio * 0.2))
            elif is_kerudung:
                final_label = LABEL_KERUDUNG
                confidence = max(yolo_conf, 0.88)
            elif is_topi or is_non_k3_tone:
                final_label = LABEL_TOPI
                confidence = max(yolo_conf, 0.82)
            else:
                final_label = LABEL_TOPI
                confidence = 0.78

        elif yolo_class in ["no-helmet", "person-head"]:
            # Verifikasi apakah benar-benar kepala terbuka (rambut/kulit)
            # atau justru helm K3 yang terlewat oleh YOLO karena silau matahari / backlight / jarak jauh outdoor:
            if is_real_helmet and k3_ratio >= 0.16 and hair_ratio < max_allowed_hair and skin_ratio < 0.30:
                # Pemulihan cerdas: Pekerja memakai helm K3 asli namun YOLO terganggu cahaya/jarak outdoor
                final_label = LABEL_REAL_HELMET
                confidence = max(0.82, min(0.95, 0.76 + k3_ratio * 0.20))
            elif hair_ratio >= 0.15 or skin_ratio >= 0.35:
                final_label = LABEL_NO_HELMET
                confidence = max(yolo_conf, 0.90)
            elif is_kerudung:
                final_label = LABEL_KERUDUNG
                confidence = max(yolo_conf, 0.90)
            elif is_topi:
                final_label = LABEL_TOPI
                confidence = max(yolo_conf, 0.80)
            elif is_helm_sepeda:
                final_label = LABEL_HELM_SEPEDA
                confidence = 0.80
            elif is_helm_motor:
                final_label = LABEL_HELM_MOTOR
                confidence = 0.80
            else:
                final_label = LABEL_NO_HELMET
                confidence = max(yolo_conf, 0.85)
        else:
            if hair_ratio >= 0.15 or skin_ratio >= 0.35:
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
            "detected_k3_color": best_color,
            "k3_color_ratio": float(k3_ratio),
            "edge_density": float(edge_density),
            "drapery_ratio": float(drapery_ratio),
            "vent_count": int(vent_count),
            "visor_shadow": float(visor_shadow),
            "dark_visor_ratio": float(dark_visor_ratio),
            "hair_ratio": float(hair_ratio),
            "skin_ratio": float(skin_ratio),
            "lighting_condition": lighting_condition,
            "features": features
        }
