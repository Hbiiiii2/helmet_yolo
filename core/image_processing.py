"""
Modul Pengolahan Citra Digital (Digital Image Processing)
Diadaptasi langsung dari materi perkuliahan Pengolahan Citra (Universitas Bina Sarana Informatika):
- Slide P02: Spatial Filtering (Gaussian Blur, Bilateral Filtering, Konvolusi Kernel & Reduksi Noise Cahaya Matahari)
- Slide P03: Peningkatan Kualitas Citra (Adaptive Gamma Backlight Recovery, Anti-Glare & CLAHE)
- Slide P04: Model Warna (Color Models - Konversi RGB/LAB/HSV & Ekstraksi Warna K3 Tahan Cuaca Outdoor)
- Slide P07: Pendeteksian Tepi (Edge Detection - Sobel & Canny)
- Slide P09: Segmentasi Citra (Thresholding & Analisis Region)
- Slide P11: Morfologi Citra (Erosi, Dilatasi, Opening, Closing)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any, List
from .config import K3_COLOR_RANGES


class ImageProcessingPipeline:
    """
    Pipeline analisis fitur citra berbasis metode pengolahan citra klasik
    dengan kompensasi khusus kondisi Outdoor (Backlight Siluet & Glare/Noise Cahaya Matahari)
    untuk membedakan secara presisi:
    1. Helm Proyek K3 (Plastik Kaku, Warna K3 Murni, Telinga Terbuka, Tanpa Ventilasi Sepeda)
    2. Topi (Lidah Visor Menonjol, Kain Twill/Camo, Bayangan Visor di Bawah)
    3. Helm Motor (Kaca Visor Menutupi Mata/Wajah, Cangkang Tebal Menutupi Telinga & Rahang)
    4. Helm Sepeda (Pori-Pori Ventilasi Aerodinamis Berulang pada Kubah Atas)
    5. Kerudung / Hijab (Kain Berlipat Vertikal Menutupi Leher dan Pundak)
    """

    def __init__(self):
        # CLAHE (Slide P03: Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        self.clahe_strong = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))

        # Kernel Morfologi (Slide P11)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def analyze_lighting_condition(self, bgr_img: np.ndarray) -> Dict[str, Any]:
        """
        Slide P02 & P03: Diagnostik Pencahayaan Outdoor (Deteksi Backlight, Siluet,
        Glare Terik Matahari, dan Noise Sensor/Kompresi).
        """
        if bgr_img is None or bgr_img.size == 0:
            return {
                "mean_l": 128.0,
                "backlight_score": 0.0,
                "glare_score": 0.0,
                "noise_sigma": 0.0,
                "is_backlight": False,
                "is_sunlight_glare": False,
                "is_noisy": False,
                "is_outdoor_challenging": False,
                "condition_label": "Normal"
            }

        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        l = lab[:, :, 0]
        s = hsv[:, :, 1]

        mean_l = float(np.mean(l))
        p10 = float(np.percentile(l, 10))
        p25 = float(np.percentile(l, 25))
        p50 = float(np.percentile(l, 50))
        p85 = float(np.percentile(l, 85))
        p90 = float(np.percentile(l, 90))

        h = l.shape[0]
        top_half_l = l[:max(1, h // 2), :]

        # Proporsi piksel gelap (siluet/bayangan) vs latar langit terik
        dark_ratio = float(np.mean(l < 75))
        bright_bg_ratio = float(np.mean(top_half_l > 200))
        glare_ratio = float(np.mean((l > 210) & (s < 60)))

        # Estimasi noise frekuensi tinggi menggunakan Median Absolute Deviation (MAD) Laplacian (Slide P02)
        lap = cv2.Laplacian(l, cv2.CV_32F, ksize=3)
        noise_sigma = float(np.median(np.abs(lap)) / 0.6745)

        backlight_score = float(
            max(0.0, (115.0 - mean_l) / 115.0) * 0.5 +
            dark_ratio * 0.35 +
            (0.15 if (p90 - p10 > 105 and bright_bg_ratio > 0.10) else 0.0)
        )
        glare_score = float(
            glare_ratio * 0.6 +
            max(0.0, (mean_l - 145.0) / 110.0) * 0.4
        )

        is_backlight = (
            (mean_l < 112.0) or
            (p25 < 68.0 and bright_bg_ratio > 0.08) or
            (dark_ratio > 0.22 and (p90 - p10) > 100.0)
        )
        is_sunlight_glare = (
            (mean_l > 148.0) or
            (glare_ratio > 0.08) or
            (p85 > 205.0)
        )
        is_noisy = noise_sigma > 6.0
        is_outdoor_challenging = is_backlight or is_sunlight_glare or is_noisy

        if is_backlight and is_sunlight_glare:
            condition_label = "Kontras Ekstrem Outdoor (Backlight + Glare)"
        elif is_backlight:
            condition_label = "Backlight / Bayangan Siluet Outdoor"
        elif is_sunlight_glare and is_noisy:
            condition_label = "Noise & Glare Cahaya Matahari Terik"
        elif is_sunlight_glare:
            condition_label = "Glare Cahaya Matahari Terik (Overexposed)"
        elif is_noisy:
            condition_label = "Noise Cahaya / Sensor Outdoor"
        else:
            condition_label = "Pencahayaan Normal / Terkontrol"

        return {
            "mean_l": round(mean_l, 1),
            "p10": round(p10, 1),
            "p50": round(p50, 1),
            "p90": round(p90, 1),
            "dark_ratio": round(dark_ratio, 3),
            "glare_ratio": round(glare_ratio, 3),
            "backlight_score": round(backlight_score, 3),
            "glare_score": round(glare_score, 3),
            "noise_sigma": round(noise_sigma, 2),
            "is_backlight": bool(is_backlight),
            "is_sunlight_glare": bool(is_sunlight_glare),
            "is_noisy": bool(is_noisy),
            "is_outdoor_challenging": bool(is_outdoor_challenging),
            "condition_label": condition_label
        }

    def generate_outdoor_recovery_variants(
        self,
        bgr_img: np.ndarray
    ) -> Tuple[List[Tuple[str, np.ndarray]], Dict[str, Any]]:
        """
        Slide P02, P03 & P04: Menghasilkan varian citra yang telah dinormalisasi
        dari gangguan Backlight (siluet gelap) dan Noise/Glare Cahaya Matahari
        untuk memastikan semua orang/pekerja di foto outdoor dapat teridentifikasi oleh YOLO.
        """
        lighting_info = self.analyze_lighting_condition(bgr_img)
        if bgr_img is None or bgr_img.size == 0:
            return [], lighting_info

        # 1. Edge-Preserving Denoising (Slide P02: Bilateral Filter + Chroma Smoothing)
        # Mereduksi bintik noise matahari & sensor tanpa mengaburkan tepi objek/kepala
        denoised = cv2.bilateralFilter(bgr_img, d=5, sigmaColor=40, sigmaSpace=40)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        a_smooth = cv2.GaussianBlur(a, (3, 3), 0.8)
        b_smooth = cv2.GaussianBlur(b, (3, 3), 0.8)

        h, w = l.shape[:2]
        sigma = max(15, min(h, w) // 10)
        l_blur = cv2.GaussianBlur(l.astype(np.float32), (0, 0), sigmaX=sigma) / 255.0
        l_norm = l.astype(np.float32) / 255.0

        variants: List[Tuple[str, np.ndarray]] = []

        # Varian 1: Spatially-Adaptive Shadow/Backlight Lift + Highlight Compression + CLAHE
        # Mengangkat siluet gelap (gamma < 1.0) sekaligus meredam langit silau (gamma > 1.0)
        gamma_map = np.clip(0.42 + 0.95 * (l_blur ** 1.1), 0.40, 1.65)
        l_lifted = np.power(np.clip(l_norm, 1e-4, 1.0), gamma_map) * 255.0
        l_lifted_u8 = np.clip(l_lifted, 0, 255).astype(np.uint8)
        l_clahe = self.clahe_strong.apply(l_lifted_u8)

        bgr_balanced = cv2.cvtColor(cv2.merge((l_clahe, a_smooth, b_smooth)), cv2.COLOR_LAB2BGR)
        hsv_bal = cv2.cvtColor(bgr_balanced, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_bal[:, :, 1] = np.clip(hsv_bal[:, :, 1] * 1.22, 0, 255)
        bgr_balanced = cv2.cvtColor(hsv_bal.astype(np.uint8), cv2.COLOR_HSV2BGR)
        variants.append(("outdoor_balanced", bgr_balanced))

        # Varian 2: Anti-Glare & Sunlight Noise Suppression (khusus citra silau / overexposed)
        mean_l = lighting_info["mean_l"]
        p10 = lighting_info["p10"]
        p90 = lighting_info["p90"]
        if lighting_info["is_sunlight_glare"] or mean_l > 135.0 or p90 > 200.0:
            low_anchor = max(0.0, p10 * 0.80)
            l_stretch = np.clip(
                (l.astype(np.float32) - low_anchor) * (255.0 / max(40.0, 255.0 - low_anchor)),
                0,
                255
            )
            l_dark = np.power(l_stretch / 255.0, 1.85) * 255.0
            l_dark_u8 = self.clahe_strong.apply(np.clip(l_dark, 0, 255).astype(np.uint8))
            bgr_antiglare = cv2.cvtColor(cv2.merge((l_dark_u8, a_smooth, b_smooth)), cv2.COLOR_LAB2BGR)
            hsv_ag = cv2.cvtColor(bgr_antiglare, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_ag[:, :, 1] = np.clip(hsv_ag[:, :, 1] * 1.35, 0, 255)
            bgr_antiglare = cv2.cvtColor(hsv_ag.astype(np.uint8), cv2.COLOR_HSV2BGR)
            variants.append(("outdoor_antiglare", bgr_antiglare))

        # Varian 3: Deep Backlight Shadow Recovery (khusus siluet sangat gelap membelakangi matahari)
        if lighting_info["is_backlight"] and (mean_l < 105.0 or lighting_info["dark_ratio"] > 0.28):
            l_deep = np.power(np.clip(l_norm, 1e-4, 1.0), 0.42) * 255.0
            l_deep_u8 = self.clahe_strong.apply(np.clip(l_deep, 0, 255).astype(np.uint8))
            bgr_backlight = cv2.cvtColor(cv2.merge((l_deep_u8, a_smooth, b_smooth)), cv2.COLOR_LAB2BGR)
            hsv_bl = cv2.cvtColor(bgr_backlight, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_bl[:, :, 1] = np.clip(hsv_bl[:, :, 1] * 1.28, 0, 255)
            bgr_backlight = cv2.cvtColor(hsv_bl.astype(np.uint8), cv2.COLOR_HSV2BGR)
            variants.append(("outdoor_backlight", bgr_backlight))

        return variants, lighting_info

    def enhance_image(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Slide P02 & P03: Image Enhancement Adaptif untuk ROI Kepala.
        Menggabungkan Bilateral Denoising, Kompensasi Gamma Backlight/Glare, dan CLAHE pada kanal LAB.
        """
        if bgr_img is None or bgr_img.size == 0:
            return bgr_img

        # Reduksi noise frekuensi tinggi (noise matahari/sensor) dengan tetap menjaga ketajaman tepi
        denoised = cv2.bilateralFilter(bgr_img, d=5, sigmaColor=35, sigmaSpace=35)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        a = cv2.GaussianBlur(a, (3, 3), 0.7)
        b = cv2.GaussianBlur(b, (3, 3), 0.7)

        mean_l = float(np.mean(l))
        l_norm = l.astype(np.float32) / 255.0

        # Kompensasi Gamma Adaptif jika ROI terkena Backlight gelap atau Glare terik
        if mean_l < 105.0:
            # Angkat bayangan backlight pada kepala/helm
            gamma = max(0.48, min(0.82, mean_l / 135.0))
            l_corr = np.clip(np.power(np.clip(l_norm, 1e-4, 1.0), gamma) * 255.0, 0, 255).astype(np.uint8)
            l_enhanced = self.clahe_strong.apply(l_corr)
        elif mean_l > 168.0:
            # Redam glare cahaya matahari berlebih pada tempurung helm
            gamma = min(1.60, max(1.18, mean_l / 130.0))
            l_corr = np.clip(np.power(np.clip(l_norm, 1e-4, 1.0), gamma) * 255.0, 0, 255).astype(np.uint8)
            l_enhanced = self.clahe.apply(l_corr)
        else:
            l_enhanced = self.clahe.apply(l)

        enhanced_lab = cv2.merge((l_enhanced, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Restorasi saturasi warna yang pudar akibat backlight atau terik matahari
        # Hanya untuk piksel yang sudah kromatik (S >= 105) agar kulit/khaki tidak ikut naik
        if mean_l < 100.0 or mean_l > 165.0:
            hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            s_chan = hsv[:, :, 1]
            s_chan[s_chan >= 105] = np.clip(s_chan[s_chan >= 105] * 1.18, 0, 255)
            hsv[:, :, 1] = s_chan
            enhanced_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return enhanced_bgr

    def _compute_single_k3_mask(
        self,
        crown_bgr: np.ndarray,
        crown_hsv: np.ndarray,
        color_name: str,
        range_data: Any
    ) -> np.ndarray:
        """Helper ekstraksi mask satu warna K3 pada area kubah."""
        crown_h, w = crown_hsv.shape[:2]
        B, G, R = cv2.split(crown_bgr.astype(float))
        S_chan = crown_hsv[:, :, 1].astype(float)

        if isinstance(range_data, dict) and range_data.get("is_rgb_balanced"):
            # Putih K3: R, G, B tinggi, seimbang, dan saturasi rendah (mengecualikan khaki/kulit)
            min_b = range_data.get("min_brightness", 165)
            max_d = range_data.get("max_diff", 30)
            max_s = range_data.get("max_saturation", 42)
            mask_bool = (
                (R >= min_b) & (G >= min_b) & (B >= min_b) &
                (np.abs(R - G) <= max_d) &
                (np.abs(G - B) <= max_d) &
                (np.abs(R - B) <= max_d) &
                (S_chan <= max_s)
            )
            return mask_bool.astype(np.uint8) * 255
        elif isinstance(range_data, list):
            # Merah K3
            mask = np.zeros((crown_h, w), dtype=np.uint8)
            for r in range_data:
                lower = np.array([r["lower"][0], max(110, r["lower"][1]), r["lower"][2]])
                m = cv2.inRange(crown_hsv, lower, np.array(r["upper"]))
                mask = cv2.bitwise_or(mask, m)
            return mask
        else:
            s_min = range_data["lower"][1]
            v_min = range_data["lower"][2]
            if "Oranye" in color_name:
                s_min = max(130, s_min)
                v_min = max(105, v_min)
            elif "Kuning" in color_name:
                s_min = max(82, s_min)
                v_min = max(80, v_min)
            lower = np.array([range_data["lower"][0], s_min, v_min])
            upper = np.array(range_data["upper"])
            mask = cv2.inRange(crown_hsv, lower, upper)
            if "Kuning" in color_name:
                # Kecualikan warna kulit/dahi hangat (H <= 19 dengan saturasi moderat < 120 atau R - G > 42)
                H_chan = crown_hsv[:, :, 0]
                not_warm_skin = ((H_chan >= 20) | (S_chan >= 125)) & ((R - G) <= 42)
                mask = mask & (not_warm_skin.astype(np.uint8) * 255)
            return mask

    def extract_k3_color_mask(
        self,
        raw_bgr: np.ndarray,
        hsv_enhanced: np.ndarray,
        enhanced_bgr: np.ndarray = None
    ) -> Tuple[np.ndarray, str, float, Dict[str, float]]:
        """
        Slide P04: Model Warna HSV & RGB terkalibrasi ketat & tahan cuaca outdoor.
        Mengevaluasi kubah atas (crown/top 58%) menggunakan fusi citra asli & citra hasil
        kompensasi Backlight/Glare, serta mengabaikan titik pantulan silau matahari (specular glare).
        """
        if enhanced_bgr is None:
            enhanced_bgr = raw_bgr

        h, w = hsv_enhanced.shape[:2]
        crown_h = max(4, int(h * 0.58))
        # Fokuskan evaluasi rasio pada kubah tengah (x: 10% s.d. 90%) agar latar belakang di pojok tidak mendominasi
        x_pad = max(1, int(w * 0.10)) if w >= 12 else 0

        raw_hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)

        crown_raw_bgr = raw_bgr[:crown_h, :]
        crown_raw_hsv = raw_hsv[:crown_h, :]
        crown_enh_bgr = enhanced_bgr[:crown_h, :]
        crown_enh_hsv = hsv_enhanced[:crown_h, :]

        # Deteksi piksel rambut gelap asli di pusat ubun-ubun agar tidak tertukar dengan warna helm saat di-enhance
        raw_v = crown_raw_hsv[:, :, 2]
        raw_s = crown_raw_hsv[:, :, 1]
        enh_s = crown_enh_hsv[:, :, 1]
        enh_v = crown_enh_hsv[:, :, 2]
        definite_dark_hair = (raw_v < 78) | ((raw_v < 92) & (raw_s < 80) & (enh_v < 118))
    
        # Titik pantulan cahaya matahari langsung (Specular Glare: V sangat tinggi, S sangat rendah)
        specular_glare = (crown_raw_hsv[:, :, 2] > 238) & (crown_raw_hsv[:, :, 1] < 32)
        center_slice = slice(x_pad, w - x_pad) if (w - 2 * x_pad) >= 4 else slice(0, w)
        center_total = max(1, crown_h * (center_slice.stop - center_slice.start))
        glare_pixels = int(np.count_nonzero(specular_glare[:, center_slice]))
        # Diskon area silau matahari maksimal 30% dari penyebut luas kubah
        effective_crown_pixels = max(1, center_total - min(int(center_total * 0.30), glare_pixels))

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        color_scores = {}
        best_color_name = "Bukan Warna K3"
        max_ratio = 0.0

        for color_name, range_data in K3_COLOR_RANGES.items():
            mask_raw = self._compute_single_k3_mask(crown_raw_bgr, crown_raw_hsv, color_name, range_data)
            mask_enh = self._compute_single_k3_mask(crown_enh_bgr, crown_enh_hsv, color_name, range_data)
            mask = cv2.bitwise_or(mask_raw, mask_enh)

            # Kecualikan piksel rambut hitam pekat
            mask[definite_dark_hair] = 0

            center_mask = mask[:, center_slice]
            pixel_count = cv2.countNonZero(center_mask)
            ratio = min(1.0, pixel_count / effective_crown_pixels)
            color_scores[color_name] = ratio

            if ratio > max_ratio and ratio > 0.13:
                max_ratio = ratio
                best_color_name = color_name

            combined_mask[:crown_h, :] = cv2.bitwise_or(combined_mask[:crown_h, :], mask)

        # Slide P11: Morfologi Opening (bersihkan noise) & Closing (tutup celah pantulan matahari)
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_open)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.kernel_close)

        overall_ratio = min(1.0, cv2.countNonZero(clean_mask[:crown_h, center_slice]) / effective_crown_pixels)
        if overall_ratio < 0.12:
            best_color_name = "Bukan Warna K3"

        return clean_mask, best_color_name, overall_ratio, color_scores

    def detect_aerodynamic_vents(self, gray_roi: np.ndarray, hsv_roi: np.ndarray) -> Tuple[int, float]:
        """
        Slide P07 & P09: Deteksi Pori-Pori Ventilasi Aerodinamis (Ciri Khas Helm Sepeda).
        Helm sepeda memiliki 3 s.d. 15 lubang ventilasi memanjang pada kubah atas
        di mana bagian dalam lubang tampak gelap (rambut/bayangan interior).
        Dilengkapi filter anti-noise matahari agar bintik cahaya tidak salah dihitung sebagai ventilasi.
        """
        h, w = gray_roi.shape[:2]
        upper_h = max(5, int(h * 0.55))
        upper_dome = gray_roi[:upper_h, :]
        upper_hsv = hsv_roi[:upper_h, :]
        if upper_dome.size < 50:
            return 0, 0.0

        blurred = cv2.GaussianBlur(upper_dome, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 45, 130)
        edge_density = cv2.countNonZero(edges) / max(1, upper_dome.size)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        vent_count = 0
        min_vent_area = max(22, int(upper_dome.size * 0.012))
        max_vent_area = int(upper_dome.size * 0.14)
        dome_v_mean = float(np.mean(upper_hsv[:, :, 2]))

        for c in contours:
            area = cv2.contourArea(c)
            if min_vent_area < area < max_vent_area:
                mask = np.zeros(upper_dome.shape, dtype=np.uint8)
                cv2.drawContours(mask, [c], -1, 255, -1)
                vent_v = cv2.mean(upper_hsv[:, :, 2], mask=mask)[0]
                # Pori ventilasi sepeda asli adalah celah berongga yang gelap di dalamnya
                if vent_v < (dome_v_mean * 0.62) and vent_v < 80:
                    vent_count += 1

        return vent_count, float(edge_density)

    def detect_visor_and_cap_features(self, bgr_roi: np.ndarray, hsv_roi: np.ndarray) -> Dict[str, float]:
        """
        Slide P07: Deteksi Lidah Topi (Visor), Bayangan Bawah Lidah, dan Tekstur Topi Kasual.
        """
        h, w = bgr_roi.shape[:2]

        # 1. Kontras Bayangan Visor (Kecerahan atas vs bawah topi)
        v_upper = float(np.mean(hsv_roi[:max(2, int(h * 0.40)), :, 2]))
        v_lower = float(np.mean(hsv_roi[int(h * 0.60):, :, 2]))
        visor_shadow = max(0.0, (v_upper - v_lower) / max(1.0, v_upper + v_lower))

        # 2. Warmth / Khaki Score: Selisih R - B
        B, _, R = cv2.split(bgr_roi.astype(float))
        r_minus_b = float(np.mean(R - B))
        is_khaki_tone = r_minus_b > 25.0

        # 3. Variansi Hue (Kain corak / camo)
        h_std = float(np.std(hsv_roi[:int(h * 0.60), :, 0]))

        # 4. Deteksi Lengkungan Visor di area bawah (bottom 40%)
        lower_gray = cv2.cvtColor(bgr_roi[int(h * 0.50):, :], cv2.COLOR_BGR2GRAY)
        lower_edges = cv2.Canny(lower_gray, 35, 100)
        lower_edge_ratio = cv2.countNonZero(lower_edges) / max(1, lower_gray.size)

        return {
            "visor_shadow": visor_shadow,
            "r_minus_b": r_minus_b,
            "is_khaki_tone": is_khaki_tone,
            "h_std": h_std,
            "lower_edge_ratio": float(lower_edge_ratio)
        }

    def detect_motorcycle_helmet_features(
        self,
        enhanced_bgr: np.ndarray,
        hsv_roi: np.ndarray,
        raw_bgr: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Slide P04 & P09: Deteksi Ciri Khas Helm Motor yang Tahan terhadap Bayangan Backlight Outdoor.
        Penting: Pada foto outdoor backlight, wajah pekerja di bawah pinggiran helm K3 sering gelap.
        Oleh karena itu, deteksi kaca visor helm motor harus mengevaluasi citra yang sudah dinormalisasi
        pencahayaannya serta memeriksa apakah bagian tengah benar-benar kaca visor mengilap (bukan bayangan wajah).
        """
        h, w = enhanced_bgr.shape[:2]
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

        mid_y1 = int(h * 0.42)
        mid_y2 = int(h * 0.78)
        mid_x1 = int(w * 0.18)
        mid_x2 = int(w * 0.82)

        mid_zone = gray[mid_y1:mid_y2, mid_x1:mid_x2]
        mid_hsv = hsv_roi[mid_y1:mid_y2, mid_x1:mid_x2]
        dark_visor_ratio = 0.0
        visor_horizontal_edge = 0.0

        if mid_zone.size > 20:
            # Cek apakah ROI secara keseluruhan memang sedang mengalami backlight gelap
            raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) if raw_bgr is not None else gray
            ambient_median = float(np.median(raw_gray))
            dark_thresh = 35 if ambient_median < 90 else 44

            # Piksel kaca visor hitam: gelap DAN bersaturasi rendah (bukan bayangan kulit hangat)
            is_skin_tone = (mid_hsv[:, :, 0] <= 25) & (mid_hsv[:, :, 1] >= 40) & (mid_hsv[:, :, 2] >= 50)
            dark_visor_pixels = (mid_zone < dark_thresh) & (~is_skin_tone)
            dark_visor_ratio = float(np.count_nonzero(dark_visor_pixels) / mid_zone.size)

            sobel_y = cv2.Sobel(mid_zone, cv2.CV_32F, 0, 1, ksize=3)
            visor_horizontal_edge = float(np.mean(np.abs(sobel_y)))

        return {
            "dark_visor_ratio": float(dark_visor_ratio),
            "visor_horizontal_edge": float(visor_horizontal_edge),
            "aspect_ratio": float(h / max(1, w))
        }

    def analyze_edges_and_texture(self, gray_roi: np.ndarray) -> Dict[str, float]:
        """
        Slide P02 & P07: Tepi Sobel & Canny untuk mengukur tekstur interior vs lipatan kain.
        """
        if gray_roi is None or gray_roi.size == 0:
            return {
                "edge_density": 0.0,
                "drapery_ratio": 1.0,
                "texture_var": 0.0,
                "canny_img": np.zeros((10, 10), dtype=np.uint8)
            }

        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 45, 125)

        h, w = gray_roi.shape[:2]
        margin_y, margin_x = max(2, int(h * 0.12)), max(2, int(w * 0.12))
        inner_edges = edges[margin_y:h - margin_y, margin_x:w - margin_x]

        inner_area = max(1, (h - 2 * margin_y) * (w - 2 * margin_x))
        edge_density = cv2.countNonZero(inner_edges) / inner_area

        sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

        mag_x = np.mean(np.abs(sobel_x[margin_y:h - margin_y, margin_x:w - margin_x]))
        mag_y = np.mean(np.abs(sobel_y[margin_y:h - margin_y, margin_x:w - margin_x]))

        drapery_ratio = float(mag_y / (mag_x + 1e-5))
        texture_var = float(np.var(inner_edges))

        return {
            "edge_density": float(edge_density),
            "drapery_ratio": drapery_ratio,
            "texture_var": texture_var,
            "canny_img": edges
        }

    def analyze_headwear_morphology(self, roi_bgr: np.ndarray, person_crop: np.ndarray = None) -> Dict[str, Any]:
        """
        Slide P09 & P11: Analisis morfologi dan ekstensi leher.
        """
        h, w = roi_bgr.shape[:2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morphed = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0.8
        symmetry_score = 0.5

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area / hull_area)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                symmetry_score = 1.0 - min(1.0, abs(cx - (w / 2.0)) / (w / 4.0))

        neck_coverage_score = 0.0
        if person_crop is not None and person_crop.size > 0:
            ph, _ = person_crop.shape[:2]
            neck_y1 = int(ph * 0.25)
            neck_y2 = int(ph * 0.50)
            neck_zone = person_crop[neck_y1:neck_y2, :]
            if neck_zone.size > 0:
                neck_gray = cv2.cvtColor(neck_zone, cv2.COLOR_BGR2GRAY)
                neck_edges = cv2.Canny(neck_gray, 50, 150)
                neck_coverage_score = float(
                    cv2.countNonZero(neck_edges) / max(1, neck_zone.shape[0] * neck_zone.shape[1])
                )

        return {
            "aspect_ratio": float(h / max(1, w)),
            "solidity": solidity,
            "symmetry_score": symmetry_score,
            "neck_coverage_score": neck_coverage_score,
            "morphed_mask": morphed
        }

    def detect_cranial_hair(
        self,
        raw_bgr: np.ndarray,
        hsv_enhanced: np.ndarray,
        k3_color_mask: np.ndarray = None,
        is_tiny_roi: bool = False
    ) -> Tuple[float, float]:
        """
        Slide P04 & P09: Deteksi Kehadiran Rambut Asli / Kulit Kepala pada Kubah Atas
        yang Tahan terhadap Bayangan Backlight Outdoor & Jarak Jauh.
        - Mengevaluasi puncak ubun-ubun (y: 4%-38%) agar bayangan wajah/masker gelap di bawah
          helm pada foto outdoor tidak salah terhitung sebagai rambut hitam.
        - Mengecualikan seluruh piksel yang sudah tervalidasi sebagai kubah warna K3 (termasuk Putih K3).
        """
        h, w = hsv_enhanced.shape[:2]
        cy1, cy2 = max(1, int(0.06 * h)), max(4, int(0.42 * h))
        cx1, cx2 = max(1, int(0.24 * w)), min(w, max(3, int(0.76 * w)))

        raw_hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
        c_raw_hsv = raw_hsv[cy1:cy2, cx1:cx2]
        c_enh_hsv = hsv_enhanced[cy1:cy2, cx1:cx2]
        if c_enh_hsv.size == 0:
            return 0.0, 0.0

        raw_v = c_raw_hsv[:, :, 2]
        raw_s = c_raw_hsv[:, :, 1]
        enh_v = c_enh_hsv[:, :, 2]
        enh_s = c_enh_hsv[:, :, 1]

        # Rambut gelap asli tetap gelap meskipun sudah dinormalisasi backlight
        dark_hair = (raw_v < 78) | ((raw_v < 92) & (raw_s < 80) & (enh_v < 118))
        if k3_color_mask is not None and k3_color_mask.shape[:2] == (h, w):
            c_k3 = k3_color_mask[cy1:cy2, cx1:cx2] > 0
            # Semua piksel yang sah masuk masker warna K3 (termasuk Putih K3 & warna jenuh) bukan rambut hitam
            dark_hair = dark_hair & (~c_k3)
            # Khusus ROI kepala jarak jauh (tiny ROI) yang kubah tengahnya (y: 26%-62%) didominasi K3 solid,
            # abaikan piksel gelap latar belakang di tepi atas lengkungan helm (y < 28%)
            if is_tiny_roi:
                mid_y1, mid_y2 = max(1, int(0.26 * h)), max(4, int(0.62 * h))
                mid_k3_ratio = float(np.mean(k3_color_mask[mid_y1:mid_y2, cx1:cx2] > 0))
                if mid_k3_ratio >= 0.35:
                    top_border_h = max(1, int(0.28 * h) - cy1)
                    dark_hair[:top_border_h, :] = False

        # Cek polaritas vertikal: Jika kubah atas (0-38%) jauh lebih terang daripada area wajah bawah (42-80%),
        # maka objek di atas adalah kubah helm terang di atas wajah yang terkena bayangan backlight
        lower_face_v = float(np.mean(raw_hsv[int(0.42 * h):max(int(0.42 * h) + 2, int(0.82 * h)), cx1:cx2, 2]))
        upper_dome_v = float(np.mean(raw_v))
        if upper_dome_v > (lower_face_v + 16.0) and upper_dome_v >= 115.0:
            dark_hair = dark_hair & (raw_v < 50)

        hair_ratio = float(np.count_nonzero(dark_hair) / dark_hair.size)

        # Kulit dahi/kepala: H <= 18, S moderat [35, 115], dan bukan kubah Kuning/Oranye K3 jenuh
        h_chan = c_enh_hsv[:, :, 0]
        skin = (
            (h_chan <= 18) &
            (enh_s >= 35) &
            (enh_s <= 115) &
            (raw_s <= 115) &
            (enh_v >= 75) &
            (enh_v <= 240)
        )
        skin_ratio = float(np.count_nonzero(skin) / skin.size)

        return hair_ratio, skin_ratio

    def detect_hijab_and_neck_continuity(
        self,
        head_roi: np.ndarray,
        person_crop: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Slide P04, P07 & P09: Analisis Kontinuitas Kain Kerudung / Hijab dari Kepala ke Leher.
        """
        if head_roi is None or head_roi.size == 0 or person_crop is None or person_crop.size == 0:
            return {
                "is_hijab": False,
                "neck_skin_ratio": 1.0,
                "color_continuity": False,
                "neck_drapery": 1.0
            }

        ph, pw = person_crop.shape[:2]
        hh, hw = head_roi.shape[:2]

        crown = head_roi[:max(4, int(hh * 0.60)), :]
        crown_hsv = cv2.cvtColor(crown, cv2.COLOR_BGR2HSV)
        head_mean = np.mean(crown_hsv, axis=(0, 1))

        neck_y1 = int(0.48 * ph)
        neck_y2 = int(0.72 * ph)
        neck_x1 = int(0.20 * pw)
        neck_x2 = int(0.80 * pw)
        neck = person_crop[neck_y1:neck_y2, neck_x1:neck_x2]

        if neck.size == 0:
            return {
                "is_hijab": False,
                "neck_skin_ratio": 1.0,
                "color_continuity": False,
                "neck_drapery": 1.0
            }

        neck_hsv = cv2.cvtColor(neck, cv2.COLOR_BGR2HSV)
        neck_mean = np.mean(neck_hsv, axis=(0, 1))

        skin = (
            (neck_hsv[:, :, 0] <= 25) &
            (neck_hsv[:, :, 1] >= 35) &
            (neck_hsv[:, :, 1] <= 180) &
            (neck_hsv[:, :, 2] >= 75) &
            (neck_hsv[:, :, 2] <= 245)
        )
        neck_skin_ratio = float(np.count_nonzero(skin) / max(1, neck_hsv.size))

        neck_gray = cv2.cvtColor(neck, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(neck_gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(neck_gray, cv2.CV_32F, 1, 0, ksize=3)
        neck_drapery = float(np.mean(np.abs(sobel_y)) / (np.mean(np.abs(sobel_x)) + 1e-5))

        h_diff = float(min(abs(head_mean[0] - neck_mean[0]), 180 - abs(head_mean[0] - neck_mean[0])))
        s_diff = float(abs(head_mean[1] - neck_mean[1]))
        v_diff = float(abs(head_mean[2] - neck_mean[2]))

        color_continuity = (h_diff < 20.0) and (s_diff < 45.0) and (v_diff < 50.0)
        is_hijab = (color_continuity and neck_skin_ratio < 0.25) or (neck_drapery > 1.25 and neck_skin_ratio < 0.15)

        return {
            "is_hijab": is_hijab,
            "neck_skin_ratio": neck_skin_ratio,
            "color_continuity": color_continuity,
            "neck_drapery": neck_drapery,
            "h_diff": h_diff,
            "s_diff": s_diff
        }

    def inspect_patch(self, roi_bgr: np.ndarray, person_crop: np.ndarray = None) -> Dict[str, Any]:
        """
        Menjalankan seluruh rangkaian analisis citra komprehensif dengan normalisasi Outdoor
        serta Super-Resolution Upscaling otomatis untuk kepala pekerja jarak jauh (small ROI).
        """
        if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.shape[0] < 5 or roi_bgr.shape[1] < 5:
            return {
                "k3_color_ratio": 0.0,
                "best_color_name": "Tidak Terdeteksi",
                "color_scores": {},
                "edge_density": 0.0,
                "drapery_ratio": 1.0,
                "texture_var": 0.0,
                "solidity": 0.8,
                "symmetry_score": 0.5,
                "neck_coverage": 0.0,
                "vent_count": 0,
                "dome_edge_density": 0.0,
                "visor_shadow": 0.0,
                "is_khaki_tone": False,
                "h_std": 0.0,
                "dark_visor_ratio": 0.0,
                "hair_ratio": 0.0,
                "skin_ratio": 0.0,
                "is_hijab": False,
                "color_continuity": False,
                "neck_skin_ratio": 1.0,
                "lighting_condition": "Normal",
                "enhanced_roi": roi_bgr,
                "color_mask": np.zeros((10, 10), dtype=np.uint8),
                "canny_img": np.zeros((10, 10), dtype=np.uint8)
            }

        # Jika ROI kepala berukuran kecil (pekerja jarak jauh di foto outdoor),
        # lakukan Bicubic Upscaling ke minimal 48px agar filter CLAHE, HSV, dan Canny akurat
        rh, rw = roi_bgr.shape[:2]
        if rh < 44 or rw < 44:
            scale_f = max(48.0 / float(rh), 48.0 / float(rw))
            proc_roi = cv2.resize(
                roi_bgr,
                (max(48, int(rw * scale_f)), max(48, int(rh * scale_f))),
                interpolation=cv2.INTER_CUBIC
            )
        else:
            proc_roi = roi_bgr

        # 0. Diagnostik Pencahayaan Outdoor pada ROI
        lighting_info = self.analyze_lighting_condition(proc_roi)

        # 1. Peningkatan Kualitas Citra Adaptif Outdoor (Slide P02 & P03)
        enhanced_roi = self.enhance_image(proc_roi)
        hsv_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2GRAY)

        # 2. Segmentasi Model Warna K3 pada Kubah Atas (Slide P04)
        color_mask, best_color, k3_ratio, color_scores = self.extract_k3_color_mask(
            raw_bgr=proc_roi,
            hsv_enhanced=hsv_roi,
            enhanced_bgr=enhanced_roi
        )

        # 3. Deteksi Kehadiran Rambut Asli pada Kubah Tengah (Slide P04 & P09)
        hair_ratio, skin_ratio = self.detect_cranial_hair(
            raw_bgr=proc_roi,
            hsv_enhanced=hsv_roi,
            k3_color_mask=color_mask,
            is_tiny_roi=(rh <= 24 or rw <= 24)
        )

        # 4. Deteksi Kontinuitas Hijab dari Kepala ke Leher (Slide P04 & P09)
        hijab_data = self.detect_hijab_and_neck_continuity(enhanced_roi, person_crop)

        # 5. Analisis Ventilasi Sepeda Berpori Gelap (Slide P07 & P09)
        vent_count, dome_edge_density = self.detect_aerodynamic_vents(gray_roi, hsv_roi)

        # 6. Analisis Ciri Topi Kasual (Slide P07: Lidah Visor & Bayangan)
        cap_feats = self.detect_visor_and_cap_features(enhanced_roi, hsv_roi)

        # 7. Analisis Ciri Helm Motor Tahan Backlight (Slide P04 & P09)
        motor_feats = self.detect_motorcycle_helmet_features(enhanced_roi, hsv_roi, raw_bgr=proc_roi)

        # 8. Analisis Tepi Interior & Drapery Kain (Slide P02 & P07)
        edge_metrics = self.analyze_edges_and_texture(gray_roi)

        # 9. Morfologi Citra (Slide P11)
        morph_metrics = self.analyze_headwear_morphology(enhanced_roi, person_crop)

        return {
            "k3_color_ratio": k3_ratio,
            "best_color_name": best_color,
            "color_scores": color_scores,
            "edge_density": edge_metrics["edge_density"],
            "drapery_ratio": edge_metrics["drapery_ratio"],
            "texture_var": edge_metrics["texture_var"],
            "canny_img": edge_metrics["canny_img"],
            "color_mask": color_mask,
            "enhanced_roi": enhanced_roi,
            "aspect_ratio": morph_metrics["aspect_ratio"],
            "solidity": morph_metrics["solidity"],
            "symmetry_score": morph_metrics["symmetry_score"],
            "neck_coverage": morph_metrics["neck_coverage_score"],
            "vent_count": vent_count,
            "dome_edge_density": dome_edge_density,
            "visor_shadow": cap_feats["visor_shadow"],
            "is_khaki_tone": cap_feats["is_khaki_tone"],
            "r_minus_b": cap_feats["r_minus_b"],
            "h_std": cap_feats["h_std"],
            "dark_visor_ratio": motor_feats["dark_visor_ratio"],
            "visor_horizontal_edge": motor_feats["visor_horizontal_edge"],
            "hair_ratio": hair_ratio,
            "skin_ratio": skin_ratio,
            "is_hijab": hijab_data["is_hijab"],
            "color_continuity": hijab_data["color_continuity"],
            "neck_skin_ratio": hijab_data["neck_skin_ratio"],
            "lighting_condition": lighting_info["condition_label"],
            "lighting_info": lighting_info
        }
