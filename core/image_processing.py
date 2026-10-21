"""
Modul Pengolahan Citra Digital (Digital Image Processing)
Diadaptasi langsung dari materi perkuliahan Pengolahan Citra (Universitas Bina Sarana Informatika):
- Slide P02: Spatial Filtering (Gaussian Blur, Konvolusi Kernel)
- Slide P03: Peningkatan Kualitas Citra (Image Enhancement - CLAHE)
- Slide P04: Model Warna (Color Models - Konversi RGB ke HSV & Ekstraksi Warna K3 yang Ketat)
- Slide P07: Pendeteksian Tepi (Edge Detection - Sobel & Canny)
- Slide P09: Segmentasi Citra (Thresholding & Analisis Region)
- Slide P11: Morfologi Citra (Erosi, Dilatasi, Opening, Closing)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any
from .config import K3_COLOR_RANGES


class ImageProcessingPipeline:
    """
    Pipeline analisis fitur citra berbasis metode pengolahan citra klasik
    untuk membedakan secara presisi:
    1. Helm Proyek K3 (Plastik Kaku, Warna K3 Murni, Telinga Terbuka, Tanpa Ventilasi Sepeda)
    2. Topi (Lidah Visor Menonjol, Kain Twill/Camo, Bayangan Visor di Bawah)
    3. Helm Motor (Kaca Visor Menutupi Mata/Wajah, Cangkang Tebal Menutupi Telinga & Rahang)
    4. Helm Sepeda (Pori-Pori Ventilasi Aerodinamis Berulang pada Kubah Atas)
    5. Kerudung / Hijab (Kain Berlipat Vertikal Menutupi Leher dan Pundak)
    """

    def __init__(self):
        # CLAHE (Slide P03: Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Kernel Morfologi (Slide P11)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def enhance_image(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Slide P03: Image Enhancement menggunakan CLAHE pada kanal Luminance LAB.
        """
        if bgr_img is None or bgr_img.size == 0:
            return bgr_img
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def extract_k3_color_mask(self, bgr_img: np.ndarray, hsv_img: np.ndarray) -> Tuple[np.ndarray, str, float, Dict[str, float]]:
        """
        Slide P04: Model Warna HSV & RGB terkalibrasi ketat.
        Penting: Evaluasi warna helm diutamakan pada kubah atas (crown/top 60%)
        agar warna kulit wajah/dahi manusia (yang sering kemerahan/kuning) tidak salah dideteksi sebagai helm!
        """
        h, w = hsv_img.shape[:2]
        crown_h = max(4, int(h * 0.60))
        crown_bgr = bgr_img[:crown_h, :]
        crown_hsv = hsv_img[:crown_h, :]
        crown_pixels = max(1, crown_h * w)

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        color_scores = {}
        best_color_name = "Bukan Warna K3"
        max_ratio = 0.0

        B, G, R = cv2.split(crown_bgr.astype(float))

        for color_name, range_data in K3_COLOR_RANGES.items():
            if isinstance(range_data, dict) and range_data.get("is_rgb_balanced"):
                # Putih K3: R, G, B semua tinggi (>185) DAN seimbang (|R - B| <= 22)
                # Mengecualikan khaki/cokelat/kulit di mana R >> B
                min_b = range_data.get("min_brightness", 185)
                max_d = range_data.get("max_diff", 22)
                mask_bool = (
                    (R >= min_b) & (G >= min_b) & (B >= min_b) &
                    (np.abs(R - G) <= max_d) &
                    (np.abs(G - B) <= max_d) &
                    (np.abs(R - B) <= max_d)
                )
                mask = mask_bool.astype(np.uint8) * 255
            elif isinstance(range_data, list):
                # Merah K3 (S >= 120 agar tidak mendeteksi warna kulit dahi)
                mask = np.zeros((crown_h, w), dtype=np.uint8)
                for r in range_data:
                    lower = np.array([r["lower"][0], max(120, r["lower"][1]), r["lower"][2]])
                    m = cv2.inRange(crown_hsv, lower, np.array(r["upper"]))
                    mask = cv2.bitwise_or(mask, m)
            else:
                lower = np.array(range_data["lower"])
                upper = np.array(range_data["upper"])
                mask = cv2.inRange(crown_hsv, lower, upper)

            # Hitung proporsi warna pada kubah atas
            pixel_count = cv2.countNonZero(mask)
            ratio = pixel_count / crown_pixels
            color_scores[color_name] = ratio

            if ratio > max_ratio and ratio > 0.14:
                max_ratio = ratio
                best_color_name = color_name

            # Tempelkan ke combined mask (hanya di zona kubah)
            combined_mask[:crown_h, :] = cv2.bitwise_or(combined_mask[:crown_h, :], mask)

        # Slide P11: Morfologi Opening (bersihkan noise) & Closing (tutup celah)
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_open)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.kernel_close)

        overall_ratio = cv2.countNonZero(clean_mask[:crown_h, :]) / crown_pixels
        if overall_ratio < 0.12:
            best_color_name = "Bukan Warna K3"

        return clean_mask, best_color_name, overall_ratio, color_scores

    def detect_aerodynamic_vents(self, gray_roi: np.ndarray, hsv_roi: np.ndarray) -> Tuple[int, float]:
        """
        Slide P07 & P09: Deteksi Pori-Pori Ventilasi Aerodinamis (Ciri Khas Helm Sepeda).
        Helm sepeda memiliki 3 s.d. 15 lubang ventilasi memanjang pada kubah atas
        di mana bagian dalam lubang tampak gelap (rambut/bayangan interior).
        """
        h, w = gray_roi.shape[:2]
        upper_h = max(5, int(h * 0.55))
        upper_dome = gray_roi[:upper_h, :]
        upper_hsv = hsv_roi[:upper_h, :]
        if upper_dome.size < 50:
            return 0, 0.0

        blurred = cv2.GaussianBlur(upper_dome, (3, 3), 0.8)
        edges = cv2.Canny(blurred, 40, 120)
        edge_density = cv2.countNonZero(edges) / max(1, upper_dome.size)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        vent_count = 0
        min_vent_area = max(20, int(upper_dome.size * 0.01))
        max_vent_area = int(upper_dome.size * 0.15)
        dome_v_mean = float(np.mean(upper_hsv[:, :, 2]))

        for c in contours:
            area = cv2.contourArea(c)
            if min_vent_area < area < max_vent_area:
                mask = np.zeros(upper_dome.shape, dtype=np.uint8)
                cv2.drawContours(mask, [c], -1, 255, -1)
                vent_v = cv2.mean(upper_hsv[:, :, 2], mask=mask)[0]
                # Pori ventilasi sepeda asli adalah celah berongga yang gelap di dalamnya
                if vent_v < (dome_v_mean * 0.65) and vent_v < 85:
                    vent_count += 1

        return vent_count, float(edge_density)

    def detect_visor_and_cap_features(self, bgr_roi: np.ndarray, hsv_roi: np.ndarray) -> Dict[str, float]:
        """
        Slide P07: Deteksi Lidah Topi (Visor), Bayangan Bawah Lidah, dan Tekstur Topi Kasual.
        - Visor Shadow: Area di bawah lidah topi memiliki penurunan kecerahan drastis.
        - Khaki / Brown Bias: Topi sering berbahan kain warna khaki/tan (R >> B).
        - Hue Diversity: Topi bercorak/camo memiliki variansi hue tinggi.
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
        lower_edges = cv2.Canny(lower_gray, 30, 90)
        lower_edge_ratio = cv2.countNonZero(lower_edges) / max(1, lower_gray.size)

        return {
            "visor_shadow": visor_shadow,
            "r_minus_b": r_minus_b,
            "is_khaki_tone": is_khaki_tone,
            "h_std": h_std,
            "lower_edge_ratio": float(lower_edge_ratio)
        }

    def detect_motorcycle_helmet_features(self, bgr_roi: np.ndarray, hsv_roi: np.ndarray) -> Dict[str, float]:
        """
        Slide P04 & P09: Deteksi Ciri Khas Helm Motor:
        1. Kaca Visor (Face Shield): Bagian tengah mata/wajah ditutupi kaca gelap (V < 45)
           atau pantulan kaca mengilap melintang horizontal.
        2. Cangkang Bulky & Tebal: Melingkupi kepala hingga menutupi telinga dan rahang.
        """
        h, w = bgr_roi.shape[:2]
        gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)

        mid_y1 = int(h * 0.40)
        mid_y2 = int(h * 0.80)
        mid_x1 = int(w * 0.15)
        mid_x2 = int(w * 0.85)

        mid_zone = gray[mid_y1:mid_y2, mid_x1:mid_x2]
        dark_visor_ratio = 0.0
        visor_horizontal_edge = 0.0

        if mid_zone.size > 20:
            dark_pixels = np.count_nonzero(mid_zone < 45)
            dark_visor_ratio = dark_pixels / mid_zone.size

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
            return {"edge_density": 0.0, "drapery_ratio": 1.0, "texture_var": 0.0, "canny_img": np.zeros((10,10), dtype=np.uint8)}

        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 40, 120)
        
        h, w = gray_roi.shape[:2]
        margin_y, margin_x = max(2, int(h * 0.12)), max(2, int(w * 0.12))
        inner_edges = edges[margin_y:h-margin_y, margin_x:w-margin_x]
        
        inner_area = max(1, (h - 2*margin_y) * (w - 2*margin_x))
        edge_density = cv2.countNonZero(inner_edges) / inner_area

        sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        
        mag_x = np.mean(np.abs(sobel_x[margin_y:h-margin_y, margin_x:w-margin_x]))
        mag_y = np.mean(np.abs(sobel_y[margin_y:h-margin_y, margin_x:w-margin_x]))
        
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
                neck_coverage_score = float(cv2.countNonZero(neck_edges) / max(1, neck_zone.shape[0] * neck_zone.shape[1]))

        return {
            "aspect_ratio": float(h / max(1, w)),
            "solidity": solidity,
            "symmetry_score": symmetry_score,
            "neck_coverage_score": neck_coverage_score,
            "morphed_mask": morphed
        }

    def detect_cranial_hair(self, bgr_roi: np.ndarray, hsv_roi: np.ndarray) -> Tuple[float, float]:
        """
        Slide P04 & P09: Deteksi Kehadiran Rambut Asli / Kulit Kepala pada Kubah Atas.
        Pada kepala terbuka tanpa helm, kubah tengah (y: 5%-50%, x: 25%-75%) didominasi
        oleh rambut asli manusia (V < 75 atau V < 95 & S < 80) atau kulit dahi/kepala.
        Sebaliknya, pada helm proyek asli (HDPE kaku), rambut tertutup rapat (hair_ratio < 0.10)
        dan kubah tengah dipenuhi oleh warna solid K3.
        """
        h, w = hsv_roi.shape[:2]
        cy1, cy2 = int(0.05 * h), int(0.50 * h)
        cx1, cx2 = int(0.25 * w), int(0.75 * w)
        c_hsv = hsv_roi[cy1:cy2, cx1:cx2]
        if c_hsv.size == 0:
            return 0.0, 0.0

        v = c_hsv[:, :, 2]
        s = c_hsv[:, :, 1]
        # Rambut gelap khas Indonesia: V < 75 atau (V < 95 dan S < 80)
        dark_hair = (v < 75) | ((v < 95) & (s < 80))
        hair_ratio = float(np.count_nonzero(dark_hair) / dark_hair.size)

        # Kulit dahi/kepala: H <= 25, S in [35, 180], V in [80, 240]
        h_chan = c_hsv[:, :, 0]
        skin = (h_chan <= 25) & (s >= 35) & (s <= 180) & (v >= 80) & (v <= 240)
        skin_ratio = float(np.count_nonzero(skin) / skin.size)

        return hair_ratio, skin_ratio

    def detect_hijab_and_neck_continuity(
        self,
        head_roi: np.ndarray,
        person_crop: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Slide P04, P07 & P09: Analisis Kontinuitas Kain Kerudung / Hijab dari Kepala ke Leher.
        Pada kerudung/hijab:
        1. Kain penutup kepala memanjang ke bawah menutupi telinga, pipi, dan leher.
        2. Area leher (di bawah dagu, di atas dada) memiliki warna kain yang identik
           dengan warna kain penutup kepala (h_diff < 20, s_diff < 45, v_diff < 50).
        3. Area leher tidak memperlihatkan kulit leher terbuka (neck_skin_ratio < 0.25).
        Sebaliknya, pada helm proyek asli K3:
        1. Tempurung helm kaku berakhir di atas dahi/telinga.
        2. Leher pekerja adalah kulit manusia terbuka atau kerah kemeja/rompi yang warnanya
           berbeda drastis dari tempurung helm di atas kepala.
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

        # 1. Warna rata-rata kubah penutup kepala
        crown = head_roi[:max(4, int(hh * 0.60)), :]
        crown_hsv = cv2.cvtColor(crown, cv2.COLOR_BGR2HSV)
        head_mean = np.mean(crown_hsv, axis=(0, 1))

        # 2. Zona leher (di bawah dagu, di atas dada: 48% s.d. 72% tinggi orang)
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

        # Cek rasio kulit leher terbuka (H in [0, 25], S in [35, 180], V in [80, 240])
        skin = (neck_hsv[:, :, 0] <= 25) & (neck_hsv[:, :, 1] >= 35) & (neck_hsv[:, :, 1] <= 180) & (neck_hsv[:, :, 2] >= 80) & (neck_hsv[:, :, 2] <= 240)
        neck_skin_ratio = float(np.count_nonzero(skin) / max(1, neck_hsv.size))

        # Cek lipatan drapery kain di leher (Sobel Y vertikal vs Sobel X)
        neck_gray = cv2.cvtColor(neck, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(neck_gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(neck_gray, cv2.CV_32F, 1, 0, ksize=3)
        neck_drapery = float(np.mean(np.abs(sobel_y)) / (np.mean(np.abs(sobel_x)) + 1e-5))

        # Perbedaan warna antara kepala dan leher (Hue sirkular OpenCV [0, 180])
        h_diff = float(min(abs(head_mean[0] - neck_mean[0]), 180 - abs(head_mean[0] - neck_mean[0])))
        s_diff = float(abs(head_mean[1] - neck_mean[1]))
        v_diff = float(abs(head_mean[2] - neck_mean[2]))

        # Kontinuitas warna: Jika warna penutup kepala dan kain di leher sama
        color_continuity = (h_diff < 20.0) and (s_diff < 45.0) and (v_diff < 50.0)

        # Kerudung terdeteksi jika:
        # a. Kain kepala berlanjut ke leher dan leher tidak berkulit terbuka
        # b. Atau leher memiliki tekstur lipatan drapery kain tinggi dan leher tertutup
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
        Menjalankan seluruh rangkaian analisis citra komprehensif.
        """
        if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.shape[0] < 10 or roi_bgr.shape[1] < 10:
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
                "enhanced_roi": roi_bgr,
                "color_mask": np.zeros((10, 10), dtype=np.uint8),
                "canny_img": np.zeros((10, 10), dtype=np.uint8)
            }

        # 1. Peningkatan Kualitas Citra (Slide P03)
        enhanced_roi = self.enhance_image(roi_bgr)
        hsv_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2GRAY)

        # 2. Segmentasi Model Warna K3 pada Kubah Atas (Slide P04)
        color_mask, best_color, k3_ratio, color_scores = self.extract_k3_color_mask(roi_bgr, hsv_roi)

        # 3. Deteksi Kehadiran Rambut Asli pada Kubah Tengah (Slide P04 & P09)
        hair_ratio, skin_ratio = self.detect_cranial_hair(roi_bgr, hsv_roi)

        # 4. Deteksi Kontinuitas Hijab dari Kepala ke Leher (Slide P04 & P09)
        hijab_data = self.detect_hijab_and_neck_continuity(roi_bgr, person_crop)

        # 5. Analisis Ventilasi Sepeda Berpori Gelap (Slide P07 & P09)
        vent_count, dome_edge_density = self.detect_aerodynamic_vents(gray_roi, hsv_roi)

        # 6. Analisis Ciri Topi Kasual (Slide P07: Lidah Visor & Bayangan)
        cap_feats = self.detect_visor_and_cap_features(enhanced_roi, hsv_roi)

        # 7. Analisis Ciri Helm Motor (Slide P04 & P09: Kaca Visor Gelap / Menutupi Rahang)
        motor_feats = self.detect_motorcycle_helmet_features(enhanced_roi, hsv_roi)

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
            "neck_skin_ratio": hijab_data["neck_skin_ratio"]
        }
