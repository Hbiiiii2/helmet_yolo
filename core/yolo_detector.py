"""
Modul YOLODetector
Wrapper deteksi objek menggunakan YOLO11n yang terlatih,
terintegrasi dengan:
1. Pre-Processing Kompensasi Outdoor (Backlight & Glare/Noise Matahari)
2. SAHI-Style Multi-Tile Crowd & Horizon Zoom Inference (Deteksi Objek Kecil / Barisan Jauh)
3. HeadwearClassifier (Metode Pengolahan Citra Digital Slide P02-P11)
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

from .config import (
    get_model_path,
    CONFIDENCE_THRESHOLD,
    OUTDOOR_CANDIDATE_CONF,
    IOU_THRESHOLD,
    LABEL_REAL_HELMET,
    LABEL_TOPI,
    LABEL_HELM_MOTOR,
    LABEL_HELM_SEPEDA,
    LABEL_KERUDUNG,
    LABEL_NO_HELMET,
    STATUS_META
)
from .headwear_classifier import HeadwearClassifier


def _box_iou_and_iom(boxA: List[int], boxB: List[int]) -> Tuple[float, float]:
    """Menghitung Intersection over Union (IoU) dan Intersection over Minimum Area (IoM)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = float(interW * interH)
    if interArea <= 0:
        return 0.0, 0.0

    areaA = float(max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])))
    areaB = float(max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])))
    iou = interArea / (areaA + areaB - interArea)
    iom = interArea / min(areaA, areaB)
    return iou, iom


class SafetyHelmetDetector:
    """
    Sistem Deteksi Keselamatan K3 Terintegrasi:
    Mendeteksi Penggunaan Helm Proyek Asli vs Topi vs Kerudung vs Tanpa Helm,
    serta Rompi Proyek (Vest / No-Vest), dilengkapi Multi-Pass Outdoor Illumination
    Recovery & SAHI-Style Multi-Tile Slicing untuk mengatasi Backlight, Glare/Noise
    Cahaya Matahari, serta foto barisan pekerja jarak jauh (crowd/apel outdoor).
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or get_model_path()
        print(f"[*] Memuat model YOLO dari: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.classifier = HeadwearClassifier()
        self.names = self.model.names
        print(f"[+] Model berhasil dimuat. Kelas model: {self.names}")

        # Inisialisasi detektor wajah Cascade sebagai fallback pengaman di kondisi backlight/glare ekstrem
        self.face_cascade = None
        try:
            cascade_cls = getattr(cv2, "CascadeClassifier", None)
            if cascade_cls is None and hasattr(cv2, "objdetect"):
                cascade_cls = getattr(cv2.objdetect, "CascadeClassifier", None)

            if cascade_cls is not None and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                if os.path.exists(cascade_path):
                    loaded_cascade = cascade_cls(cascade_path)
                    if not loaded_cascade.empty():
                        self.face_cascade = loaded_cascade
        except Exception as e:
            print(f"[*] Info: CascadeClassifier fallback tidak aktif ({e}), sistem tetap berjalan penuh dengan YOLO11.")

        self.tile_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def _parse_yolo_boxes(
        self,
        boxes_data: Any,
        orig_w: int,
        orig_h: int,
        source_tag: str = "raw",
        offset_x: int = 0,
        offset_y: int = 0,
        scale: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Mengekstrak list bounding box dari hasil inferensi YOLO (mendukung pemetaan balik koordinat Tile)."""
        parsed = []
        if boxes_data is None or len(boxes_data) == 0:
            return parsed

        for b in boxes_data:
            xyxy = b.xyxy[0].cpu().numpy()
            cls_id = int(b.cls[0].cpu().numpy())
            confidence = float(b.conf[0].cpu().numpy())
            cls_name = self.names.get(cls_id, str(cls_id))

            x1 = max(0, int(offset_x + xyxy[0] / scale))
            y1 = max(0, int(offset_y + xyxy[1] / scale))
            x2 = min(orig_w, int(offset_x + xyxy[2] / scale))
            y2 = min(orig_h, int(offset_y + xyxy[3] / scale))
            bw, bh = (x2 - x1), (y2 - y1)
            if bw < 5 or bh < 5:
                continue

            # Filter geometri untuk mencegah false-positive garis dinding/tiang pada kelas kepala
            if cls_name in ["helmet", "no-helmet"]:
                aspect = bh / float(max(1, bw))
                if aspect > 1.80 or aspect < 0.42:
                    continue

            parsed.append({
                "bbox": [x1, y1, x2, y2],
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": confidence,
                "source": source_tag,
                "hits": 1
            })
        return parsed

    def _run_tiled_crowd_inference(
        self,
        frame: np.ndarray,
        iou: float = IOU_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        SAHI-Style Multi-Tile Crowd & Horizon Zoom Inference:
        Memotong area barisan pekerja (horizon tengah) menjadi beberapa tile bertumpuk (overlapping),
         melakukan upscaling 2.0x - 2.5x + CLAHE, dan memetakan kembali deteksi ke frame asli.
        Sangat efektif untuk menangkap pekerja jarak menengah/jauh di foto outdoor wide-angle.
        """
        orig_h, orig_w = frame.shape[:2]
        if orig_w < 320 or orig_h < 220:
            return []

        slices = [
            # 4 Tile bertumpuk mencakup barisan pekerja (y: 16% s.d. 74%) dengan upscale 2.0x
            (0, int(orig_h * 0.16), int(orig_w * 0.36), int(orig_h * 0.74), 2.0, "tile_left"),
            (int(orig_w * 0.22), int(orig_h * 0.16), int(orig_w * 0.56), int(orig_h * 0.74), 2.0, "tile_mid_left"),
            (int(orig_w * 0.44), int(orig_h * 0.16), int(orig_w * 0.78), int(orig_h * 0.74), 2.0, "tile_mid_right"),
            (int(orig_w * 0.64), int(orig_h * 0.16), orig_w, int(orig_h * 0.74), 2.0, "tile_right"),
            # 3 Tile super-zoom khusus pita kepala horizon (y: 28% s.d. 48%) dengan upscale 3.0x untuk baris kedua/ketiga
            (int(orig_w * 0.15), int(orig_h * 0.28), int(orig_w * 0.45), int(orig_h * 0.48), 3.0, "zoom_horizon_L"),
            (int(orig_w * 0.38), int(orig_h * 0.28), int(orig_w * 0.68), int(orig_h * 0.48), 3.0, "zoom_horizon_C"),
            (int(orig_w * 0.60), int(orig_h * 0.28), int(orig_w * 0.90), int(orig_h * 0.48), 3.0, "zoom_horizon_R"),
        ]

        tile_boxes: List[Dict[str, Any]] = []
        for (tx1, ty1, tx2, ty2, sc, tag) in slices:
            crop = frame[ty1:ty2, tx1:tx2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            up = cv2.resize(crop, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enh = self.tile_clahe.apply(l)
            up_enh = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)

            res = self.model(up_enh, conf=0.12, iou=iou, imgsz=960, verbose=False)
            parsed = self._parse_yolo_boxes(
                res[0].boxes,
                orig_w=orig_w,
                orig_h=orig_h,
                source_tag=tag,
                offset_x=tx1,
                offset_y=ty1,
                scale=sc
            )
            # Hanya ambil deteksi person & head dari tile crowd agar tidak menduplikasi box vest pada objek jauh
            tile_boxes.extend([b for b in parsed if b["cls_name"] in ["person", "helmet", "no-helmet"]])

        return tile_boxes

    def _merge_candidate_boxes(
        self,
        candidates: List[Dict[str, Any]],
        iou_thresh: float = 0.35,
        iom_thresh: float = 0.60,
        prefer_helmet: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Menggabungkan kandidat bounding box dari multi-pass & multi-tile inferensi
        agar tidak terjadi duplikasi pada objek yang sama namun tetap mempertahankan objek bersebelahan.
        """
        if not candidates:
            return []

        sorted_cands = sorted(
            candidates,
            key=lambda x: (1.05 if (prefer_helmet and x["cls_name"] == "helmet") else 1.0) * x["conf"],
            reverse=True
        )
        merged: List[Dict[str, Any]] = []

        for cand in sorted_cands:
            c_box = cand["bbox"]
            c_area = float(max(1, (c_box[2] - c_box[0]) * (c_box[3] - c_box[1])))
            matched_existing = None
            for ex in merged:
                ex_box = ex["bbox"]
                ex_area = float(max(1, (ex_box[2] - ex_box[0]) * (ex_box[3] - ex_box[1])))
                area_ratio = max(c_area, ex_area) / min(c_area, ex_area)
                iou, iom = _box_iou_and_iom(c_box, ex_box)
                # Hanya gabungkan berdasarkan IoM jika skala luas sebanding (mencegah objek depan menelan objek baris belakang)
                if iou >= iou_thresh or (iom >= iom_thresh and area_ratio <= 2.4):
                    matched_existing = ex
                    break

            if matched_existing is None:
                merged.append(dict(cand))
            else:
                if (
                    prefer_helmet and
                    cand["cls_name"] == "helmet" and
                    matched_existing["cls_name"] != "helmet" and
                    cand["conf"] >= 0.18
                ):
                    matched_existing["cls_name"] = "helmet"
                    matched_existing["cls_id"] = cand["cls_id"]
                matched_existing["conf"] = max(matched_existing["conf"], cand["conf"])
                matched_existing["hits"] = matched_existing.get("hits", 1) + 1

        return merged

    def _discover_crowd_backrow_helmets(
        self,
        frame: np.ndarray,
        existing_heads: List[Dict[str, Any]],
        person_boxes: List[Dict[str, Any]]
    ) -> List[List[int]]:
        """
        Detektor Kubah Helm Baris Belakang (Crowd Horizon Dome Scanner):
        Pada foto barisan/apel outdoor, pekerja di baris kedua sering tertutup badannya oleh
        pekerja di depannya sehingga hanya tampak kubah helm (Putih/Merah/Kuning/Biru K3)
        dan wajah di bawahnya. Fungsi ini memindai pita horizon barisan pekerja untuk menemukan
        kubah helm yang belum terwakili oleh bounding box yang ada.
        """
        if len(existing_heads) < 4:
            return []

        orig_h, orig_w = frame.shape[:2]
        # Filter kepala ukuran barisan menengah/jauh (lebar 10px s.d. 42px)
        crowd_heads = [
            h["bbox"] for h in existing_heads
            if 9 <= (h["bbox"][2] - h["bbox"][0]) <= int(orig_w * 0.065)
        ]
        if len(crowd_heads) < 3:
            return []

        widths = [b[2] - b[0] for b in crowd_heads]
        heights = [b[3] - b[1] for b in crowd_heads]
        med_w = int(np.median(widths))
        med_h = int(np.median(heights))
        if med_w < 10 or med_w > 45:
            return []

        min_x = max(0, min(b[0] for b in crowd_heads) - med_w)
        max_x = min(orig_w, max(b[2] for b in crowd_heads) + med_w)
        min_y = max(0, int(np.percentile([b[1] for b in crowd_heads], 10)) - int(med_h * 0.6))
        max_y = min(orig_h, int(np.percentile([b[3] for b in crowd_heads], 90)) + int(med_h * 0.6))

        if (max_y - min_y) < 12 or (max_x - min_x) < 40:
            return []

        band_bgr = frame[min_y:max_y, min_x:max_x]
        band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
        B, G, R = cv2.split(band_bgr.astype(float))
        H, S, V = cv2.split(band_hsv)

        # Masker kandidat puncak helm K3 (Putih terang seimbang atau Kuning/Merah/Biru jenuh)
        white_dome = (
            (R >= 165) & (G >= 165) & (B >= 165) &
            (np.abs(R - G) <= 25) & (np.abs(G - B) <= 28) & (S <= 42)
        )
        color_dome = (S >= 85) & (V >= 95) & (
            ((H >= 18) & (H <= 38)) |  # Kuning
            (H <= 10) | (H >= 166) |   # Merah/Oranye
            ((H >= 88) & (H <= 135))   # Biru
        )
        dome_mask = ((white_dome | color_dome).astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dome_mask = cv2.morphologyEx(dome_mask, cv2.MORPH_OPEN, kernel)
        dome_mask = cv2.morphologyEx(dome_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(dome_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        discovered: List[List[int]] = []

        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # Ukuran kubah atas helm harus proporsional dengan ukuran kepala barisan
            if cw < int(med_w * 0.42) or cw > int(med_w * 1.45):
                continue
            if ch < 5 or ch > int(med_h * 1.15):
                continue

            # Konstruksi kotak kepala utuh dari puncak kubah helm
            hx1 = max(0, min_x + cx - int(cw * 0.12))
            hx2 = min(orig_w, min_x + cx + cw + int(cw * 0.12))
            hy1 = max(0, min_y + cy - 2)
            hy2 = min(orig_h, hy1 + max(int((hx2 - hx1) * 1.08), int(med_h * 0.92)))

            # Validasi anatomi: Di bawah kubah helm (area wajah/masker) harus lebih gelap dari kubah helm,
            # dan di bawah kepala (area badan/jaket) tidak boleh langit kosong
            dome_patch_v = float(np.mean(V[cy:cy + ch, cx:cx + cw]))
            face_y1 = min(orig_h - 2, hy1 + int((hy2 - hy1) * 0.45))
            face_y2 = min(orig_h, hy2 + 2)
            face_patch = frame[face_y1:face_y2, hx1:hx2]
            if face_patch.size == 0:
                continue
            face_v = float(np.mean(cv2.cvtColor(face_patch, cv2.COLOR_BGR2HSV)[:, :, 2]))

            # Pastikan berada di dalam rentang horizontal barisan pekerja (bukan tembok di pinggir luar)
            dome_cx = (hx1 + hx2) / 2.0
            near_existing_head_x = any(
                abs(dome_cx - ((h["bbox"][0] + h["bbox"][2]) / 2.0)) <= max(9, int(med_w * 0.62))
                for h in existing_heads
            )
            if near_existing_head_x:
                continue

            if dome_cx < (min_x + med_w) or dome_cx > (max_x - int(med_w * 1.8)):
                continue

            # Pastikan di bawah kubah helm ada wajah yang lebih gelap (kontras kubah vs wajah >= 20)
            if dome_patch_v < (face_v + 20.0) or face_v > 135.0:
                continue

            # Pastikan di bawah kepala ada badan/seragam pekerja (bukan tembok/langit polos)
            torso_y1 = min(orig_h - 2, hy2)
            torso_y2 = min(orig_h, hy2 + int((hy2 - hy1) * 1.6))
            torso_patch = frame[torso_y1:torso_y2, hx1:hx2]
            if torso_patch.size == 0:
                continue
            torso_hsv = cv2.cvtColor(torso_patch, cv2.COLOR_BGR2HSV)
            # Badan pekerja memiliki saturasi seragam (merah/kuning/biru) atau pakaian gelap
            has_worker_torso = (
                float(np.mean(torso_hsv[:, :, 1] >= 55)) >= 0.25 or
                float(np.mean(torso_hsv[:, :, 2] <= 95)) >= 0.30
            )
            if not has_worker_torso:
                continue

            cand_box = [hx1, hy1, hx2, hy2]

            # Cek apakah sudah terwakili oleh head_box yang ada
            overlap = False
            for h_item in existing_heads:
                iou_val, iom_val = _box_iou_and_iom(cand_box, h_item["bbox"])
                if iou_val >= 0.18 or iom_val >= 0.38:
                    overlap = True
                    break
            if overlap:
                continue

            for d_box in discovered:
                iou_val, iom_val = _box_iou_and_iom(cand_box, d_box)
                if iou_val >= 0.18 or iom_val >= 0.38:
                    overlap = True
                    break

            if not overlap:
                discovered.append(cand_box)

        return discovered

    def _discover_unidentified_faces_outdoor(
        self,
        enhanced_frame: np.ndarray,
        existing_heads: List[Dict[str, Any]],
        person_boxes: List[Dict[str, Any]]
    ) -> List[List[int]]:
        """
        Fallback deteksi wajah/kepala berbasis Haar Cascade + validasi kulit untuk
        menangkap orang di foto outdoor yang sama sekali terlewat oleh YOLO akibat silau/backlight.
        """
        if self.face_cascade is None or self.face_cascade.empty():
            return []

        orig_h, orig_w = enhanced_frame.shape[:2]
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)

        min_dim = max(24, int(min(orig_h, orig_w) * 0.045))
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(min_dim, min_dim)
        )
        if faces is None or len(faces) == 0:
            return []

        recovered_bboxes: List[List[int]] = []
        for (fx, fy, fw, fh) in faces:
            face_hsv = hsv[fy:fy + fh, fx:fx + fw]
            if face_hsv.size == 0:
                continue
            skin_mask = (
                (face_hsv[:, :, 0] <= 22) &
                (face_hsv[:, :, 1] >= 35) &
                (face_hsv[:, :, 1] <= 170) &
                (face_hsv[:, :, 2] >= 70) &
                (face_hsv[:, :, 2] <= 245)
            )
            skin_ratio = float(np.count_nonzero(skin_mask) / face_hsv.size)
            if skin_ratio < 0.25:
                continue

            hx1 = max(0, int(fx - 0.20 * fw))
            hx2 = min(orig_w, int(fx + 1.20 * fw))
            hy1 = max(0, int(fy - 0.55 * fh))
            hy2 = min(orig_h, int(fy + 1.05 * fh))
            cand_box = [hx1, hy1, hx2, hy2]

            already_covered = False
            for h_item in existing_heads:
                iou, iom = _box_iou_and_iom(cand_box, h_item["bbox"])
                if iou >= 0.15 or iom >= 0.30:
                    already_covered = True
                    break
            if already_covered:
                continue

            for r_box in recovered_bboxes:
                iou, iom = _box_iou_and_iom(cand_box, r_box)
                if iou >= 0.20 or iom >= 0.35:
                    already_covered = True
                    break

            if not already_covered:
                recovered_bboxes.append(cand_box)

        return recovered_bboxes

    def detect(
        self,
        frame: np.ndarray,
        conf: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Menjalankan deteksi pada satu frame citra dengan Multi-Condition Outdoor Recovery
        serta SAHI-Style Multi-Tile Crowd Inference.
        Mengembalikan frame teranotasi dan data analitik kepatuhan K3.
        """
        if frame is None or frame.size == 0:
            return {"annotated_frame": frame, "detections": [], "stats": {}}

        orig_h, orig_w = frame.shape[:2]

        # 1. Buat varian kompensasi pencahayaan outdoor (Backlight Lift + Anti-Glare Denoising)
        outdoor_variants, lighting_info = self.classifier.ip_pipeline.generate_outdoor_recovery_variants(frame)
        is_outdoor_hard = lighting_info.get("is_outdoor_challenging", False)

        cand_conf = max(0.18, conf * 0.62) if is_outdoor_hard else max(0.22, conf * 0.75)

        # 2. Inferensi Pass 1: Citra Asli pada resolusi tinggi (imgsz=960 agar objek jauh/kecil terbaca tajam)
        raw_results = self.model(frame, conf=cand_conf, iou=iou, imgsz=960, verbose=False)
        all_parsed_boxes = self._parse_yolo_boxes(raw_results[0].boxes, orig_w, orig_h, source_tag="raw")

        # 3. Inferensi Pass 2: Varian Kompensasi Outdoor (Backlight / Glare Matahari)
        balanced_frame = frame
        for v_name, v_img in outdoor_variants:
            if v_name == "outdoor_balanced":
                balanced_frame = v_img
            if v_name == "outdoor_balanced" or is_outdoor_hard or len(all_parsed_boxes) < 2:
                v_res = self.model(v_img, conf=cand_conf, iou=iou, imgsz=960, verbose=False)
                v_boxes = self._parse_yolo_boxes(v_res[0].boxes, orig_w, orig_h, source_tag=v_name)
                all_parsed_boxes.extend(v_boxes)

        # 4. Inferensi Pass 3: SAHI-Style Multi-Tile Crowd Slicing jika terdeteksi banyak orang / objek kecil
        prelim_persons = [b for b in all_parsed_boxes if b["cls_name"] == "person"]
        has_small_persons = any((b["bbox"][2] - b["bbox"][0]) < (orig_w * 0.14) for b in prelim_persons)
        is_crowd_or_distant = has_small_persons or len(prelim_persons) >= 3
        if is_crowd_or_distant:
            tiled_boxes = self._run_tiled_crowd_inference(frame, iou=iou)
            all_parsed_boxes.extend(tiled_boxes)

        # Pisahkan kandidat berdasarkan kelompok kelas
        raw_persons = [b for b in all_parsed_boxes if b["cls_name"] == "person"]
        raw_heads = [b for b in all_parsed_boxes if b["cls_name"] in ["helmet", "no-helmet"]]
        raw_vests = [b for b in all_parsed_boxes if b["cls_name"] in ["vest", "no-vest"]]

        # Gabungkan kandidat multi-pass & multi-tile dengan NMS cerdas
        merged_persons = self._merge_candidate_boxes(raw_persons, iou_thresh=0.42, iom_thresh=0.68)
        merged_heads = self._merge_candidate_boxes(raw_heads, iou_thresh=0.32, iom_thresh=0.58, prefer_helmet=True)
        merged_vests = self._merge_candidate_boxes(raw_vests, iou_thresh=0.40, iom_thresh=0.65)

        min_head_dim = 5 if is_crowd_or_distant else 16
        person_boxes = [
            p for p in merged_persons
            if p["conf"] >= conf or p.get("hits", 1) >= 2 or (is_crowd_or_distant and p["conf"] >= 0.18)
        ]
        head_boxes = [
            h for h in merged_heads
            if (
                (h["bbox"][2] - h["bbox"][0]) >= min_head_dim and
                (h["bbox"][3] - h["bbox"][1]) >= min_head_dim and
                (h["conf"] >= conf or h.get("hits", 1) >= 2 or (is_crowd_or_distant and h["conf"] >= 0.14))
            )
        ]
        vest_boxes = [
            v for v in merged_vests
            if v["conf"] >= max(conf, 0.42)
        ]

        annotated_frame = frame.copy()
        processed_heads: List[Dict[str, Any]] = []
        matched_person_indices = set()

        # 5. Proses verifikasi setiap head_box yang terdeteksi langsung oleh YOLO
        for head in head_boxes:
            hx1, hy1, hx2, hy2 = head["bbox"]
            head_crop = frame[hy1:hy2, hx1:hx2]
            if head_crop.size == 0:
                continue

            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0

            # Cari 1 person terbaik yang membungkus kepala ini (berdasarkan jarak horizontal terdekat)
            best_p_idx = None
            best_x_dist = 1e9
            for idx, p in enumerate(person_boxes):
                px1, py1, px2, py2 = p["bbox"]
                pw = px2 - px1
                ph = py2 - py1
                pcx = (px1 + px2) / 2.0
                if (
                    abs(hcx - pcx) <= (0.42 * pw) and
                    (py1 - 0.20 * ph) <= hcy <= (py1 + 0.45 * ph)
                ):
                    x_dist = abs(hcx - pcx)
                    if x_dist < best_x_dist:
                        best_x_dist = x_dist
                        best_p_idx = idx

            person_crop = None
            if best_p_idx is not None:
                matched_person_indices.add(best_p_idx)
                px1, py1, px2, py2 = person_boxes[best_p_idx]["bbox"]
                person_crop = frame[py1:py2, px1:px2]

            classification = self.classifier.classify_headwear(
                yolo_class=head["cls_name"],
                yolo_conf=head["conf"],
                head_roi=head_crop,
                person_crop=person_crop
            )

            processed_heads.append({
                "bbox": [hx1, hy1, hx2, hy2],
                "yolo_cls": head["cls_name"],
                "yolo_conf": head["conf"],
                "classification": classification,
                "crop": head_crop
            })

        # 6. Untuk setiap person yang belum memiliki head_box, lakukan inspeksi upper-head secara presisi
        for idx, p in enumerate(person_boxes):
            if idx in matched_person_indices:
                continue

            px1, py1, px2, py2 = p["bbox"]
            pw = px2 - px1
            ph = py2 - py1
            # Mendukung pekerja jarak jauh di foto outdoor (lebar >= 11px, tinggi >= 26px)
            if ph < 26 or pw < 11:
                continue

            # Proporsi tinggi kepala adaptif terhadap rasio tubuh (full-body berdiri vs close-up)
            p_aspect = ph / float(max(1, pw))
            if p_aspect >= 2.1:
                head_ratio_h = 0.23
                x_margin = 0.06 if pw < 35 else 0.12
            elif p_aspect >= 1.5:
                head_ratio_h = 0.30
                x_margin = 0.10
            else:
                head_ratio_h = 0.36
                x_margin = 0.12

            head_y1 = max(0, py1 - int(ph * 0.02))
            head_y2 = min(orig_h, py1 + max(12, int(ph * head_ratio_h)))
            head_x1 = max(0, px1 + int(pw * x_margin))
            head_x2 = min(orig_w, px2 - int(pw * x_margin))
            if (head_x2 - head_x1) < 8 or (head_y2 - head_y1) < 8:
                continue

            est_box = [head_x1, head_y1, head_x2, head_y2]
            est_cx = (head_x1 + head_x2) / 2.0
            est_cy = (head_y1 + head_y2) / 2.0
            est_w = head_x2 - head_x1
            est_h = head_y2 - head_y1

            # Cek tumpang tindih hanya terhadap kepala milik orang yang sama (tidak menekan bahu tetangga di barisan)
            head_already_exists = False
            for h_res in processed_heads:
                hx1, hy1, hx2, hy2 = h_res["bbox"]
                hcx = (hx1 + hx2) / 2.0
                hcy = (hy1 + hy2) / 2.0
                iou_val, iom_val = _box_iou_and_iom(est_box, h_res["bbox"])
                if (
                    iou_val >= 0.22 or
                    iom_val >= 0.45 or
                    (abs(hcx - est_cx) <= 0.42 * est_w and abs(hcy - est_cy) <= 0.48 * est_h)
                ):
                    head_already_exists = True
                    break

            if not head_already_exists:
                head_crop = frame[head_y1:head_y2, head_x1:head_x2]
                person_crop = frame[py1:py2, px1:px2]
                if head_crop.size == 0:
                    continue

                classification = self.classifier.classify_headwear(
                    yolo_class="person-head",
                    yolo_conf=max(0.75, p["conf"]),
                    head_roi=head_crop,
                    person_crop=person_crop
                )

                if (
                    py1 <= 5 and
                    classification["k3_color_ratio"] < 0.25 and
                    classification["skin_ratio"] < 0.04
                ):
                    continue

                processed_heads.append({
                    "bbox": est_box,
                    "yolo_cls": "person-head",
                    "yolo_conf": max(0.75, p["conf"]),
                    "classification": classification,
                    "crop": head_crop
                })

        # 7. Pemindaian Kubah Helm Baris Belakang (Crowd Back-Row Dome Scanner)
        backrow_boxes = self._discover_crowd_backrow_helmets(
            frame=frame,
            existing_heads=processed_heads,
            person_boxes=person_boxes
        )
        for br_box in backrow_boxes:
            bx1, by1, bx2, by2 = br_box
            head_crop = frame[by1:by2, bx1:bx2]
            if head_crop.size == 0:
                continue
            classification = self.classifier.classify_headwear(
                yolo_class="helmet",
                yolo_conf=0.78,
                head_roi=head_crop,
                person_crop=None
            )
            if classification["label"] == LABEL_REAL_HELMET:
                processed_heads.append({
                    "bbox": br_box,
                    "yolo_cls": "crowd-dome",
                    "yolo_conf": 0.78,
                    "classification": classification,
                    "crop": head_crop
                })

        # 8. Fallback Cascade Wajah untuk foto outdoor ekstrem jika masih ada orang yang luput
        if is_outdoor_hard or len(processed_heads) == 0:
            extra_face_boxes = self._discover_unidentified_faces_outdoor(
                enhanced_frame=balanced_frame,
                existing_heads=processed_heads,
                person_boxes=person_boxes
            )
            for f_box in extra_face_boxes:
                fx1, fy1, fx2, fy2 = f_box
                head_crop = frame[fy1:fy2, fx1:fx2]
                if head_crop.size == 0:
                    continue
                classification = self.classifier.classify_headwear(
                    yolo_class="person-head",
                    yolo_conf=0.78,
                    head_roi=head_crop,
                    person_crop=None
                )
                processed_heads.append({
                    "bbox": f_box,
                    "yolo_cls": "face-fallback",
                    "yolo_conf": 0.78,
                    "classification": classification,
                    "crop": head_crop
                })

        # 9. Deduplikasi akhir pada processed_heads (toleran terhadap pekerja yang berdiri rapat di barisan)
        deduped_heads: List[Dict[str, Any]] = []
        processed_heads.sort(
            key=lambda h: (
                1 if h["yolo_cls"] in ["helmet", "no-helmet"] else 0,
                1 if h["classification"]["label"] == LABEL_REAL_HELMET else 0,
                h["yolo_conf"]
            ),
            reverse=True
        )
        for h_item in processed_heads:
            is_dup = False
            for kept in deduped_heads:
                iou_val, iom_val = _box_iou_and_iom(h_item["bbox"], kept["bbox"])
                if iou_val >= 0.35 or iom_val >= 0.62:
                    is_dup = True
                    break
            if not is_dup:
                deduped_heads.append(h_item)

        processed_heads = deduped_heads

        # --- Visualisasi & Anotasi Bounding Box Berkualitas Tinggi (Scale-Adaptive) ---
        stats_count = {
            LABEL_REAL_HELMET: 0,
            LABEL_TOPI: 0,
            LABEL_HELM_MOTOR: 0,
            LABEL_HELM_SEPEDA: 0,
            LABEL_KERUDUNG: 0,
            LABEL_NO_HELMET: 0,
            "vest": 0,
            "no-vest": 0
        }

        # Gambar box orang (tipis/subtle)
        for p in person_boxes:
            px1, py1, px2, py2 = p["bbox"]
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (180, 180, 180), 1)

        # Gambar box rompi
        for v in vest_boxes:
            vx1, vy1, vx2, vy2 = v["bbox"]
            vcls = v["cls_name"]
            stats_count[vcls] = stats_count.get(vcls, 0) + 1
            vcolor = (0, 200, 0) if vcls == "vest" else (0, 100, 255)
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), vcolor, 2)
            cv2.putText(
                annotated_frame,
                f"{vcls.upper()} ({v['conf']:.2f})",
                (vx1, max(15, vy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                vcolor,
                1,
                cv2.LINE_AA
            )

        # Gambar box penutup kepala dengan mode adaptif (Compact untuk barisan kecil vs Full untuk close-up)
        is_crowd_scene = len(processed_heads) >= 8
        for h_item in processed_heads:
            x1, y1, x2, y2 = h_item["bbox"]
            bw = x2 - x1
            cls_info = h_item["classification"]
            lbl = cls_info["label"]
            conf_val = cls_info["confidence"]
            k3_color = cls_info["detected_k3_color"]
            short_color = k3_color.split(" ")[0].upper() if k3_color != "Bukan Warna K3" else "K3"

            stats_count[lbl] = stats_count.get(lbl, 0) + 1

            if lbl == LABEL_REAL_HELMET:
                bgr_color = (0, 210, 80)  # Hijau terang
                label_txt = f"HELM PROYEK K3 ({conf_val:.0%})"
                sub_txt = f"{k3_color}"
                compact_txt = f"K3-{short_color}"
            elif lbl == LABEL_TOPI:
                bgr_color = (0, 140, 255)  # Oranye/Amber
                label_txt = "TOPI (BUKAN HELM)"
                sub_txt = "Pelanggaran K3!"
                compact_txt = "TOPI"
            elif lbl == LABEL_HELM_MOTOR:
                bgr_color = (230, 180, 0)  # Cyan/Biru muda
                label_txt = "HELM MOTOR (BUKAN K3)"
                sub_txt = "Dilarang di Proyek!"
                compact_txt = "H.MOTOR"
            elif lbl == LABEL_HELM_SEPEDA:
                bgr_color = (0, 215, 255)  # Kuning cerah
                label_txt = "HELM SEPEDA (BUKAN K3)"
                sub_txt = "Tidak Standar K3!"
                compact_txt = "H.SEPEDA"
            elif lbl == LABEL_KERUDUNG:
                bgr_color = (210, 50, 180)  # Ungu/Magenta
                label_txt = "KERUDUNG (BUKAN HELM)"
                sub_txt = "Wajib Pakai Helm K3!"
                compact_txt = "HIJAB"
            else:
                bgr_color = (0, 0, 230)  # Merah bahaya
                label_txt = f"TANPA HELM ({conf_val:.0%})"
                sub_txt = "BAHAYA K3!"
                compact_txt = "NO-HELM"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bgr_color, 2)

            if bw < 52 or (is_crowd_scene and bw < 68):
                # Mode Compact Badge agar label pada barisan pekerja (crowd) rapi & tidak saling menutupi
                tag_w = max(bw, len(compact_txt) * 6 + 6)
                tag_h = 14
                ty1 = max(0, y1 - tag_h)
                cv2.rectangle(annotated_frame, (x1, ty1), ( min(orig_w, x1 + tag_w), y1), bgr_color, -1)
                cv2.putText(
                    annotated_frame,
                    compact_txt,
                    (x1 + 2, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
            else:
                # Mode Detail 2 Baris untuk objek kepala ukuran sedang/besar
                header_h = 32
                cv2.rectangle(
                    annotated_frame,
                    (x1, max(0, y1 - header_h)),
                    (min(orig_w, x1 + max(195, bw)), y1),
                    bgr_color,
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    label_txt,
                    (x1 + 4, max(12, y1 - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame,
                    sub_txt,
                    (x1 + 4, max(26, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Hitung Ringkasan Kepatuhan
        total_workers = len(processed_heads)
        compliant_count = stats_count[LABEL_REAL_HELMET]
        violation_count = (
            stats_count[LABEL_TOPI] +
            stats_count[LABEL_HELM_MOTOR] +
            stats_count[LABEL_HELM_SEPEDA] +
            stats_count[LABEL_KERUDUNG] +
            stats_count[LABEL_NO_HELMET]
        )
        compliance_rate = (compliant_count / total_workers * 100.0) if total_workers > 0 else 100.0

        summary_stats = {
            "total_workers": total_workers,
            "compliant": compliant_count,
            "violations": violation_count,
            "compliance_rate": round(compliance_rate, 1),
            "lighting_condition": lighting_info.get("condition_label", "Normal"),
            "breakdown": {
                "real_helmet": stats_count[LABEL_REAL_HELMET],
                "topi": stats_count[LABEL_TOPI],
                "helm_motor": stats_count[LABEL_HELM_MOTOR],
                "helm_sepeda": stats_count[LABEL_HELM_SEPEDA],
                "kerudung": stats_count[LABEL_KERUDUNG],
                "no_helmet": stats_count[LABEL_NO_HELMET]
            }
        }

        # Gambar Banner Statistik Ringkas di pojok atas frame
        banner_w = 370
        banner_h = 90
        cv2.rectangle(annotated_frame, (10, 10), (10 + banner_w, 10 + banner_h), (25, 25, 30), -1)
        cv2.rectangle(annotated_frame, (10, 10), (10 + banner_w, 10 + banner_h), (80, 80, 80), 1)

        status_color = (
            (0, 220, 100) if compliance_rate >= 80
            else (0, 140, 255) if compliance_rate >= 50
            else (0, 0, 240)
        )
        cv2.putText(
            annotated_frame,
            "SISTEM K3 HELMET DETECTION",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated_frame,
            f"Kepatuhan: {compliance_rate:.1f}%",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated_frame,
            f"Aman: {compliant_count} | Pelanggaran: {violation_count}",
            (20, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated_frame,
            f"Mode: {lighting_info.get('condition_label', 'Normal')[:48]}",
            (20, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (130, 210, 255),
            1,
            cv2.LINE_AA
        )

        return {
            "annotated_frame": annotated_frame,
            "detections": processed_heads,
            "person_count": max(len(person_boxes), total_workers),
            "lighting_info": lighting_info,
            "stats": summary_stats
        }
