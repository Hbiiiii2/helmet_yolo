"""
Modul YOLODetector
Wrapper deteksi objek menggunakan YOLO11n yang terlatih,
terintegrasi langsung dengan HeadwearClassifier (Metode Pengolahan Citra).
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

from .config import (
    get_model_path,
    CONFIDENCE_THRESHOLD,
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


class SafetyHelmetDetector:
    """
    Sistem Deteksi Keselamatan K3 Terintegrasi:
    Mendeteksi Penggunaan Helm Proyek Asli vs Topi vs Kerudung vs Tanpa Helm,
    serta Rompi Proyek (Vest / No-Vest).
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or get_model_path()
        print(f"[*] Memuat model YOLO dari: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.classifier = HeadwearClassifier()
        self.names = self.model.names
        print(f"[+] Model berhasil dimuat. Kelas model: {self.names}")

    def detect(
        self,
        frame: np.ndarray,
        conf: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Menjalankan deteksi pada satu frame citra.
        Mengembalikan frame teranotasi dan data analitik kepatuhan K3.
        """
        if frame is None or frame.size == 0:
            return {"annotated_frame": frame, "detections": [], "stats": {}}

        orig_h, orig_w = frame.shape[:2]
        results = self.model(frame, conf=conf, iou=iou, verbose=False)
        
        boxes_data = results[0].boxes
        detections = []

        annotated_frame = frame.copy()

        # Pisahkan deteksi berdasarkan kelas
        person_boxes = []
        head_boxes = []
        vest_boxes = []

        if boxes_data is not None and len(boxes_data) > 0:
            for b in boxes_data:
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(b.cls[0].cpu().numpy())
                confidence = float(b.conf[0].cpu().numpy())
                cls_name = self.names.get(cls_id, str(cls_id))

                x1, y1, x2, y2 = xyxy
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)

                item = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": confidence
                }

                if cls_name == "person":
                    person_boxes.append(item)
                elif cls_name in ["helmet", "no-helmet"]:
                    head_boxes.append(item)
                elif cls_name in ["vest", "no-vest"]:
                    vest_boxes.append(item)

        # Proses verifikasi penutup kepala
        processed_heads = []
        used_head_indices = set()

        # 1. Asosiasikan head_boxes dengan person jika ada, atau proses langsung
        for head in head_boxes:
            hx1, hy1, hx2, hy2 = head["bbox"]
            head_crop = frame[hy1:hy2, hx1:hx2]
            
            # Cari person pembungkus untuk context leher/bahu
            matched_person = None
            for p in person_boxes:
                px1, py1, px2, py2 = p["bbox"]
                if px1 <= hx1 and py1 <= hy1 and px2 >= hx2 and py2 >= hy2:
                    matched_person = p
                    break
            
            person_crop = None
            if matched_person:
                px1, py1, px2, py2 = matched_person["bbox"]
                person_crop = frame[py1:py2, px1:px2]

            # Analisis multi-tahap (Pengolahan Citra + YOLO)
            classification = self.classifier.classify_headwear(
                yolo_class=head["cls_name"],
                yolo_conf=head["conf"],
                head_roi=head_crop,
                person_crop=person_crop
            )

            head_result = {
                "bbox": head["bbox"],
                "yolo_cls": head["cls_name"],
                "yolo_conf": head["conf"],
                "classification": classification,
                "crop": head_crop
            }
            processed_heads.append(head_result)

        # 2. Jika ada person yang tidak memiliki head_boxes, lakukan inspeksi upper-head
        for p in person_boxes:
            px1, py1, px2, py2 = p["bbox"]
            pw = px2 - px1
            ph = py2 - py1
            
            # Cek apakah sudah ada head_box di area atas person ini
            head_center_in_p = False
            for h_res in processed_heads:
                hx1, hy1, hx2, hy2 = h_res["bbox"]
                hcx = (hx1 + hx2) / 2
                hcy = (hy1 + hy2) / 2
                if px1 <= hcx <= px2 and py1 <= hcy <= (py1 + ph * 0.45):
                    head_center_in_p = True
                    break

            if not head_center_in_p and ph > 40 and pw > 20:
                # Estimasi crop kepala (atas 30% dari bounding box orang)
                head_y1 = py1
                head_y2 = min(orig_h, py1 + int(ph * 0.32))
                head_x1 = max(0, px1 + int(pw * 0.15))
                head_x2 = min(orig_w, px2 - int(pw * 0.15))
                
                head_crop = frame[head_y1:head_y2, head_x1:head_x2]
                person_crop = frame[py1:py2, px1:px2]

                classification = self.classifier.classify_headwear(
                    yolo_class="no-helmet",
                    yolo_conf=0.75,
                    head_roi=head_crop,
                    person_crop=person_crop
                )

                processed_heads.append({
                    "bbox": [head_x1, head_y1, head_x2, head_y2],
                    "yolo_cls": "no-helmet",
                    "yolo_conf": 0.75,
                    "classification": classification,
                    "crop": head_crop
                })

        # --- Visualisasi & Anotasi Bounding Box Berkualitas Tinggi ---
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
                0.45,
                vcolor,
                1,
                cv2.LINE_AA
            )

        # Gambar box penutup kepala dengan warna & label jelas
        for h_item in processed_heads:
            x1, y1, x2, y2 = h_item["bbox"]
            cls_info = h_item["classification"]
            lbl = cls_info["label"]
            conf_val = cls_info["confidence"]
            k3_color = cls_info["detected_k3_color"]

            stats_count[lbl] = stats_count.get(lbl, 0) + 1

            # Warna BGR sesuai status
            if lbl == LABEL_REAL_HELMET:
                bgr_color = (0, 210, 80) # Hijau terang
                label_txt = f"HELM PROYEK K3 ({conf_val:.0%})"
                sub_txt = f"{k3_color}"
            elif lbl == LABEL_TOPI:
                bgr_color = (0, 140, 255) # Oranye/Amber
                label_txt = f"TOPI (BUKAN HELM)"
                sub_txt = "Pelanggaran K3!"
            elif lbl == LABEL_HELM_MOTOR:
                bgr_color = (230, 180, 0) # Cyan/Biru muda
                label_txt = f"HELM MOTOR (BUKAN K3)"
                sub_txt = "Dilarang di Proyek!"
            elif lbl == LABEL_HELM_SEPEDA:
                bgr_color = (0, 215, 255) # Kuning cerah
                label_txt = f"HELM SEPEDA (BUKAN K3)"
                sub_txt = "Tidak Standar K3!"
            elif lbl == LABEL_KERUDUNG:
                bgr_color = (210, 50, 180) # Ungu/Magenta
                label_txt = f"KERUDUNG (BUKAN HELM)"
                sub_txt = "Wajib Pakai Helm K3!"
            else:
                bgr_color = (0, 0, 230) # Merah bahaya
                label_txt = f"TANPA HELM ({conf_val:.0%})"
                sub_txt = "BAHAYA K3!"

            # Border Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bgr_color, 2)

            # Header Box Background
            header_h = 32
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - header_h)), (x1 + max(195, (x2 - x1)), y1), bgr_color, -1)
            
            # Teks Label
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
        banner_w = 260
        banner_h = 75
        cv2.rectangle(annotated_frame, (10, 10), (10 + banner_w, 10 + banner_h), (25, 25, 30), -1)
        cv2.rectangle(annotated_frame, (10, 10), (10 + banner_w, 10 + banner_h), (80, 80, 80), 1)
        
        status_color = (0, 220, 100) if compliance_rate >= 80 else (0, 140, 255) if compliance_rate >= 50 else (0, 0, 240)
        cv2.putText(annotated_frame, "SISTEM K3 HELMET DETECTION", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Kepatuhan: {compliance_rate:.1f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Aman: {compliant_count} | Pelanggaran: {violation_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        return {
            "annotated_frame": annotated_frame,
            "detections": processed_heads,
            "person_count": len(person_boxes),
            "stats": summary_stats
        }
