"""
CLI Image Processing Script
Deteksi Citra Helm K3 & Ekspor Hasil Anotasi
Jalankan: python detect_image.py --input path/to/image.jpg [--output output.jpg]
"""

import cv2
import argparse
import sys
from pathlib import Path
from core.yolo_detector import SafetyHelmetDetector

def main():
    parser = argparse.ArgumentParser(description="Deteksi Helm K3 pada Citra Statis")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path ke file citra input")
    parser.add_argument("--output", "-o", type=str, default="output_detected.jpg", help="Path output citra hasil deteksi")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] File tidak ditemukan: {args.input}")
        sys.exit(1)

    img = cv2.imread(str(input_path))
    if img is None:
        print(f"[!] Gagal membaca citra dari {args.input}")
        sys.exit(1)

    print(f"[*] Memproses citra: {args.input}")
    detector = SafetyHelmetDetector()
    results = detector.detect(img, conf=args.conf)

    cv2.imwrite(args.output, results["annotated_frame"])
    print(f"[+] Hasil deteksi disimpan di: {args.output}")

    stats = results["stats"]
    print("\n--- RINGKASAN KEPATUHAN K3 ---")
    print(f"Total Pekerja     : {stats['total_workers']}")
    print(f"Helm Proyek Resmi : {stats['compliant']}")
    print(f"Pelanggaran K3    : {stats['violations']}")
    print(f"Tingkat Kepatuhan : {stats['compliance_rate']}%")
    print(f"Rincian           : {stats['breakdown']}")
    print("------------------------------\n")

if __name__ == "__main__":
    main()
