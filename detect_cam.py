"""
Desktop OpenCV Webcam Launcher
Deteksi Helm Proyek K3 Real-Time Melalui Kamera / Webcam Laptop / CCTV
Jalankan: python detect_cam.py [--source 0] [--conf 0.35]
"""

import cv2
import argparse
import time
from pathlib import Path
from core.yolo_detector import SafetyHelmetDetector

def main():
    parser = argparse.ArgumentParser(description="Deteksi Helm Proyek K3 via Webcam Desktop")
    parser.add_argument("--source", type=str, default="0", help="Index webcam (0, 1) atau URL RTSP/video file")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    args = parser.parse_args()

    # Parse camera index if numeric
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 65)
    print("  MEMBUKA KAMERA DESKTOP - SISTEM K3 HELMET SURVEILLANCE")
    print("=" * 65)
    print(f"[*] Sumber Video: {source}")
    print(f"[*] Confidence: {args.conf}")
    print("[*] Tombol Kontrol:")
    print("    - Tekan 'q' : Keluar dari aplikasi")
    print("    - Tekan 's' : Ambil tangkapan layar (snapshot)")
    print("=" * 65)

    detector = SafetyHelmetDetector()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[!] Error: Gagal membuka sumber video: {source}")
        return

    # Buat direktori snapshots jika belum ada
    snapshots_dir = Path("snapshots")
    snapshots_dir.mkdir(exist_ok=True)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Sinyal kamera terputus atau video selesai.")
            break

        # Jalankan deteksi
        results = detector.detect(frame, conf=args.conf)
        annotated = results["annotated_frame"]

        # Hitung FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (annotated.shape[1] - 120, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("K3 Smart Helmet Detection - OpenCV Desktop", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[*] Menutup aplikasi...")
            break
        elif key == ord('s'):
            filename = snapshots_dir / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(filename), annotated)
            print(f"[+] Snapshot tersimpan: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
