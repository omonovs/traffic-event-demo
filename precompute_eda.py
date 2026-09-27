"""
Sample videolarni oldindan (bir marta) tahlil qilib, natijalarni statik
fayllar sifatida saqlaydi: assets/eda_stats.json va assets/heatmaps/*.png

Nega kerak: sayt hakamlarga ochilganda ular sizning samples/ papkangizni
yuklay olmaydi (fayllar ularda yo'q). Shuning uchun EDA natijalarini
OLDINDAN hisoblab, saytga "qotirib" qo'yamiz — hakamlar hech narsa
yuklamasdan darhol natijalarni ko'radi.

Ishlatish:
    python precompute_eda.py

Bu skript samples/ papkadagi barcha .mp4 fayllarni o'qiydi va
assets/ papkasiga natijalarni yozadi. Ishga tushirgach, assets/ papkasini
ham git'ga qo'shib push qilasiz.
"""

import json
import os

import cv2
import numpy as np

SAMPLES_DIR = "samples"
ASSETS_DIR = "assets"
HEATMAPS_DIR = os.path.join(ASSETS_DIR, "heatmaps")


def video_basic_stats(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps if fps else 0
    cap.release()
    return {"fps": round(fps, 2), "width": w, "height": h,
            "n_frames": n, "duration_sec": round(dur, 1)}


def motion_heatmap(path: str, max_samples: int = 120):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, n // max_samples)
    prev = None
    acc = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if prev is not None:
                diff = cv2.absdiff(gray, prev)
                acc = diff.astype(np.float32) if acc is None else acc + diff.astype(np.float32)
            prev = gray
        idx += 1
    cap.release()
    if acc is None:
        return None
    acc = acc / (acc.max() + 1e-6)
    return acc


def save_heatmap_png(heat: np.ndarray, out_path: str):
    # inferno-ga o'xshash oddiy rangli xarita, matplotlib'siz
    heat_u8 = (heat * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    colored = cv2.resize(colored, (640, 360), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, colored)


def main():
    os.makedirs(HEATMAPS_DIR, exist_ok=True)
    videos = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".mp4")]
    if not videos:
        print(f"'{SAMPLES_DIR}' papkasida .mp4 fayl topilmadi.")
        return

    results = []
    for name in sorted(videos):
        path = os.path.join(SAMPLES_DIR, name)
        print(f"Ishlanmoqda: {name}")
        stats = video_basic_stats(path)
        stats["name"] = name

        heat = motion_heatmap(path)
        heatmap_file = None
        if heat is not None:
            safe_name = "".join(c if c.isalnum() else "_" for c in name)
            heatmap_file = f"{safe_name}_heatmap.png"
            save_heatmap_png(heat, os.path.join(HEATMAPS_DIR, heatmap_file))

        stats["heatmap_file"] = heatmap_file
        results.append(stats)
        print(f"  -> {stats}")

    with open(os.path.join(ASSETS_DIR, "eda_stats.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nTayyor! Natijalar: {ASSETS_DIR}/eda_stats.json va {HEATMAPS_DIR}/*.png")
    print("Endi: git add assets/ && git commit -m 'Precomputed EDA' && git push")


if __name__ == "__main__":
    main()