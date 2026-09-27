"""
Sample videolarni oldindan tahlil qilib, "Natijalar" tabi uchun statik
natijalarni saqlaydi:

  assets/results/<video>.json        — events + har bir kadr uchun risk qiymati
  assets/results/<video>_annotated.mp4 — faol hodisa banneri va risk qiymati
                                          "kuydirilgan" (burned-in) video
  predictions_samples.json           — repo root'ida, hackathon talab qilgan
                                          rasmiy format (team + videos)

MUHIM: bu skript hozircha solution_stub bilan ishlaydi. 1-ISHTIROKCHINING
haqiqiy solution.py fayli tayyor bo'lgach:
  1) pastdagi importni almashtiring (bitta qatorni ochib, bittasini yoping)
  2) `python precompute_results.py` ni qayta ishga tushiring
  3) yangilangan assets/ va predictions_samples.json ni push qiling
Boshqa hech narsa o'zgartirilmaydi — app.py "Natijalar" tabi avtomatik
yangi natijalarni ko'rsatadi.

Ishlatish:
    python precompute_results.py
"""

import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

# ---- MUHIM: haqiqiy model tayyor bo'lganda shu qatorni almashtiring ----
from solution_stub import detect_events, RiskEstimator, CLASSES
# from solution import detect_events, RiskEstimator, CLASSES
# --------------------------------------------------------------------

TEAM_NAME = "omonovs"

SAMPLES_DIR = "samples"
ASSETS_DIR = "assets"
RESULTS_DIR = os.path.join(ASSETS_DIR, "results")
PREDICTIONS_OUT = "predictions_samples.json"

# Annotatsiyalangan video uchun sozlamalar (tez ishlashi va fayl hajmi kichik
# bo'lishi uchun asl videoning oldi qisqartirilgan, chunki bu faqat vizual
# namoyish uchun — yashirin test baholashiga aloqasi yo'q)
MAX_OUT_WIDTH = 960
ALARM_THRESHOLD = 0.5

# Har bir sinf uchun barqaror (deterministik) rang — BGR format (OpenCV)
def _class_color_bgr(label: str):
    h = abs(hash(label)) % 360
    hsv = np.uint8([[[h // 2, 220, 255]]])  # OpenCV hue 0-179
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def active_events_at(events, t):
    return [e for e in events if e[0] <= t <= e[1]]


def draw_overlay(frame, t_sec, active, risk_score):
    h, w = frame.shape[:2]
    banner_h = 34 + 26 * max(1, len(active))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, f"t = {t_sec:5.1f}s", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    risk_color = (0, 0, 255) if risk_score >= ALARM_THRESHOLD else (0, 200, 0)
    cv2.putText(frame, f"risk = {risk_score:.2f}", (150, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, risk_color, 2, cv2.LINE_AA)
    if risk_score >= ALARM_THRESHOLD:
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 255), 4)

    y = 46
    if active:
        for label, *_ in [(e[2],) for e in active]:
            color = _class_color_bgr(label)
            cv2.putText(frame, f"* {label}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y += 26
    else:
        cv2.putText(frame, "hodisa yo'q", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def open_writer(out_path, fps, size):
    """avc1 (H.264) bilan urinadi — brauzerda to'g'ri ishlaydi.
    Agar mavjud bo'lmasa mp4v'ga qaytadi (keyin ffmpeg bilan qayta kodlashga
    harakat qilamiz)."""
    for fourcc_str in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(out_path, fourcc, fps, size)
        if vw.isOpened():
            return vw, fourcc_str
    raise RuntimeError("VideoWriter ochilmadi — OpenCV kodeklarini tekshiring.")


def _find_ffmpeg():
    """Avval tizim PATH'idagi ffmpeg'ni, topilmasa imageio-ffmpeg paketi bilan
    birga keladigan portativ ffmpeg binary'sini qidiradi (pip install imageio-ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def maybe_reencode_with_ffmpeg(path):
    """ffmpeg (tizim yoki imageio-ffmpeg orqali) topilsa, faylni brauzer uchun
    kafolatlangan H.264 + faststart formatiga qayta kodlaydi. Topilmasa, jim
    o'tkazib yuboradi."""
    ffmpeg_exe = _find_ffmpeg()
    if ffmpeg_exe is None:
        print("  [!] ffmpeg topilmadi — `pip install imageio-ffmpeg` ni bajarib "
              "skriptni qayta ishga tushiring, video brauzerda ishlashi uchun shart.")
        return
    tmp_path = path + ".tmp.mp4"
    cmd = [ffmpeg_exe, "-y", "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-loglevel", "error", tmp_path]
    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp_path, path)
        print("  [ok] ffmpeg orqali brauzerga mos H.264'ga qayta kodlandi.")
    except Exception as e:
        print(f"  [!] ffmpeg qayta kodlashda xato ({e}) — mp4v fayl qoldirildi.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_video(name: str):
    path = os.path.join(SAMPLES_DIR, name)
    print(f"Ishlanmoqda: {name}")

    events = detect_events(path)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps else 0

    scale = min(1.0, MAX_OUT_WIDTH / w) if w else 1.0
    out_w, out_h = int(w * scale), int(h * scale)

    est = RiskEstimator()
    est.reset({"video_id": name, "fps": fps, "width": w, "height": h, "n_frames": n_frames})

    safe_name = "".join(c if c.isalnum() else "_" for c in name)
    out_video_path = os.path.join(RESULTS_DIR, f"{safe_name}_annotated.mp4")
    writer, fourcc_used = open_writer(out_video_path, fps, (out_w, out_h))

    risk_series = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        score = float(est.step(frame, t))
        risk_series.append([round(t, 3), round(score, 4)])

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h))
        active = active_events_at(events, t)
        frame = draw_overlay(frame, t, active, score)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"  -> {len(events)} hodisa, {len(risk_series)} risk nuqtasi, kodek={fourcc_used}")

    # Har doim qayta kodlashga harakat qilamiz — ba'zi Windows OpenCV
    # build'larida avc1/H264 "ochiladi" (isOpened=True), lekin fayl brauzerda
    # baribir ishlamaydi, shuning uchun mp4v'ga tayanib qolmaymiz.
    maybe_reencode_with_ffmpeg(out_video_path)

    result = {
        "name": name,
        "fps": round(fps, 2),
        "width": w,
        "height": h,
        "duration_sec": round(duration, 1),
        "events": events,
        "risk": risk_series,
        "annotated_video": os.path.basename(out_video_path),
    }
    with open(os.path.join(RESULTS_DIR, f"{safe_name}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    if not os.path.isdir(SAMPLES_DIR):
        print(f"'{SAMPLES_DIR}' papkasi topilmadi. Skriptni repo root'idan ishga tushiring.")
        sys.exit(1)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    videos = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".mp4"))
    if not videos:
        print(f"'{SAMPLES_DIR}' papkasida .mp4 fayl topilmadi.")
        return

    predictions = {"team": TEAM_NAME, "videos": {}}
    for name in videos:
        result = process_video(name)
        predictions["videos"][name] = {"events": result["events"], "risk": result["risk"]}

    with open(PREDICTIONS_OUT, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\nTayyor!")
    print(f"  - {RESULTS_DIR}/*.json va *_annotated.mp4 — 'Natijalar' tabi uchun")
    print(f"  - {PREDICTIONS_OUT} — repo root'ida, hackathon talabiga mos")
    print("\nEndi:")
    print("  git add assets/results predictions_samples.json")
    print("  git commit -m 'Precomputed Natijalar tab results'")
    print("  git push")


if __name__ == "__main__":
    main()