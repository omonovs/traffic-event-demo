"""
STUB solution.py — 1-ISHTIROKCHI haqiqiy solution.py ni tayyorlagunicha ishlatiladi.

Bu fayl hackathon interfeysiga (Output format & interface) mos ravishda
detect_events() va RiskEstimator ni implementatsiya qiladi, lekin
haqiqiy model o'rniga soxta (random/heuristic) natijalar qaytaradi.

1-ISHTIROKCHI kodi tayyor bo'lgach:
  - Bu faylni o'chirmang, yoniga chindan ishlaydigan solution.py ni qo'shing
  - app.py dagi `from solution_stub import ...` qatorini
    `from solution import ...` ga almashtiring
  - Interfeys bir xil bo'lgani uchun boshqa hech narsa o'zgarmaydi
"""

import random
import numpy as np
import cv2

CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]


def _get_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        fps = 25.0
    return n_frames / fps if n_frames else 60.0


def detect_events(video_path: str) -> list:
    """STUB: duration bo'yicha 2-4 ta tasodifiy (lekin realistik ko'rinishdagi)
    hodisa generatsiya qiladi — faqat UI/pipeline ni sinash uchun."""
    duration = _get_duration(video_path)
    n_events = random.randint(2, 4)
    events = []
    used_spans = []
    for _ in range(n_events):
        label = random.choice(CLASSES)
        start = round(random.uniform(0, max(1.0, duration - 5)), 1)
        length = round(random.uniform(1.5, 6.0), 1)
        end = min(duration, round(start + length, 1))
        if end <= start:
            continue
        events.append([start, end, label])
        used_spans.append((start, end))
    events.sort(key=lambda e: e[0])
    return events


class RiskEstimator:
    """STUB: mashinalar sonini simulyatsiya qiluvchi oddiy sinusoidal +
    tasodifiy shovqin bilan xavf balli. Haqiqiy versiyada bu tracked
    obyektlar orasidagi time-to-collision asosida hisoblanadi."""

    def reset(self, meta: dict) -> None:
        self.fps = meta.get("fps", 25.0)
        self.t0 = 0.0
        self._phase = random.uniform(0, 6.28)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        base = 0.15 + 0.15 * np.sin(t_sec / 7.0 + self._phase)
        noise = random.uniform(-0.05, 0.05)
        score = max(0.0, min(1.0, base + noise))
        return float(score)
