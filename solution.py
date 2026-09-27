"""
Final WIUT Hackathon 2026 CV submission.

Part A:
    detect_events(video_path) -> [[start_sec, end_sec, label], ...]

Part B:
    RiskEstimator.reset(meta)
    RiskEstimator.step(frame, t_sec) -> P(accident starts within 5 seconds)

This file intentionally uses only the event classes for which this repository
currently has an implemented detector.

The organizer harness imports this file directly. Do not rename:
    CLASSES
    detect_events
    RiskEstimator
"""

from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# solution.py lives inside wiut_cv_scripts/, while our custom src/ and config/
# live one directory above it.
STARTER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STARTER_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scene import SceneConfig

from src.events.jaywalking import JaywalkingDetector
from src.events.stopped_vehicle import StoppedVehicleDetector
from src.events.congestion import CongestionDetector
from src.events.failure_to_yield import FailureToYieldDetector
from src.events.wrong_way import WrongWayDetector
from src.events.collision_risk import CollisionRiskEngine


# ============================================================
# Official interface
# ============================================================

CLASSES: list[str] = [
    "accident",
    "near_miss",
    "wrong_way",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "congestion",
]

RISK_HORIZON_SEC = 5.0


# ============================================================
# Paths / constants
# ============================================================

# Actual repository layout:
#
# HackatonWUIT/
# ├── config/
# ├── src/
# └── wiut_cv_scripts/
#     ├── solution.py
#     ├── run_submission.py
#     ├── evaluate.py
#     └── yolo11n.pt
#
# Therefore:
#   model  -> wiut_cv_scripts/yolo11n.pt
#   config -> ../config/
#   src    -> ../src/

MODEL_PATH = STARTER_ROOT / "yolo11n.pt"

BASE_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "scene_config.json"
)

C3902_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "C3902.json"
)

PERSON_CLASS = 0

MOTOR_VEHICLE_CLASSES = {
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
}

TRACK_CLASSES = [
    0,  # person
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
]

# Part A and Part B each sample the video internally.
# Official harness may call RiskEstimator.step on every source frame.
TARGET_EVENT_FPS = 8.0
TARGET_RISK_FPS = 7.0

DETECTION_CONF = 0.25

# Our current wrong-way rule works, but the samples showed that very long
# segments can be false positives in two-way / ambiguous regions.
# Keep short, sustained candidates and reject suspiciously long ones.
WRONG_WAY_MIN_SEC = 1.5
WRONG_WAY_MAX_SEC = 8.5


# ============================================================
# Determinism
# ============================================================

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# ============================================================
# Lazy models
# ============================================================

_EVENT_MODEL: YOLO | None = None
_RISK_MODEL: YOLO | None = None


def _device():
    return 0 if torch.cuda.is_available() else "cpu"


def _load_event_model() -> YOLO:
    global _EVENT_MODEL

    if _EVENT_MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {MODEL_PATH}"
            )

        _EVENT_MODEL = YOLO(
            str(MODEL_PATH)
        )

    return _EVENT_MODEL


def _load_risk_model() -> YOLO:
    global _RISK_MODEL

    if _RISK_MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {MODEL_PATH}"
            )

        _RISK_MODEL = YOLO(
            str(MODEL_PATH)
        )

    return _RISK_MODEL


def _reset_tracker(model: YOLO) -> None:
    """
    Ultralytics stores ByteTrack state on the predictor.

    Resetting predictor between videos prevents track IDs / tracker history
    from leaking from one video into the next while keeping model weights
    loaded.
    """
    try:
        model.predictor = None
    except Exception:
        pass


# ============================================================
# Scene config selection
# ============================================================

def _camera_id(
    value: str,
) -> str | None:
    match = re.search(
        r"(C\d{4,})",
        Path(value).stem,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(
        1
    ).upper()


def _choose_config(
    video_id_or_path: str,
) -> Path:
    """
    Local sample C3902 has a small camera-framing shift and therefore uses
    the automatically aligned config.

    Hidden videos have unknown filenames, so they fall back to the base
    fixed-camera scene config.
    """
    camera = _camera_id(
        video_id_or_path
    )

    if (
        camera == "C3902"
        and C3902_CONFIG_PATH.exists()
    ):
        return C3902_CONFIG_PATH

    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Scene config not found: {BASE_CONFIG_PATH}"
        )

    return BASE_CONFIG_PATH


# ============================================================
# Generic helpers
# ============================================================

def _near_frame_edge(
    box,
    width: int,
    height: int,
    margin_ratio: float = 0.01,
) -> bool:
    x1, y1, x2, y2 = map(
        float,
        box,
    )

    margin = (
        min(
            width,
            height,
        )
        * float(
            margin_ratio
        )
    )

    return (
        x1 <= margin
        or y1 <= margin
        or x2 >= width - margin
        or y2 >= height - margin
    )


def _sampling_stride(
    fps: float,
    target_fps: float,
) -> int:
    if fps <= 0:
        return 1

    return max(
        1,
        int(
            round(
                fps
                / target_fps
            )
        ),
    )


def _tracked_objects(
    model: YOLO,
    frame: np.ndarray,
    width: int,
    height: int,
) -> list[dict]:
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=TRACK_CLASSES,
        conf=DETECTION_CONF,
        device=_device(),
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    objects: list[dict] = []

    if boxes.id is None:
        return objects

    xyxy = (
        boxes.xyxy
        .cpu()
        .numpy()
    )

    ids = (
        boxes.id
        .int()
        .cpu()
        .tolist()
    )

    classes = (
        boxes.cls
        .int()
        .cpu()
        .tolist()
    )

    confidences = (
        boxes.conf
        .cpu()
        .tolist()
    )

    for (
        box,
        track_id,
        cls_id,
        confidence,
    ) in zip(
        xyxy,
        ids,
        classes,
        confidences,
    ):
        if _near_frame_edge(
            box,
            width,
            height,
        ):
            continue

        x1, y1, x2, y2 = map(
            float,
            box,
        )

        point = (
            int(
                (
                    x1
                    + x2
                )
                / 2.0
            ),
            int(
                y2
            ),
        )

        objects.append(
            {
                "id": int(
                    track_id
                ),
                "class_id": int(
                    cls_id
                ),
                "confidence": float(
                    confidence
                ),
                "box": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                "point": point,
            }
        )

    return objects


# ============================================================
# Event post-processing
# ============================================================

def _event_padding(
    label: str,
) -> tuple[float, float]:
    """
    Our collision rule identifies the contact / strongest interaction.
    Official accident boundaries include some aftermath, so extend the end.

    Near-miss rule tends to trigger after conflict begins, so pad slightly
    on both sides.
    """
    if label == "accident":
        return (
            0.10,
            2.00,
        )

    if label == "near_miss":
        return (
            0.30,
            0.45,
        )

    if label == "failure_to_yield":
        return (
            0.15,
            0.25,
        )

    return (
        0.0,
        0.0,
    )


def _minimum_duration(
    label: str,
) -> float:
    return {
        "accident": 0.35,
        "near_miss": 0.60,
        "wrong_way": WRONG_WAY_MIN_SEC,
        "stopped_vehicle": 9.0,
        "jaywalking": 1.0,
        "failure_to_yield": 0.40,
        "congestion": 4.0,
    }.get(
        label,
        0.40,
    )


def _merge_gap(
    label: str,
) -> float:
    return {
        "accident": 1.0,
        "near_miss": 0.8,
        "wrong_way": 0.5,
        "stopped_vehicle": 1.2,
        "jaywalking": 0.7,
        "failure_to_yield": 0.5,
        "congestion": 1.5,
    }.get(
        label,
        0.5,
    )


def _postprocess_events(
    raw_events,
    duration: float,
) -> list[list]:
    grouped: dict[
        str,
        list[
            tuple[
                float,
                float,
            ]
        ],
    ] = {}

    for event in raw_events:
        if (
            not isinstance(
                event,
                (
                    list,
                    tuple,
                ),
            )
            or len(event) != 3
        ):
            continue

        start, end, label = event

        label = str(
            label
        )

        if label not in CLASSES:
            continue

        try:
            start = float(
                start
            )
            end = float(
                end
            )
        except Exception:
            continue

        if not (
            math.isfinite(
                start
            )
            and math.isfinite(
                end
            )
        ):
            continue

        if end <= start:
            continue

        pre_pad, post_pad = (
            _event_padding(
                label
            )
        )

        start = max(
            0.0,
            start
            - pre_pad,
        )

        end = min(
            float(
                duration
            ),
            end
            + post_pad,
        )

        event_duration = (
            end
            - start
        )

        if (
            label == "wrong_way"
            and event_duration
            > WRONG_WAY_MAX_SEC
        ):
            # Precision-first rejection of the long false positives
            # observed during sample validation.
            continue

        if (
            event_duration
            < _minimum_duration(
                label
            )
        ):
            continue

        grouped.setdefault(
            label,
            [],
        ).append(
            (
                start,
                end,
            )
        )

    output: list[list] = []

    for (
        label,
        segments,
    ) in grouped.items():
        segments.sort(
            key=lambda item: item[0]
        )

        merged: list[
            list[
                float
            ]
        ] = []

        gap = _merge_gap(
            label
        )

        for (
            start,
            end,
        ) in segments:
            if not merged:
                merged.append(
                    [
                        start,
                        end,
                    ]
                )
                continue

            previous = merged[
                -1
            ]

            if (
                start
                <= previous[1]
                + gap
            ):
                previous[
                    1
                ] = max(
                    previous[
                        1
                    ],
                    end,
                )
            else:
                merged.append(
                    [
                        start,
                        end,
                    ]
                )

        for (
            start,
            end,
        ) in merged:
            if (
                end
                <= start
            ):
                continue

            output.append(
                [
                    round(
                        float(
                            start
                        ),
                        3,
                    ),
                    round(
                        float(
                            end
                        ),
                        3,
                    ),
                    label,
                ]
            )

    output.sort(
        key=lambda event: (
            event[0],
            event[2],
        )
    )

    return output


# ============================================================
# Part A
# ============================================================

def detect_events(
    video_path: str,
) -> list[list]:
    """
    Detect traffic events in one video.

    Uses YOLO11n + ByteTrack, the mapped fixed-camera scene geometry,
    rule-based event detectors, and the precision-tuned collision / near-miss
    trajectory engine.
    """
    path = Path(
        video_path
    )

    if not path.exists():
        raise FileNotFoundError(
            video_path
        )

    cap = cv2.VideoCapture(
        str(
            path
        )
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 25.0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    n_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        n_frames
        / fps
        if fps > 0
        else 0.0
    )

    config_path = _choose_config(
        path.name
    )

    scene = SceneConfig(
        config_path,
        width,
        height,
    )

    jaywalking = JaywalkingDetector(
        scene=scene,
        frame_width=width,
        frame_height=height,
        min_persistence_sec=1.2,
        missing_timeout_sec=0.9,
    )

    stopped = StoppedVehicleDetector(
        scene=scene,
        frame_width=width,
        frame_height=height,
        stationary_sec=10.0,
        missing_timeout_sec=1.2,
    )

    congestion = CongestionDetector(
        scene=scene,
        frame_width=width,
        frame_height=height,
        min_vehicles=6,
        min_persistence_sec=5.0,
    )

    failure_to_yield = FailureToYieldDetector(
        scene=scene,
        frame_width=width,
        frame_height=height,
        min_persistence_sec=0.30,
    )

    wrong_way = WrongWayDetector(
        scene=scene,
        frame_width=width,
        frame_height=height,
        warmup_sec=20.0,
        min_persistence_sec=1.5,
    )

    collision = CollisionRiskEngine(
        scene=scene,
        frame_width=width,
        frame_height=height,
        near_miss_horizon_sec=1.55,
        near_miss_persistence_sec=0.60,
        accident_persistence_sec=0.35,
    )

    model = _load_event_model()
    _reset_tracker(
        model
    )

    stride = _sampling_stride(
        fps,
        TARGET_EVENT_FPS,
    )

    frame_idx = 0
    last_t = 0.0

    try:
        while True:
            ok = cap.grab()

            if not ok:
                break

            process_frame = (
                frame_idx % stride == 0
                or (
                    n_frames > 0
                    and frame_idx
                    == n_frames - 1
                )
            )

            if not process_frame:
                frame_idx += 1
                continue

            ok, frame = cap.retrieve()

            if not ok:
                break

            t = (
                frame_idx
                / fps
                if fps > 0
                else 0.0
            )

            last_t = t

            objects = _tracked_objects(
                model,
                frame,
                width,
                height,
            )

            persons = []
            vehicles = []

            seen_person_ids = set()
            seen_vehicle_ids = set()

            for obj in objects:
                track_id = int(
                    obj[
                        "id"
                    ]
                )

                cls_id = int(
                    obj[
                        "class_id"
                    ]
                )

                point = obj[
                    "point"
                ]

                if (
                    cls_id
                    == PERSON_CLASS
                ):
                    persons.append(
                        obj
                    )

                    seen_person_ids.add(
                        track_id
                    )

                    jaywalking.update(
                        track_id=track_id,
                        t=t,
                        point=point,
                    )

                if (
                    cls_id
                    in MOTOR_VEHICLE_CLASSES
                ):
                    vehicles.append(
                        obj
                    )

                    seen_vehicle_ids.add(
                        track_id
                    )

                    stopped.update(
                        track_id=track_id,
                        t=t,
                        point=point,
                    )

            jaywalking.handle_missing(
                current_t=t,
                seen_ids=seen_person_ids,
            )

            stopped.handle_missing(
                current_t=t,
                seen_ids=seen_vehicle_ids,
            )

            congestion.update_frame(
                t,
                vehicles,
            )

            failure_to_yield.update_frame(
                t,
                persons,
                vehicles,
            )

            wrong_way.update_frame(
                t,
                vehicles,
            )

            collision.update(
                t,
                objects,
            )

            frame_idx += 1

    finally:
        cap.release()

    # Use actual metadata duration when available; otherwise the last timestamp.
    final_t = (
        duration
        if duration > 0
        else last_t
    )

    jaywalking.finish(
        final_t
    )

    stopped.finish(
        final_t
    )

    congestion.finish(
        final_t
    )

    failure_to_yield.finish(
        final_t
    )

    wrong_way.finish(
        final_t
    )

    collision.finish(
        final_t
    )

    raw_events = (
        jaywalking.events
        + stopped.events
        + congestion.events
        + failure_to_yield.events
        + wrong_way.events
        + collision.events
    )

    return _postprocess_events(
        raw_events,
        final_t,
    )


# ============================================================
# Part B
# ============================================================

class RiskEstimator:
    """
    Causal accident anticipation.

    The organizer calls step() on every frame. To stay comfortably inside
    the time limit, YOLO/ByteTrack is run only at TARGET_RISK_FPS and the
    latest risk is returned for skipped source frames.

    No video file is opened here and no Part-A output is reused.
    """

    def __init__(
        self,
    ):
        self.meta = None
        self.scene = None
        self.engine = None
        self.model = None

        self.frame_index = 0
        self.stride = 1

        self.last_score = 0.0
        self.last_processed_t = None

    def reset(
        self,
        meta: dict,
    ) -> None:
        self.meta = dict(
            meta
        )

        fps = float(
            self.meta.get(
                "fps",
                25.0,
            )
            or 25.0
        )

        width = int(
            self.meta.get(
                "width",
                0,
            )
        )

        height = int(
            self.meta.get(
                "height",
                0,
            )
        )

        video_id = str(
            self.meta.get(
                "video_id",
                "",
            )
        )

        if (
            width <= 0
            or height <= 0
        ):
            raise ValueError(
                "RiskEstimator.reset received invalid width/height"
            )

        config_path = _choose_config(
            video_id
        )

        self.scene = SceneConfig(
            config_path,
            width,
            height,
        )

        # Wider horizon than Part A. The continuous score should rise before
        # contact, because Part B asks about an accident starting within 5 s.
        self.engine = CollisionRiskEngine(
            scene=self.scene,
            frame_width=width,
            frame_height=height,
            near_miss_horizon_sec=4.5,
            near_miss_persistence_sec=0.45,
            accident_persistence_sec=0.30,
        )

        self.model = _load_risk_model()

        _reset_tracker(
            self.model
        )

        self.frame_index = 0

        self.stride = _sampling_stride(
            fps,
            TARGET_RISK_FPS,
        )

        self.last_score = 0.0
        self.last_processed_t = None

    def step(
        self,
        frame: np.ndarray,
        t_sec: float,
    ) -> float:
        if (
            self.meta is None
            or self.scene is None
            or self.engine is None
            or self.model is None
        ):
            raise RuntimeError(
                "RiskEstimator.reset(meta) must be called before step()"
            )

        should_process = (
            self.frame_index
            % self.stride
            == 0
        )

        self.frame_index += 1

        if not should_process:
            return float(
                max(
                    0.0,
                    min(
                        1.0,
                        self.last_score,
                    ),
                )
            )

        width = int(
            self.meta[
                "width"
            ]
        )

        height = int(
            self.meta[
                "height"
            ]
        )

        objects = _tracked_objects(
            self.model,
            frame,
            width,
            height,
        )

        state = self.engine.update(
            float(
                t_sec
            ),
            objects,
        )

        score = float(
            state[
                "risk"
            ]
        )

        # The event engine was built for current conflict severity. For
        # anticipation, preserve emerging conflict scores but prevent tiny
        # tracker noise from creating a non-zero baseline everywhere.
        if score < 0.15:
            score = 0.0

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        self.last_score = score

        self.last_processed_t = float(
            t_sec
        )

        return float(
            score
        )