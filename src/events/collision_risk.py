from __future__ import annotations

from collections import deque
from itertools import combinations
from math import acos, degrees
from typing import Iterable

import numpy as np


MOTOR_VEHICLE_CLASSES = {
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
}

ROAD_USER_CLASSES = {
    0,  # person
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
}


def _clip01(value: float) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def _box_iou(
    box_a,
    box_b,
) -> float:
    ax1, ay1, ax2, ay2 = map(
        float,
        box_a,
    )
    bx1, by1, bx2, by2 = map(
        float,
        box_b,
    )

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )
    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 1e-9:
        return 0.0

    return float(
        intersection / union
    )


def _box_height(box) -> float:
    _, y1, _, y2 = map(
        float,
        box,
    )

    return max(
        1.0,
        y2 - y1,
    )


def _box_width(box) -> float:
    x1, _, x2, _ = map(
        float,
        box,
    )

    return max(
        1.0,
        x2 - x1,
    )


class CollisionRiskEngine:
    """
    Precision-first image-plane engine for:

        - near_miss
        - accident
        - continuous risk score [0, 1]

    Compared with the first Stage-5 version, this version is deliberately
    stricter. It requires several signals to agree before an event becomes
    active.

    NEAR MISS requires:
        - at least one moving motor vehicle
        - both objects in relevant road/crosswalk geometry
        - meaningful closing motion
        - short time to closest approach
        - small predicted miss distance
        - temporal persistence
        - no collision-like overlap

    ACCIDENT requires:
        - strong spatial contact / overlap
        - substantial relative motion
        - AND an impact signature:
            sudden speed loss, acceleration impulse, or heading change
        - temporal persistence

    IMPORTANT:
        This is an image-plane heuristic. It does not estimate true metric
        distance in meters. Thresholds are normalized by image size and
        apparent object size to reduce perspective sensitivity.
    """

    def __init__(
        self,
        scene,
        frame_width: int,
        frame_height: int,
        near_miss_horizon_sec: float = 1.55,
        near_miss_persistence_sec: float = 0.60,
        accident_persistence_sec: float = 0.35,
        pair_timeout_sec: float = 0.85,
        history_sec: float = 1.60,
    ):
        self.scene = scene

        self.width = int(
            frame_width
        )
        self.height = int(
            frame_height
        )

        self.diagonal = float(
            np.hypot(
                self.width,
                self.height,
            )
        )

        self.near_miss_horizon_sec = float(
            near_miss_horizon_sec
        )
        self.near_miss_persistence_sec = float(
            near_miss_persistence_sec
        )
        self.accident_persistence_sec = float(
            accident_persistence_sec
        )
        self.pair_timeout_sec = float(
            pair_timeout_sec
        )
        self.history_sec = float(
            history_sec
        )

        # track_id -> deque[(t, point)]
        self.track_history = {}

        # track_id -> {
        #     "t": ...,
        #     "velocity": np.ndarray,
        #     "speed_norm": ...
        # }
        self.previous_motion = {}

        # pair_key -> event state
        self.pair_states = {}

        self.events = []

        self.frame_risk = 0.0
        self.smoothed_risk = 0.0

        self.top_pair = None
        self.top_pair_metrics = None

        self.active_near_miss_pairs = set()
        self.active_accident_pairs = set()

    # ========================================================
    # Track motion
    # ========================================================

    def _history_for(
        self,
        track_id: int,
    ):
        return self.track_history.setdefault(
            int(track_id),
            deque(),
        )

    def _update_history(
        self,
        obj: dict,
        t: float,
    ):
        track_id = int(
            obj["id"]
        )

        history = self._history_for(
            track_id
        )

        point = np.asarray(
            obj["point"],
            dtype=float,
        )

        history.append(
            (
                float(t),
                point,
            )
        )

        while (
            history
            and float(t) - history[0][0]
            > self.history_sec
        ):
            history.popleft()

    def _robust_velocity(
        self,
        track_id: int,
    ):
        """
        Least-squares velocity over recent trajectory.

        Using several points instead of two makes the estimate much less
        sensitive to YOLO / ByteTrack box jitter.
        """

        history = self.track_history.get(
            int(track_id)
        )

        if (
            history is None
            or len(history) < 3
        ):
            return None

        newest_t = float(
            history[-1][0]
        )

        usable = [
            (
                float(t),
                p,
            )
            for t, p in history
            if (
                newest_t - float(t)
                <= 0.95
            )
        ]

        if len(usable) < 3:
            return None

        times = np.asarray(
            [
                t
                for t, _
                in usable
            ],
            dtype=float,
        )

        points = np.asarray(
            [
                p
                for _, p
                in usable
            ],
            dtype=float,
        )

        span = float(
            times[-1]
            - times[0]
        )

        if span < 0.30:
            return None

        centered_t = (
            times
            - times.mean()
        )

        denominator = float(
            np.dot(
                centered_t,
                centered_t,
            )
        )

        if denominator <= 1e-9:
            return None

        centered_points = (
            points
            - points.mean(
                axis=0
            )
        )

        vx = float(
            np.dot(
                centered_t,
                centered_points[:, 0],
            )
            / denominator
        )

        vy = float(
            np.dot(
                centered_t,
                centered_points[:, 1],
            )
            / denominator
        )

        return np.asarray(
            [
                vx,
                vy,
            ],
            dtype=float,
        )

    def _motion_features(
        self,
        obj: dict,
        t: float,
    ):
        track_id = int(
            obj["id"]
        )

        velocity = self._robust_velocity(
            track_id
        )

        if velocity is None:
            return {
                "velocity": None,
                "speed_norm": 0.0,
                "accel_norm": 0.0,
                "heading_change_deg": 0.0,
                "speed_drop_ratio": 0.0,
            }

        speed_px = float(
            np.linalg.norm(
                velocity
            )
        )

        speed_norm = (
            speed_px
            / self.diagonal
        )

        previous = self.previous_motion.get(
            track_id
        )

        accel_norm = 0.0
        heading_change_deg = 0.0
        speed_drop_ratio = 0.0

        if previous is not None:
            previous_t = float(
                previous["t"]
            )

            previous_velocity = np.asarray(
                previous["velocity"],
                dtype=float,
            )

            previous_speed_norm = float(
                previous["speed_norm"]
            )

            dt = (
                float(t)
                - previous_t
            )

            if (
                0.08 <= dt <= 1.0
            ):
                accel_norm = float(
                    np.linalg.norm(
                        velocity
                        - previous_velocity
                    )
                    / dt
                    / self.diagonal
                )

            previous_speed_px = float(
                np.linalg.norm(
                    previous_velocity
                )
            )

            if (
                previous_speed_px > 1e-6
                and speed_px > 1e-6
            ):
                cosine = float(
                    np.dot(
                        previous_velocity,
                        velocity,
                    )
                    / (
                        previous_speed_px
                        * speed_px
                    )
                )

                cosine = max(
                    -1.0,
                    min(
                        1.0,
                        cosine,
                    ),
                )

                heading_change_deg = degrees(
                    acos(
                        cosine
                    )
                )

            if previous_speed_norm > 1e-5:
                speed_drop_ratio = max(
                    0.0,
                    (
                        previous_speed_norm
                        - speed_norm
                    )
                    / previous_speed_norm,
                )

        self.previous_motion[
            track_id
        ] = {
            "t": float(t),
            "velocity": velocity.copy(),
            "speed_norm": float(
                speed_norm
            ),
        }

        return {
            "velocity": velocity,
            "speed_norm": float(
                speed_norm
            ),
            "accel_norm": float(
                accel_norm
            ),
            "heading_change_deg": float(
                heading_change_deg
            ),
            "speed_drop_ratio": float(
                speed_drop_ratio
            ),
        }

    # ========================================================
    # Scene / pair filtering
    # ========================================================

    def _near_relevant_scene(
        self,
        obj: dict,
    ) -> bool:
        point = obj[
            "point"
        ]

        cls_id = int(
            obj[
                "class_id"
            ]
        )

        margin = (
            self.diagonal
            * 0.014
        )

        if cls_id in MOTOR_VEHICLE_CLASSES:
            return (
                self.scene.on_road(
                    point
                )
                or (
                    self.scene.distance_to_category(
                        "road_polygons",
                        point,
                    )
                    <= margin
                )
            )

        if cls_id in {
            0,
            1,
        }:
            return (
                self.scene.on_road(
                    point
                )
                or self.scene.near_crosswalk(
                    point,
                    margin,
                )
            )

        return False

    @staticmethod
    def _valid_pair(
        obj_a: dict,
        obj_b: dict,
    ) -> bool:
        class_a = int(
            obj_a[
                "class_id"
            ]
        )

        class_b = int(
            obj_b[
                "class_id"
            ]
        )

        if (
            class_a
            not in ROAD_USER_CLASSES
            or class_b
            not in ROAD_USER_CLASSES
        ):
            return False

        # At least one motor vehicle is required.
        return (
            class_a in MOTOR_VEHICLE_CLASSES
            or class_b in MOTOR_VEHICLE_CLASSES
        )

    def _pair_scene_allowed(
        self,
        obj_a: dict,
        obj_b: dict,
    ) -> bool:
        if (
            not self._near_relevant_scene(
                obj_a
            )
            or not self._near_relevant_scene(
                obj_b
            )
        ):
            return False

        class_a = int(
            obj_a[
                "class_id"
            ]
        )

        class_b = int(
            obj_b[
                "class_id"
            ]
        )

        # Two vehicles crawling together inside the mapped normal queue
        # are ordinary signal traffic, not a near miss.
        if (
            class_a in MOTOR_VEHICLE_CLASSES
            and class_b in MOTOR_VEHICLE_CLASSES
            and self.scene.in_queue_zone(
                obj_a[
                    "point"
                ]
            )
            and self.scene.in_queue_zone(
                obj_b[
                    "point"
                ]
            )
        ):
            return False

        return True

    @staticmethod
    def _pair_key(
        obj_a: dict,
        obj_b: dict,
    ):
        return tuple(
            sorted(
                (
                    int(
                        obj_a[
                            "id"
                        ]
                    ),
                    int(
                        obj_b[
                            "id"
                        ]
                    ),
                )
            )
        )

    # ========================================================
    # Pair metrics
    # ========================================================

    def _pair_metrics(
        self,
        obj_a: dict,
        obj_b: dict,
        motion_a: dict,
        motion_b: dict,
    ):
        velocity_a = motion_a[
            "velocity"
        ]

        velocity_b = motion_b[
            "velocity"
        ]

        if (
            velocity_a is None
            or velocity_b is None
        ):
            return None

        point_a = np.asarray(
            obj_a[
                "point"
            ],
            dtype=float,
        )

        point_b = np.asarray(
            obj_b[
                "point"
            ],
            dtype=float,
        )

        relative_position = (
            point_b
            - point_a
        )

        relative_velocity = (
            velocity_b
            - velocity_a
        )

        current_distance_px = float(
            np.linalg.norm(
                relative_position
            )
        )

        relative_speed_px = float(
            np.linalg.norm(
                relative_velocity
            )
        )

        if current_distance_px <= 1e-6:
            closing_speed_px = (
                relative_speed_px
            )
        else:
            closing_speed_px = float(
                -np.dot(
                    relative_position,
                    relative_velocity,
                )
                / current_distance_px
            )

        relative_velocity_sq = float(
            np.dot(
                relative_velocity,
                relative_velocity,
            )
        )

        if (
            relative_velocity_sq
            <= 1e-8
        ):
            tca_sec = float(
                "inf"
            )

            closest_distance_px = (
                current_distance_px
            )

        else:
            tca_sec = float(
                -np.dot(
                    relative_position,
                    relative_velocity,
                )
                / relative_velocity_sq
            )

            if tca_sec < 0.0:
                closest_distance_px = (
                    current_distance_px
                )

            else:
                closest_vector = (
                    relative_position
                    + relative_velocity
                    * tca_sec
                )

                closest_distance_px = float(
                    np.linalg.norm(
                        closest_vector
                    )
                )

        iou = _box_iou(
            obj_a[
                "box"
            ],
            obj_b[
                "box"
            ],
        )

        mean_height = (
            _box_height(
                obj_a[
                    "box"
                ]
            )
            + _box_height(
                obj_b[
                    "box"
                ]
            )
        ) / 2.0

        mean_width = (
            _box_width(
                obj_a[
                    "box"
                ]
            )
            + _box_width(
                obj_b[
                    "box"
                ]
            )
        ) / 2.0

        class_a = int(
            obj_a[
                "class_id"
            ]
        )

        class_b = int(
            obj_b[
                "class_id"
            ]
        )

        pedestrian_pair = (
            class_a == 0
            or class_b == 0
        )

        if pedestrian_pair:
            near_radius_px = max(
                self.diagonal
                * 0.010,
                mean_height
                * 0.42,
            )

            contact_radius_px = max(
                self.diagonal
                * 0.006,
                mean_height
                * 0.22,
            )

        else:
            near_radius_px = max(
                self.diagonal
                * 0.012,
                mean_height
                * 0.34,
                mean_width
                * 0.20,
            )

            contact_radius_px = max(
                self.diagonal
                * 0.007,
                mean_height
                * 0.19,
            )

        return {
            "current_distance_px": float(
                current_distance_px
            ),
            "current_distance_norm": float(
                current_distance_px
                / self.diagonal
            ),

            "relative_speed_norm": float(
                relative_speed_px
                / self.diagonal
            ),

            "closing_speed_norm": float(
                closing_speed_px
                / self.diagonal
            ),

            "tca_sec": float(
                tca_sec
            ),

            "closest_distance_px": float(
                closest_distance_px
            ),

            "closest_distance_norm": float(
                closest_distance_px
                / self.diagonal
            ),

            "near_radius_px": float(
                near_radius_px
            ),

            "contact_radius_px": float(
                contact_radius_px
            ),

            "iou": float(
                iou
            ),

            "speed_a_norm": float(
                motion_a[
                    "speed_norm"
                ]
            ),

            "speed_b_norm": float(
                motion_b[
                    "speed_norm"
                ]
            ),

            "max_speed_norm": float(
                max(
                    motion_a[
                        "speed_norm"
                    ],
                    motion_b[
                        "speed_norm"
                    ],
                )
            ),

            "max_accel_norm": float(
                max(
                    motion_a[
                        "accel_norm"
                    ],
                    motion_b[
                        "accel_norm"
                    ],
                )
            ),

            "max_heading_change_deg": float(
                max(
                    motion_a[
                        "heading_change_deg"
                    ],
                    motion_b[
                        "heading_change_deg"
                    ],
                )
            ),

            "max_speed_drop_ratio": float(
                max(
                    motion_a[
                        "speed_drop_ratio"
                    ],
                    motion_b[
                        "speed_drop_ratio"
                    ],
                )
            ),

            "pedestrian_pair": bool(
                pedestrian_pair
            ),
        }

    # ========================================================
    # Precision-first event conditions
    # ========================================================

    def _near_miss_condition(
        self,
        metrics: dict,
    ) -> bool:
        tca = float(
            metrics[
                "tca_sec"
            ]
        )

        if (
            not np.isfinite(
                tca
            )
            or tca < 0.05
            or tca
            > self.near_miss_horizon_sec
        ):
            return False

        # Someone must actually be moving at meaningful speed.
        if (
            metrics[
                "max_speed_norm"
            ]
            < 0.014
        ):
            return False

        # Strong closing motion.
        if (
            metrics[
                "closing_speed_norm"
            ]
            < 0.018
        ):
            return False

        # Pair must already be reasonably close.
        if (
            metrics[
                "current_distance_norm"
            ]
            > 0.085
        ):
            return False

        # Predicted trajectories must pass close enough.
        if (
            metrics[
                "closest_distance_px"
            ]
            > metrics[
                "near_radius_px"
            ]
        ):
            return False

        # Significant box overlap is more collision-like than near-miss-like.
        if (
            metrics[
                "iou"
            ]
            >= 0.10
        ):
            return False

        return True

    def _accident_condition(
        self,
        metrics: dict,
    ) -> bool:
        iou = float(
            metrics[
                "iou"
            ]
        )

        relative_speed = float(
            metrics[
                "relative_speed_norm"
            ]
        )

        current_distance = float(
            metrics[
                "current_distance_px"
            ]
        )

        contact_radius = float(
            metrics[
                "contact_radius_px"
            ]
        )

        acceleration = float(
            metrics[
                "max_accel_norm"
            ]
        )

        heading_change = float(
            metrics[
                "max_heading_change_deg"
            ]
        )

        speed_drop_ratio = float(
            metrics[
                "max_speed_drop_ratio"
            ]
        )

        impact_signature = (
            acceleration >= 0.060
            or heading_change >= 55.0
            or speed_drop_ratio >= 0.48
        )

        strong_contact = (
            iou >= 0.20
            or (
                current_distance
                <= contact_radius
                and iou >= 0.08
            )
        )

        enough_relative_motion = (
            relative_speed >= 0.030
        )

        # Main collision rule:
        # contact + relative motion + impact response.
        if (
            strong_contact
            and enough_relative_motion
            and impact_signature
        ):
            return True

        # Very strong overlap at high relative speed is accepted even if
        # the motion-response estimate is temporarily noisy.
        if (
            iou >= 0.34
            and relative_speed >= 0.040
        ):
            return True

        return False

    # ========================================================
    # Risk score
    # ========================================================

    def _risk_score(
        self,
        metrics: dict,
        near_condition: bool,
        accident_condition: bool,
    ) -> float:
        tca = float(
            metrics[
                "tca_sec"
            ]
        )

        if (
            np.isfinite(
                tca
            )
            and 0.0 <= tca
            <= self.near_miss_horizon_sec
        ):
            time_risk = (
                1.0
                - tca
                / self.near_miss_horizon_sec
            )
        else:
            time_risk = 0.0

        distance_risk = (
            1.0
            - (
                metrics[
                    "closest_distance_px"
                ]
                / max(
                    metrics[
                        "near_radius_px"
                    ]
                    * 1.8,
                    1.0,
                )
            )
        )

        closing_risk = (
            metrics[
                "closing_speed_norm"
            ]
            / 0.055
        )

        overlap_risk = (
            metrics[
                "iou"
            ]
            / 0.30
        )

        impact_risk = max(
            metrics[
                "max_accel_norm"
            ]
            / 0.070,

            metrics[
                "max_heading_change_deg"
            ]
            / 70.0,

            metrics[
                "max_speed_drop_ratio"
            ]
            / 0.60,
        )

        # Lower baseline than the old Stage-5 version.
        risk = (
            0.34
            * _clip01(
                time_risk
            )
            + 0.29
            * _clip01(
                distance_risk
            )
            + 0.21
            * _clip01(
                closing_risk
            )
            + 0.08
            * _clip01(
                overlap_risk
            )
            + 0.08
            * _clip01(
                impact_risk
            )
        )

        # Ordinary close traffic should not become "high risk" merely from
        # one noisy metric.
        if (
            not near_condition
            and not accident_condition
        ):
            risk = min(
                risk,
                0.64,
            )

        if near_condition:
            risk = max(
                risk,
                0.78,
            )

        if accident_condition:
            risk = max(
                risk,
                0.96,
            )

        return _clip01(
            risk
        )

    # ========================================================
    # Pair-state management
    # ========================================================

    def _pair_state(
        self,
        key,
        t: float,
    ):
        return self.pair_states.setdefault(
            key,
            {
                "near_candidate_start": None,
                "near_active": False,
                "near_start": None,

                "accident_candidate_start": None,
                "accident_active": False,
                "accident_start": None,

                "last_seen": float(
                    t
                ),
            },
        )

    def _close_near_miss(
        self,
        state,
        end_t: float,
    ):
        if (
            state[
                "near_active"
            ]
            and state[
                "near_start"
            ]
            is not None
            and float(
                end_t
            )
            > state[
                "near_start"
            ]
        ):
            self.events.append(
                [
                    state[
                        "near_start"
                    ],
                    float(
                        end_t
                    ),
                    "near_miss",
                ]
            )

        state[
            "near_active"
        ] = False

        state[
            "near_start"
        ] = None

        state[
            "near_candidate_start"
        ] = None

    def _close_accident(
        self,
        state,
        end_t: float,
    ):
        if (
            state[
                "accident_active"
            ]
            and state[
                "accident_start"
            ]
            is not None
            and float(
                end_t
            )
            > state[
                "accident_start"
            ]
        ):
            self.events.append(
                [
                    state[
                        "accident_start"
                    ],
                    float(
                        end_t
                    ),
                    "accident",
                ]
            )

        state[
            "accident_active"
        ] = False

        state[
            "accident_start"
        ] = None

        state[
            "accident_candidate_start"
        ] = None

    # ========================================================
    # Public update
    # ========================================================

    def update(
        self,
        t: float,
        objects: Iterable[dict],
    ) -> dict:
        objects = [
            obj
            for obj in objects
            if int(
                obj[
                    "class_id"
                ]
            )
            in ROAD_USER_CLASSES
        ]

        for obj in objects:
            self._update_history(
                obj,
                t,
            )

        motions = {
            int(
                obj[
                    "id"
                ]
            ): self._motion_features(
                obj,
                t,
            )
            for obj in objects
        }

        self.active_near_miss_pairs = set()
        self.active_accident_pairs = set()

        self.frame_risk = 0.0
        self.top_pair = None
        self.top_pair_metrics = None

        seen_pair_keys = set()

        for (
            obj_a,
            obj_b,
        ) in combinations(
            objects,
            2,
        ):
            if not self._valid_pair(
                obj_a,
                obj_b,
            ):
                continue

            if not self._pair_scene_allowed(
                obj_a,
                obj_b,
            ):
                continue

            key = self._pair_key(
                obj_a,
                obj_b,
            )

            seen_pair_keys.add(
                key
            )

            state = self._pair_state(
                key,
                t,
            )

            state[
                "last_seen"
            ] = float(
                t
            )

            motion_a = motions[
                int(
                    obj_a[
                        "id"
                    ]
                )
            ]

            motion_b = motions[
                int(
                    obj_b[
                        "id"
                    ]
                )
            ]

            metrics = self._pair_metrics(
                obj_a,
                obj_b,
                motion_a,
                motion_b,
            )

            if metrics is None:
                continue

            accident_now = (
                self._accident_condition(
                    metrics
                )
            )

            near_now = (
                self._near_miss_condition(
                    metrics
                )
                and not accident_now
            )

            risk = self._risk_score(
                metrics,
                near_now,
                accident_now,
            )

            if (
                risk
                > self.frame_risk
            ):
                self.frame_risk = (
                    risk
                )

                self.top_pair = key

                self.top_pair_metrics = (
                    metrics
                )

            # -----------------------------------------------
            # Accident state
            # -----------------------------------------------

            if state[
                "accident_active"
            ]:
                self.active_accident_pairs.add(
                    key
                )

                if not accident_now:
                    self._close_accident(
                        state,
                        t,
                    )

            elif accident_now:
                if (
                    state[
                        "accident_candidate_start"
                    ]
                    is None
                ):
                    state[
                        "accident_candidate_start"
                    ] = float(
                        t
                    )

                if (
                    float(
                        t
                    )
                    - state[
                        "accident_candidate_start"
                    ]
                    >= self.accident_persistence_sec
                ):
                    state[
                        "accident_active"
                    ] = True

                    state[
                        "accident_start"
                    ] = state[
                        "accident_candidate_start"
                    ]

                    self.active_accident_pairs.add(
                        key
                    )

                    # Accident supersedes near-miss for the same pair.
                    if state[
                        "near_active"
                    ]:
                        self._close_near_miss(
                            state,
                            t,
                        )

            else:
                state[
                    "accident_candidate_start"
                ] = None

            # -----------------------------------------------
            # Near-miss state
            # -----------------------------------------------

            if (
                state[
                    "accident_active"
                ]
                or accident_now
            ):
                state[
                    "near_candidate_start"
                ] = None

                continue

            if state[
                "near_active"
            ]:
                self.active_near_miss_pairs.add(
                    key
                )

                if not near_now:
                    self._close_near_miss(
                        state,
                        t,
                    )

            elif near_now:
                if (
                    state[
                        "near_candidate_start"
                    ]
                    is None
                ):
                    state[
                        "near_candidate_start"
                    ] = float(
                        t
                    )

                if (
                    float(
                        t
                    )
                    - state[
                        "near_candidate_start"
                    ]
                    >= self.near_miss_persistence_sec
                ):
                    state[
                        "near_active"
                    ] = True

                    state[
                        "near_start"
                    ] = state[
                        "near_candidate_start"
                    ]

                    self.active_near_miss_pairs.add(
                        key
                    )

            else:
                state[
                    "near_candidate_start"
                ] = None

        # ----------------------------------------------------
        # Close stale pair states
        # ----------------------------------------------------

        for (
            key,
            state,
        ) in list(
            self.pair_states.items()
        ):
            if key in seen_pair_keys:
                continue

            missing_for = (
                float(
                    t
                )
                - state[
                    "last_seen"
                ]
            )

            if (
                missing_for
                < self.pair_timeout_sec
            ):
                continue

            if state[
                "near_active"
            ]:
                self._close_near_miss(
                    state,
                    state[
                        "last_seen"
                    ],
                )

            if state[
                "accident_active"
            ]:
                self._close_accident(
                    state,
                    state[
                        "last_seen"
                    ],
                )

            del self.pair_states[
                key
            ]

        # ----------------------------------------------------
        # Risk smoothing
        # ----------------------------------------------------

        if (
            self.frame_risk
            >= self.smoothed_risk
        ):
            alpha = 0.72
        else:
            alpha = 0.28

        self.smoothed_risk = (
            alpha
            * self.frame_risk
            + (
                1.0
                - alpha
            )
            * self.smoothed_risk
        )

        self.smoothed_risk = _clip01(
            self.smoothed_risk
        )

        return {
            "risk": self.smoothed_risk,
            "raw_risk": self.frame_risk,
            "top_pair": self.top_pair,
            "top_pair_metrics": self.top_pair_metrics,
            "active_near_miss_pairs": set(
                self.active_near_miss_pairs
            ),
            "active_accident_pairs": set(
                self.active_accident_pairs
            ),
        }

    # ========================================================
    # Finalization
    # ========================================================

    def finish(
        self,
        video_end: float,
    ):
        for state in self.pair_states.values():
            end = min(
                float(
                    video_end
                ),
                float(
                    state[
                        "last_seen"
                    ]
                ),
            )

            if state[
                "near_active"
            ]:
                self._close_near_miss(
                    state,
                    end,
                )

            if state[
                "accident_active"
            ]:
                self._close_accident(
                    state,
                    end,
                )

        self.pair_states.clear()
        self.track_history.clear()
        self.previous_motion.clear()

        self.active_near_miss_pairs.clear()
        self.active_accident_pairs.clear()

    # ========================================================
    # Diagnostics
    # ========================================================

    def debug_snapshot(
        self,
    ) -> dict:
        metrics = None

        if (
            self.top_pair_metrics
            is not None
        ):
            metrics = {}

            for (
                key,
                value,
            ) in self.top_pair_metrics.items():
                if isinstance(
                    value,
                    (
                        int,
                        float,
                        np.floating,
                    ),
                ):
                    metrics[
                        key
                    ] = round(
                        float(
                            value
                        ),
                        4,
                    )
                else:
                    metrics[
                        key
                    ] = value

        return {
            "frame_risk": round(
                float(
                    self.frame_risk
                ),
                4,
            ),

            "smoothed_risk": round(
                float(
                    self.smoothed_risk
                ),
                4,
            ),

            "top_pair": (
                list(
                    self.top_pair
                )
                if self.top_pair
                is not None
                else None
            ),

            "top_pair_metrics": metrics,

            "active_near_miss_pairs": [
                list(
                    pair
                )
                for pair
                in sorted(
                    self.active_near_miss_pairs
                )
            ],

            "active_accident_pairs": [
                list(
                    pair
                )
                for pair
                in sorted(
                    self.active_accident_pairs
                )
            ],
        }