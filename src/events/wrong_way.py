from __future__ import annotations

import cv2
import numpy as np


class WrongWayDetector:
    """
    Conservative wrong-way detector without lane polygons.

    Strategy:
      1. Learn dominant vehicle direction separately for each road polygon.
      2. Only trust a road region when the learned directions are strongly
         one-sided.
      3. After warmup, flag tracks that move strongly opposite the trusted
         dominant direction for a sustained period.

    Important:
      - If a road polygon contains normal two-way traffic, its direction
        samples tend to cancel out, so this detector intentionally refuses
        to classify that region. This is safer than producing false positives.
    """

    def __init__(
        self,
        scene,
        frame_width: int,
        frame_height: int,
        warmup_sec: float = 20.0,
        min_region_samples: int = 25,
        min_direction_dominance: float = 0.72,
        opposite_cos_threshold: float = -0.72,
        min_speed_norm_per_sec: float = 0.018,
        min_persistence_sec: float = 1.5,
        missing_timeout_sec: float = 1.0,
    ):
        self.scene = scene

        self.diagonal = (
            frame_width ** 2
            + frame_height ** 2
        ) ** 0.5

        self.warmup_sec = float(
            warmup_sec
        )
        self.min_region_samples = int(
            min_region_samples
        )
        self.min_direction_dominance = float(
            min_direction_dominance
        )
        self.opposite_cos_threshold = float(
            opposite_cos_threshold
        )
        self.min_speed = float(
            min_speed_norm_per_sec
        )
        self.min_persistence_sec = float(
            min_persistence_sec
        )
        self.missing_timeout_sec = float(
            missing_timeout_sec
        )

        self.previous = {}
        self.region_vectors = {}
        self.states = {}
        self.events = []

        self.active_ids = set()

    def _region_for_point(
        self,
        point: tuple[int, int],
    ):
        for name, polygon in (
            self.scene
            .polygons(
                "road_polygons"
            )
            .items()
        ):
            if len(polygon) < 3:
                continue

            inside = cv2.pointPolygonTest(
                polygon,
                (
                    float(point[0]),
                    float(point[1]),
                ),
                False,
            )

            if inside >= 0:
                return str(name)

        return None

    def _state(
        self,
        track_id: int,
        t: float,
    ):
        return self.states.setdefault(
            track_id,
            {
                "candidate_start": None,
                "active": False,
                "start": None,
                "last_seen": float(t),
            },
        )

    def _region_model(
        self,
        region: str,
    ):
        vectors = self.region_vectors.get(
            region,
            []
        )

        if (
            len(vectors)
            < self.min_region_samples
        ):
            return None

        arr = np.asarray(
            vectors,
            dtype=float,
        )

        mean = arr.mean(
            axis=0
        )

        dominance = float(
            np.linalg.norm(mean)
        )

        if (
            dominance
            < self.min_direction_dominance
        ):
            return None

        norm = float(
            np.linalg.norm(mean)
        )

        if norm <= 1e-8:
            return None

        direction = (
            mean / norm
        )

        return (
            direction,
            dominance,
            len(vectors),
        )

    def update_frame(
        self,
        t: float,
        vehicles: list[dict],
    ) -> set[int]:
        self.active_ids = set()

        seen_ids = set()

        for vehicle in vehicles:
            track_id = int(
                vehicle["id"]
            )

            point = vehicle["point"]

            seen_ids.add(
                track_id
            )

            state = self._state(
                track_id,
                t,
            )

            state["last_seen"] = float(t)

            current_region = self._region_for_point(
                point
            )

            previous = self.previous.get(
                track_id
            )

            self.previous[
                track_id
            ] = (
                float(t),
                point,
                current_region,
            )

            if previous is None:
                continue

            old_t, old_point, old_region = previous

            dt = float(t) - old_t

            if dt <= 0.0 or dt > 1.5:
                state["candidate_start"] = None
                continue

            # Require track to stay in one mapped road region.
            if (
                current_region is None
                or old_region is None
                or current_region != old_region
            ):
                state["candidate_start"] = None
                continue

            dx = float(
                point[0] - old_point[0]
            )
            dy = float(
                point[1] - old_point[1]
            )

            displacement = (
                dx * dx + dy * dy
            ) ** 0.5

            speed = (
                displacement
                / dt
                / self.diagonal
            )

            if (
                speed < self.min_speed
                or displacement <= 1e-6
            ):
                state["candidate_start"] = None
                continue

            unit = np.array(
                [
                    dx / displacement,
                    dy / displacement,
                ],
                dtype=float,
            )

            # During warmup, learn normal movement.
            if float(t) <= self.warmup_sec:
                self.region_vectors.setdefault(
                    current_region,
                    [],
                ).append(
                    unit
                )

                state["candidate_start"] = None
                continue

            model = self._region_model(
                current_region
            )

            # If a region was not sufficiently learned during warmup,
            # keep gathering evidence until it becomes trustworthy.
            if model is None:
                vectors = self.region_vectors.setdefault(
                    current_region,
                    [],
                )

                if len(vectors) < 120:
                    vectors.append(
                        unit
                    )

                state["candidate_start"] = None
                continue

            dominant_direction, _, _ = model

            cosine = float(
                np.dot(
                    unit,
                    dominant_direction,
                )
            )

            opposite_now = (
                cosine
                <= self.opposite_cos_threshold
            )

            if state["active"]:
                if not opposite_now:
                    if (
                        state["start"] is not None
                        and float(t) > state["start"]
                    ):
                        self.events.append(
                            [
                                state["start"],
                                float(t),
                                "wrong_way",
                            ]
                        )

                    state["active"] = False
                    state["start"] = None
                    state["candidate_start"] = None

                else:
                    self.active_ids.add(
                        track_id
                    )

                continue

            if opposite_now:
                if state["candidate_start"] is None:
                    state["candidate_start"] = float(t)

                if (
                    float(t) - state["candidate_start"]
                    >= self.min_persistence_sec
                ):
                    state["active"] = True
                    state["start"] = state["candidate_start"]
                    self.active_ids.add(
                        track_id
                    )

            else:
                state["candidate_start"] = None

        # Missing-track cleanup.
        for track_id, state in list(
            self.states.items()
        ):
            if track_id in seen_ids:
                continue

            missing_for = (
                float(t)
                - state["last_seen"]
            )

            if (
                missing_for
                < self.missing_timeout_sec
            ):
                continue

            if (
                state["active"]
                and state["start"] is not None
                and state["last_seen"] > state["start"]
            ):
                self.events.append(
                    [
                        state["start"],
                        state["last_seen"],
                        "wrong_way",
                    ]
                )

            del self.states[
                track_id
            ]

            self.previous.pop(
                track_id,
                None,
            )

        return set(
            self.active_ids
        )

    def finish(
        self,
        video_end: float,
    ):
        for state in self.states.values():
            if (
                state["active"]
                and state["start"] is not None
            ):
                end = min(
                    float(video_end),
                    state["last_seen"],
                )

                if end > state["start"]:
                    self.events.append(
                        [
                            state["start"],
                            end,
                            "wrong_way",
                        ]
                    )

        self.states.clear()
        self.previous.clear()
        self.active_ids.clear()

    def get_region_models(self):
        result = {}

        for region in self.region_vectors:
            model = self._region_model(
                region
            )

            if model is None:
                result[region] = {
                    "trusted": False,
                    "samples": len(
                        self.region_vectors[
                            region
                        ]
                    ),
                }
            else:
                direction, dominance, samples = model

                result[region] = {
                    "trusted": True,
                    "samples": int(samples),
                    "dominance": round(
                        float(dominance),
                        3,
                    ),
                    "direction": [
                        round(
                            float(direction[0]),
                            3,
                        ),
                        round(
                            float(direction[1]),
                            3,
                        ),
                    ],
                }

        return result
