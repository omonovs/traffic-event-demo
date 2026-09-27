from __future__ import annotations

import cv2


class FailureToYieldDetector:
    """
    Conservative pedestrian-crosswalk conflict detector.

    A candidate requires:
      - a pedestrian on/very near a mapped crosswalk
      - a moving vehicle on/very near THE SAME crosswalk
      - the pedestrian and vehicle are spatially close
      - the conflict persists briefly

    This is an MVP heuristic for "failure_to_yield", not a semantic
    understanding model. It intentionally requires a close conflict to
    reduce false positives.
    """

    def __init__(
        self,
        scene,
        frame_width: int,
        frame_height: int,
        min_vehicle_speed_norm_per_sec: float = 0.012,
        conflict_distance_ratio: float = 0.075,
        pedestrian_margin_ratio: float = 0.008,
        vehicle_margin_ratio: float = 0.010,
        min_persistence_sec: float = 0.30,
    ):
        self.scene = scene

        self.diagonal = (
            frame_width ** 2
            + frame_height ** 2
        ) ** 0.5

        self.min_vehicle_speed = float(
            min_vehicle_speed_norm_per_sec
        )

        self.conflict_distance = (
            self.diagonal
            * float(conflict_distance_ratio)
        )

        self.pedestrian_margin = (
            self.diagonal
            * float(pedestrian_margin_ratio)
        )

        self.vehicle_margin = (
            self.diagonal
            * float(vehicle_margin_ratio)
        )

        self.min_persistence_sec = float(
            min_persistence_sec
        )

        self.vehicle_previous = {}

        self.candidate_start = None
        self.active = False
        self.start = None

        self.events = []

        self.last_conflict_pair = None

    @staticmethod
    def _distance(
        p1: tuple[int, int],
        p2: tuple[int, int],
    ) -> float:
        dx = float(p1[0] - p2[0])
        dy = float(p1[1] - p2[1])

        return (
            dx * dx + dy * dy
        ) ** 0.5

    @staticmethod
    def _signed_distance_to_polygon(
        polygon,
        point: tuple[int, int],
    ) -> float:
        return float(
            cv2.pointPolygonTest(
                polygon,
                (
                    float(point[0]),
                    float(point[1]),
                ),
                True,
            )
        )

    def _near_polygon(
        self,
        polygon,
        point: tuple[int, int],
        margin: float,
    ) -> bool:
        signed_distance = (
            self._signed_distance_to_polygon(
                polygon,
                point,
            )
        )

        return (
            signed_distance >= 0
            or abs(signed_distance) <= margin
        )

    def _vehicle_speed(
        self,
        track_id: int,
        t: float,
        point: tuple[int, int],
    ):
        previous = self.vehicle_previous.get(
            track_id
        )

        self.vehicle_previous[
            track_id
        ] = (
            float(t),
            point,
        )

        if previous is None:
            return None

        old_t, old_point = previous

        dt = float(t) - old_t

        if dt <= 0.0 or dt > 1.5:
            return None

        dx = float(
            point[0] - old_point[0]
        )
        dy = float(
            point[1] - old_point[1]
        )

        pixels_per_sec = (
            dx * dx + dy * dy
        ) ** 0.5 / dt

        return (
            pixels_per_sec
            / self.diagonal
        )

    def update_frame(
        self,
        t: float,
        persons: list[dict],
        vehicles: list[dict],
    ) -> bool:
        crosswalks = self.scene.polygons(
            "crosswalks"
        )

        vehicle_speeds = {}

        seen_vehicle_ids = set()

        for vehicle in vehicles:
            track_id = int(
                vehicle["id"]
            )

            seen_vehicle_ids.add(
                track_id
            )

            vehicle_speeds[
                track_id
            ] = self._vehicle_speed(
                track_id,
                t,
                vehicle["point"],
            )

        for track_id, (last_t, _) in list(
            self.vehicle_previous.items()
        ):
            if (
                track_id not in seen_vehicle_ids
                and float(t) - last_t > 2.0
            ):
                del self.vehicle_previous[
                    track_id
                ]

        conflict = False
        conflict_pair = None

        for crosswalk_name, polygon in crosswalks.items():
            if len(polygon) < 3:
                continue

            people_here = [
                person
                for person in persons
                if self._near_polygon(
                    polygon,
                    person["point"],
                    self.pedestrian_margin,
                )
            ]

            if not people_here:
                continue

            vehicles_here = [
                vehicle
                for vehicle in vehicles
                if self._near_polygon(
                    polygon,
                    vehicle["point"],
                    self.vehicle_margin,
                )
            ]

            if not vehicles_here:
                continue

            for person in people_here:
                for vehicle in vehicles_here:
                    speed = vehicle_speeds.get(
                        int(vehicle["id"])
                    )

                    if (
                        speed is None
                        or speed < self.min_vehicle_speed
                    ):
                        continue

                    distance = self._distance(
                        person["point"],
                        vehicle["point"],
                    )

                    if (
                        distance
                        <= self.conflict_distance
                    ):
                        conflict = True
                        conflict_pair = (
                            int(person["id"]),
                            int(vehicle["id"]),
                            str(crosswalk_name),
                        )
                        break

                if conflict:
                    break

            if conflict:
                break

        self.last_conflict_pair = (
            conflict_pair
        )

        if self.active:
            if not conflict:
                if (
                    self.start is not None
                    and float(t) > self.start
                ):
                    self.events.append(
                        [
                            self.start,
                            float(t),
                            "failure_to_yield",
                        ]
                    )

                self.active = False
                self.start = None
                self.candidate_start = None

            return self.active

        if conflict:
            if self.candidate_start is None:
                self.candidate_start = float(t)

            if (
                float(t) - self.candidate_start
                >= self.min_persistence_sec
            ):
                self.active = True
                self.start = self.candidate_start

        else:
            self.candidate_start = None

        return self.active

    def finish(
        self,
        video_end: float,
    ):
        if (
            self.active
            and self.start is not None
            and float(video_end) > self.start
        ):
            self.events.append(
                [
                    self.start,
                    float(video_end),
                    "failure_to_yield",
                ]
            )

        self.active = False
        self.start = None
        self.candidate_start = None

    def is_active(self) -> bool:
        return self.active
