from __future__ import annotations

from statistics import median


class CongestionDetector:
    """
    Conservative congestion detector.

    A frame is considered congested when:
      - enough vehicles are on the roadway OUTSIDE the normal queue zone
      - most measurable vehicles are moving slowly
      - the condition persists for several seconds

    Speeds are normalized by the image diagonal, so the thresholds scale
    reasonably between 720p and 1080p development videos.
    """

    def __init__(
        self,
        scene,
        frame_width: int,
        frame_height: int,
        min_vehicles: int = 6,
        slow_ratio: float = 0.70,
        slow_speed_norm_per_sec: float = 0.018,
        median_speed_norm_per_sec: float = 0.014,
        min_persistence_sec: float = 5.0,
    ):
        self.scene = scene

        self.min_vehicles = int(min_vehicles)
        self.slow_ratio = float(slow_ratio)
        self.slow_speed_norm_per_sec = float(
            slow_speed_norm_per_sec
        )
        self.median_speed_norm_per_sec = float(
            median_speed_norm_per_sec
        )
        self.min_persistence_sec = float(
            min_persistence_sec
        )

        self.diagonal = (
            frame_width ** 2
            + frame_height ** 2
        ) ** 0.5

        self.previous = {}

        self.candidate_start = None
        self.active = False
        self.start = None

        self.events = []

        self.last_vehicle_count = 0
        self.last_measured_count = 0
        self.last_slow_ratio = 0.0
        self.last_median_speed = 0.0

    def _speed(
        self,
        track_id: int,
        t: float,
        point: tuple[int, int],
    ):
        previous = self.previous.get(track_id)

        self.previous[track_id] = (
            float(t),
            point,
        )

        if previous is None:
            return None

        old_t, old_point = previous

        dt = float(t) - old_t

        if dt <= 0.0 or dt > 1.5:
            return None

        dx = float(point[0] - old_point[0])
        dy = float(point[1] - old_point[1])

        pixels_per_sec = (
            (dx * dx + dy * dy) ** 0.5
        ) / dt

        return pixels_per_sec / self.diagonal

    def update_frame(
        self,
        t: float,
        vehicles: list[dict],
    ) -> bool:
        speeds = []
        eligible_count = 0
        seen_ids = set()

        for vehicle in vehicles:
            track_id = int(vehicle["id"])
            point = vehicle["point"]

            seen_ids.add(track_id)

            speed = self._speed(
                track_id,
                t,
                point,
            )

            if not self.scene.on_road(point):
                continue

            # Ordinary traffic-light queue should not become "congestion".
            if self.scene.in_queue_zone(point):
                continue

            eligible_count += 1

            if speed is not None:
                speeds.append(speed)

        # Remove stale speed history.
        for track_id, (last_t, _) in list(
            self.previous.items()
        ):
            if (
                track_id not in seen_ids
                and float(t) - last_t > 2.0
            ):
                del self.previous[track_id]

        self.last_vehicle_count = eligible_count
        self.last_measured_count = len(speeds)

        if speeds:
            slow_count = sum(
                speed
                <= self.slow_speed_norm_per_sec
                for speed in speeds
            )

            current_slow_ratio = (
                slow_count / len(speeds)
            )

            current_median_speed = median(
                speeds
            )
        else:
            current_slow_ratio = 0.0
            current_median_speed = 0.0

        self.last_slow_ratio = current_slow_ratio
        self.last_median_speed = current_median_speed

        enough_speed_samples = (
            len(speeds)
            >= max(
                3,
                self.min_vehicles // 2,
            )
        )

        congested_now = (
            eligible_count >= self.min_vehicles
            and enough_speed_samples
            and current_slow_ratio >= self.slow_ratio
            and current_median_speed
            <= self.median_speed_norm_per_sec
        )

        if self.active:
            if not congested_now:
                if (
                    self.start is not None
                    and float(t) > self.start
                ):
                    self.events.append(
                        [
                            self.start,
                            float(t),
                            "congestion",
                        ]
                    )

                self.active = False
                self.start = None
                self.candidate_start = None

            return self.active

        if congested_now:
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
                    "congestion",
                ]
            )

        self.active = False
        self.start = None
        self.candidate_start = None

    def is_active(self) -> bool:
        return self.active
