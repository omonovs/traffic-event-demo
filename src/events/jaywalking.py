from __future__ import annotations


class JaywalkingDetector:
    """
    Conservative jaywalking detector.

    Candidate requirements:
      - pedestrian footpoint is on mapped roadway
      - pedestrian is not inside or near a mapped crosswalk
      - condition persists for min_persistence_sec
    """

    def __init__(
        self,
        scene,
        frame_width: int,
        frame_height: int,
        min_persistence_sec: float = 1.2,
        missing_timeout_sec: float = 0.8,
    ):
        self.scene = scene

        self.min_persistence_sec = float(
            min_persistence_sec
        )
        self.missing_timeout_sec = float(
            missing_timeout_sec
        )

        diagonal = (
            frame_width ** 2
            + frame_height ** 2
        ) ** 0.5

        # About 18 px at 1280x720 and 26 px at 1920x1080.
        self.crosswalk_margin_px = diagonal * 0.012

        self.states = {}
        self.events = []

    def _new_state(
        self,
        t: float,
    ):
        return {
            "candidate_start": None,
            "active": False,
            "start": None,
            "last_seen": float(t),
        }

    def update(
        self,
        track_id: int,
        t: float,
        point: tuple[int, int],
    ):
        state = self.states.setdefault(
            track_id,
            self._new_state(t),
        )

        state["last_seen"] = float(t)

        on_road = self.scene.on_road(
            point
        )

        near_crosswalk = self.scene.near_crosswalk(
            point,
            self.crosswalk_margin_px,
        )

        violating = (
            on_road
            and not near_crosswalk
        )

        # Active event
        if state["active"]:
            if not violating:
                if (
                    state["start"] is not None
                    and float(t) > state["start"]
                ):
                    self.events.append(
                        [
                            state["start"],
                            float(t),
                            "jaywalking",
                        ]
                    )

                state["active"] = False
                state["start"] = None
                state["candidate_start"] = None

            return

        # Candidate event
        if violating:
            if state["candidate_start"] is None:
                state["candidate_start"] = float(t)

            duration = (
                float(t)
                - state["candidate_start"]
            )

            if duration >= self.min_persistence_sec:
                state["active"] = True
                state["start"] = state["candidate_start"]

        else:
            state["candidate_start"] = None

    def handle_missing(
        self,
        current_t: float,
        seen_ids: set[int],
    ):
        for track_id, state in list(
            self.states.items()
        ):
            if track_id in seen_ids:
                continue

            missing_for = (
                float(current_t)
                - state["last_seen"]
            )

            if missing_for < self.missing_timeout_sec:
                continue

            if (
                state["active"]
                and state["start"] is not None
            ):
                end = state["last_seen"]

                if end > state["start"]:
                    self.events.append(
                        [
                            state["start"],
                            end,
                            "jaywalking",
                        ]
                    )

            del self.states[track_id]

    def finish(
        self,
        video_end: float,
    ):
        for state in self.states.values():
            if (
                not state["active"]
                or state["start"] is None
            ):
                continue

            end = min(
                float(video_end),
                state["last_seen"],
            )

            if end > state["start"]:
                self.events.append(
                    [
                        state["start"],
                        end,
                        "jaywalking",
                    ]
                )

        self.states.clear()

    def is_active(
        self,
        track_id: int,
    ) -> bool:
        state = self.states.get(
            track_id
        )

        return bool(
            state
            and state["active"]
        )