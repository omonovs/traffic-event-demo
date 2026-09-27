from __future__ import annotations

from collections import deque

import numpy as np


class StoppedVehicleDetector:
    """
    Conservative stopped-vehicle detector.

    Requirements:
      - vehicle is on mapped roadway
      - vehicle is outside normal queue zones
      - track previously showed meaningful movement
      - vehicle remains nearly stationary for stationary_sec
    """

    def __init__(
        self,
        scene,
        frame_width: int,
        frame_height: int,
        stationary_sec: float = 10.0,
        missing_timeout_sec: float = 1.0,
    ):
        self.scene = scene

        self.stationary_sec = float(
            stationary_sec
        )
        self.missing_timeout_sec = float(
            missing_timeout_sec
        )

        diagonal = (
            frame_width ** 2
            + frame_height ** 2
        ) ** 0.5

        self.stationary_radius = (
            diagonal * 0.008
        )

        self.meaningful_movement = (
            diagonal * 0.025
        )

        self.resume_radius = (
            diagonal * 0.018
        )

        self.states = {}
        self.events = []

    def _new_state(self):
        return {
            "history": deque(),

            "first_point": None,
            "max_displacement": 0.0,
            "has_moved": False,

            "active": False,
            "start": None,
            "anchor": None,

            "last_seen": 0.0,
        }

    def _reset_stop_state(
        self,
        state,
    ):
        state["history"].clear()
        state["active"] = False
        state["start"] = None
        state["anchor"] = None

    def _close_active_event(
        self,
        state,
        end_t: float,
    ):
        if (
            state["active"]
            and state["start"] is not None
            and float(end_t) > state["start"]
        ):
            self.events.append(
                [
                    state["start"],
                    float(end_t),
                    "stopped_vehicle",
                ]
            )

        self._reset_stop_state(
            state
        )

    def update(
        self,
        track_id: int,
        t: float,
        point: tuple[int, int],
    ):
        state = self.states.setdefault(
            track_id,
            self._new_state(),
        )

        state["last_seen"] = float(t)

        current = np.array(
            point,
            dtype=float,
        )

        # Learn whether the track has actually moved.
        if state["first_point"] is None:
            state["first_point"] = current.copy()

        displacement = np.linalg.norm(
            current
            - state["first_point"]
        )

        state["max_displacement"] = max(
            state["max_displacement"],
            float(displacement),
        )

        if (
            state["max_displacement"]
            >= self.meaningful_movement
        ):
            state["has_moved"] = True

        on_road = self.scene.on_road(
            point
        )

        in_queue = self.scene.in_queue_zone(
            point
        )

        # Not eligible as stopped vehicle here.
        if not on_road or in_queue:
            if state["active"]:
                self._close_active_event(
                    state,
                    t,
                )
            else:
                state["history"].clear()

            return

        # Active stopped event.
        if state["active"]:
            anchor = np.array(
                state["anchor"],
                dtype=float,
            )

            movement = np.linalg.norm(
                current - anchor
            )

            if movement > self.resume_radius:
                self._close_active_event(
                    state,
                    t,
                )

            return

        # Ignore objects that have never demonstrated motion.
        if not state["has_moved"]:
            state["history"].clear()
            return

        history = state["history"]

        history.append(
            (
                float(t),
                int(point[0]),
                int(point[1]),
            )
        )

        while (
            history
            and float(t) - history[0][0]
            > self.stationary_sec
        ):
            history.popleft()

        if len(history) < 2:
            return

        window_duration = (
            history[-1][0]
            - history[0][0]
        )

        if (
            window_duration
            < self.stationary_sec * 0.95
        ):
            return

        positions = np.array(
            [
                [x, y]
                for _, x, y in history
            ],
            dtype=float,
        )

        center = np.median(
            positions,
            axis=0,
        )

        distances = np.linalg.norm(
            positions - center,
            axis=1,
        )

        max_distance = float(
            distances.max()
        )

        if (
            max_distance
            <= self.stationary_radius
        ):
            state["active"] = True
            state["start"] = history[0][0]
            state["anchor"] = (
                float(center[0]),
                float(center[1]),
            )

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

            if state["active"]:
                self._close_active_event(
                    state,
                    state["last_seen"],
                )

            del self.states[track_id]

    def finish(
        self,
        video_end: float,
    ):
        for state in self.states.values():
            if not state["active"]:
                continue

            end = min(
                float(video_end),
                state["last_seen"],
            )

            self._close_active_event(
                state,
                end,
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