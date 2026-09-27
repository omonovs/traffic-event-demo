from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class SceneConfig:
    """
    Loads normalized scene geometry from scene_config.json.

    Coordinates in JSON are normalized:
        x in [0, 1]
        y in [0, 1]
    """

    def __init__(
        self,
        config_path: str | Path,
        width: int,
        height: int,
    ):
        self.config_path = Path(config_path)
        self.width = int(width)
        self.height = int(height)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Scene config not found: {self.config_path}"
            )

        self.data = json.loads(
            self.config_path.read_text()
        )

    # ========================================================
    # Geometry
    # ========================================================

    def _to_pixels(self, points):
        return np.array(
            [
                [
                    int(float(x) * self.width),
                    int(float(y) * self.height),
                ]
                for x, y in points
            ],
            dtype=np.int32,
        )

    def polygons(self, category: str) -> dict[str, np.ndarray]:
        objects = self.data.get(category, {})

        if not isinstance(objects, dict):
            return {}

        result = {}

        for name, points in objects.items():
            if not isinstance(points, list):
                continue

            result[name] = self._to_pixels(points)

        return result

    def contains(
        self,
        category: str,
        point: tuple[int, int],
    ) -> bool:
        """
        True if point lies inside any polygon in category.
        """

        x, y = point

        for polygon in self.polygons(category).values():
            if len(polygon) < 3:
                continue

            result = cv2.pointPolygonTest(
                polygon,
                (float(x), float(y)),
                False,
            )

            if result >= 0:
                return True

        return False

    def distance_to_category(
        self,
        category: str,
        point: tuple[int, int],
    ) -> float:
        """
        Pixel distance from point to nearest polygon.

        Returns:
            0.0 if point is inside/on a polygon
            positive distance if outside
            inf if no valid polygon exists
        """

        x, y = point
        best = float("inf")

        for polygon in self.polygons(category).values():
            if len(polygon) < 3:
                continue

            signed_distance = cv2.pointPolygonTest(
                polygon,
                (float(x), float(y)),
                True,
            )

            if signed_distance >= 0:
                return 0.0

            best = min(
                best,
                abs(float(signed_distance)),
            )

        return best

    # ========================================================
    # Semantic helpers
    # ========================================================

    def on_road(
        self,
        point: tuple[int, int],
    ) -> bool:
        return self.contains(
            "road_polygons",
            point,
        )

    def on_crosswalk(
        self,
        point: tuple[int, int],
    ) -> bool:
        return self.contains(
            "crosswalks",
            point,
        )

    def near_crosswalk(
        self,
        point: tuple[int, int],
        margin_px: float,
    ) -> bool:
        return (
            self.distance_to_category(
                "crosswalks",
                point,
            )
            <= float(margin_px)
        )

    def in_queue_zone(
        self,
        point: tuple[int, int],
    ) -> bool:
        return self.contains(
            "queue_zones",
            point,
        )

    # ========================================================
    # Visualization
    # ========================================================

    def draw(self, frame):
        colors = {
            "road_polygons": (0, 0, 255),
            "crosswalks": (255, 255, 0),
            "queue_zones": (0, 255, 255),
            "stop_lines": (255, 0, 255),
            "traffic_lights": (0, 255, 0),
            "lane_polygons": (255, 255, 255),
        }

        for category, color in colors.items():
            for name, polygon in self.polygons(category).items():
                if len(polygon) < 2:
                    continue

                if category == "stop_lines":
                    cv2.polylines(
                        frame,
                        [polygon],
                        False,
                        color,
                        2,
                    )
                else:
                    cv2.polylines(
                        frame,
                        [polygon],
                        True,
                        color,
                        2,
                    )

                x, y = polygon[0]

                cv2.putText(
                    frame,
                    str(name),
                    (
                        int(x),
                        max(15, int(y) - 5),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return frame