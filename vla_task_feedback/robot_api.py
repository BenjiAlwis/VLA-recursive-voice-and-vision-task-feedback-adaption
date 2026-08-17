"""
THE CONTRACT.  Owner: Benji.  FROZEN — do not change signatures after T-2:40.

Team A implements these three classes.
Team B only ever imports get_robot() and calls these five methods.

Backend chosen by env var:  ROBOT_BACKEND=temp | mock | go2
"""
import os
import time
import math
import random
from typing import Optional

import numpy as np


class RobotBase:
    """Every backend implements exactly this. Nothing more."""

    def forward(self, meters: float) -> None:
        """Blocking. Walks forward, then STOPS. Returns after motion ends."""
        raise NotImplementedError

    def turn(self, degrees: float) -> None:
        """Blocking. + = left / CCW. Then STOPS."""
        raise NotImplementedError

    def stop(self) -> None:
        """Idempotent. Safe to call any time, from anywhere."""
        raise NotImplementedError

    def get_pose(self) -> dict:
        """{'x_cm', 'y_cm', 'heading_deg', 'source'}
        source is 'aruco' (ground truth) or 'odom' (drifts) or 'sim'."""
        raise NotImplementedError

    def get_front_frame(self) -> Optional[np.ndarray]:
        """BGR image from the robot's own camera, or None."""
        raise NotImplementedError


class TempFake(RobotBase):
    """Team B's unblocker. Delete when Team A ships MockSportClient.

    Deliberately imperfect: the whole point of the project is that
    commanded motion != actual motion. If this were exact, the
    calibration and memory layers would have nothing to learn.
    """

    SLIP = 0.75        # asked for 1.0 m, get 0.75 m
    TURN_BIAS = 2.0    # every turn drifts 2 deg left
    NOISE = 0.03

    def __init__(self):
        self.x_cm = 0.0
        self.y_cm = 0.0
        self.heading_deg = 0.0

    def forward(self, meters: float) -> None:
        actual = meters * self.SLIP * (1 + random.gauss(0, self.NOISE))
        rad = math.radians(self.heading_deg)
        self.x_cm += actual * 100 * math.cos(rad)
        self.y_cm += actual * 100 * math.sin(rad)
        self.heading_deg += random.gauss(0, 1.5)
        time.sleep(0.2)

    def turn(self, degrees: float) -> None:
        self.heading_deg += degrees + self.TURN_BIAS + random.gauss(0, 1.0)
        time.sleep(0.2)

    def stop(self) -> None:
        pass

    def get_pose(self) -> dict:
        return {
            "x_cm": round(self.x_cm, 1),
            "y_cm": round(self.y_cm, 1),
            "heading_deg": round(self.heading_deg, 1),
            "source": "sim",
        }

    def get_front_frame(self) -> Optional[np.ndarray]:
        return None


def get_robot() -> RobotBase:
    backend = os.getenv("ROBOT_BACKEND", "temp")
    if backend == "temp":
        return TempFake()
    if backend == "mock":
        from mock_robot import MockSportClient   # Team A
        return MockSportClient()
    if backend == "go2":
        from go2_robot import Go2Robot           # Team A
        return Go2Robot()
    raise ValueError(f"unknown ROBOT_BACKEND={backend}")
