from enum import Enum
import time


class RepPhase(Enum):
    STANDING = "Standing"
    DESCENDING = "Descending"
    BOTTOM = "Bottom"
    ASCENDING = "Ascending"


class RepCounter:
    """Finite-state machine that counts squat reps from knee-angle readings.

    State transitions:
        STANDING  --knee drops below *descent_threshold*-->  DESCENDING
        DESCENDING  --knee reaches *bottom_angle*-->         BOTTOM
        BOTTOM  --knee rises above *bottom_angle*-->         ASCENDING
        ASCENDING  --knee reaches *standing_angle*-->        STANDING  (rep++)

    A hysteresis band prevents noisy oscillations at the thresholds.
    """

    def __init__(
        self,
        standing_angle: float = 160,
        descent_threshold: float = 150,
        bottom_angle: float = 100,
        hysteresis: float = 5,
    ):
        self.standing_angle = standing_angle
        self.descent_threshold = descent_threshold
        self.bottom_angle = bottom_angle
        self.hysteresis = hysteresis

        self.phase = RepPhase.STANDING
        self.rep_count = 0
        self.min_angle_in_rep = 180.0
        self._phase_log: list[tuple[RepPhase, float]] = []

    def update(self, knee_angle: float) -> bool:
        """Feed the current knee angle. Returns True when a rep completes."""
        prev = self.phase
        completed = False

        if self.phase == RepPhase.STANDING:
            if knee_angle < self.descent_threshold - self.hysteresis:
                self.phase = RepPhase.DESCENDING
                self.min_angle_in_rep = knee_angle

        elif self.phase == RepPhase.DESCENDING:
            self.min_angle_in_rep = min(self.min_angle_in_rep, knee_angle)
            if knee_angle <= self.bottom_angle:
                self.phase = RepPhase.BOTTOM
            elif knee_angle > self.descent_threshold + self.hysteresis:
                self.phase = RepPhase.STANDING

        elif self.phase == RepPhase.BOTTOM:
            self.min_angle_in_rep = min(self.min_angle_in_rep, knee_angle)
            if knee_angle > self.bottom_angle + self.hysteresis:
                self.phase = RepPhase.ASCENDING

        elif self.phase == RepPhase.ASCENDING:
            if knee_angle >= self.standing_angle - self.hysteresis:
                self.phase = RepPhase.STANDING
                self.rep_count += 1
                completed = True
            elif knee_angle < self.bottom_angle:
                self.phase = RepPhase.BOTTOM

        if self.phase != prev:
            self._phase_log.append((self.phase, time.time()))

        return completed

    def reset(self):
        self.phase = RepPhase.STANDING
        self.rep_count = 0
        self.min_angle_in_rep = 180.0
        self._phase_log.clear()
