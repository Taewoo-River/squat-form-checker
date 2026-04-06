import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .utils import calculate_angle
from .pose_estimator import PoseEstimator


class FormIssue(Enum):
    SHALLOW_DEPTH = "shallow_depth"
    KNEE_VALGUS = "knee_valgus"
    EXCESSIVE_LEAN = "excessive_lean"
    INCOMPLETE_LOCKOUT = "incomplete_lockout"


@dataclass
class RepScore:
    rep_number: int
    depth_score: float = 0.0
    knee_score: float = 100.0
    torso_score: float = 100.0
    lockout_score: float = 100.0
    issues: List[FormIssue] = field(default_factory=list)
    min_knee_angle: float = 180.0
    max_torso_lean: float = 0.0

    @property
    def overall_score(self) -> float:
        return (
            self.depth_score
            + self.knee_score
            + self.torso_score
            + self.lockout_score
        ) / 4

    @property
    def quality(self) -> str:
        s = self.overall_score
        if s >= 80:
            return "Good"
        if s >= 60:
            return "Fair"
        return "Poor"


class SquatFormEvaluator:
    """Per-frame form tracking and per-rep scoring for squats."""

    def __init__(
        self,
        good_depth_angle: float = 90,
        partial_depth_angle: float = 120,
        max_torso_lean: float = 45,
        valgus_ratio_threshold: float = 0.85,
        lockout_angle: float = 170,
    ):
        self.good_depth_angle = good_depth_angle
        self.partial_depth_angle = partial_depth_angle
        self.max_torso_lean = max_torso_lean
        self.valgus_ratio_threshold = valgus_ratio_threshold
        self.lockout_angle = lockout_angle
        self._reset_rep_tracking()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def update(self, landmarks: dict, pe: PoseEstimator) -> dict:
        """Compute and return live metrics from the current frame's landmarks.

        Only the sagittal-plane landmarks on the best-visible side
        (shoulder, hip, knee, ankle) are required.  This lets us keep
        tracking even when a barbell occludes the face, hands, or the
        opposite side.
        """
        def _pt(idx):
            return pe.get_point(landmarks, idx)

        def _vis(idx):
            return pe.get_visibility(landmarks, idx)

        l_vis = min(_vis(PoseEstimator.LEFT_HIP),
                    _vis(PoseEstimator.LEFT_KNEE),
                    _vis(PoseEstimator.LEFT_ANKLE))
        r_vis = min(_vis(PoseEstimator.RIGHT_HIP),
                    _vis(PoseEstimator.RIGHT_KNEE),
                    _vis(PoseEstimator.RIGHT_ANKLE))

        if l_vis >= r_vis:
            shoulder = _pt(PoseEstimator.LEFT_SHOULDER)
            hip      = _pt(PoseEstimator.LEFT_HIP)
            knee     = _pt(PoseEstimator.LEFT_KNEE)
            ankle    = _pt(PoseEstimator.LEFT_ANKLE)
        else:
            shoulder = _pt(PoseEstimator.RIGHT_SHOULDER)
            hip      = _pt(PoseEstimator.RIGHT_HIP)
            knee     = _pt(PoseEstimator.RIGHT_KNEE)
            ankle    = _pt(PoseEstimator.RIGHT_ANKLE)

        if any(p is None for p in (hip, knee, ankle)):
            return {}

        self._frame_count += 1

        knee_angle = calculate_angle(hip, knee, ankle)

        if shoulder is not None:
            hip_angle = calculate_angle(shoulder, hip, knee)
            torso_dx = shoulder[0] - hip[0]
            torso_dy = shoulder[1] - hip[1]
            torso_lean = abs(math.degrees(math.atan2(abs(torso_dx), abs(torso_dy))))
        else:
            hip_angle = 0.0
            torso_lean = 0.0

        # Knee-valgus proxy -- only when both sides are available
        l_knee_pt = _pt(PoseEstimator.LEFT_KNEE)
        r_knee_pt = _pt(PoseEstimator.RIGHT_KNEE)
        l_ankle_pt = _pt(PoseEstimator.LEFT_ANKLE)
        r_ankle_pt = _pt(PoseEstimator.RIGHT_ANKLE)

        if all(p is not None for p in (l_knee_pt, r_knee_pt, l_ankle_pt, r_ankle_pt)):
            knee_spread = abs(l_knee_pt[0] - r_knee_pt[0])
            ankle_spread = abs(l_ankle_pt[0] - r_ankle_pt[0])
            knee_ratio = knee_spread / max(ankle_spread, 1.0)
            if knee_ratio < self.valgus_ratio_threshold:
                self._valgus_frames += 1
        else:
            knee_ratio = 1.0

        self._min_knee_angle = min(self._min_knee_angle, knee_angle)
        self._max_torso_lean = max(self._max_torso_lean, torso_lean)

        return {
            "knee_angle": knee_angle,
            "hip_angle": hip_angle,
            "torso_lean": torso_lean,
            "knee_spread_ratio": knee_ratio,
        }

    # ------------------------------------------------------------------
    # Per-rep scoring
    # ------------------------------------------------------------------
    def score_rep(self, rep_number: int, final_knee_angle: float) -> RepScore:
        score = RepScore(rep_number=rep_number)
        score.min_knee_angle = self._min_knee_angle
        score.max_torso_lean = self._max_torso_lean

        # Depth
        if self._min_knee_angle <= self.good_depth_angle:
            score.depth_score = 100.0
        elif self._min_knee_angle <= self.partial_depth_angle:
            rng = self.partial_depth_angle - self.good_depth_angle
            score.depth_score = 100.0 * (self.partial_depth_angle - self._min_knee_angle) / rng
        else:
            score.depth_score = 0.0
            score.issues.append(FormIssue.SHALLOW_DEPTH)

        # Torso lean
        if self._max_torso_lean <= self.max_torso_lean:
            score.torso_score = 100.0
        else:
            excess = self._max_torso_lean - self.max_torso_lean
            score.torso_score = max(0.0, 100.0 - excess * 3)
            if score.torso_score < 70:
                score.issues.append(FormIssue.EXCESSIVE_LEAN)

        # Knee valgus
        if self._frame_count > 0:
            valgus_pct = self._valgus_frames / self._frame_count
            score.knee_score = max(0.0, 100.0 - valgus_pct * 200)
            if score.knee_score < 70:
                score.issues.append(FormIssue.KNEE_VALGUS)

        # Lockout
        if final_knee_angle >= self.lockout_angle:
            score.lockout_score = 100.0
        else:
            score.lockout_score = max(0.0, (final_knee_angle / self.lockout_angle) * 100)
            if score.lockout_score < 70:
                score.issues.append(FormIssue.INCOMPLETE_LOCKOUT)

        self._reset_rep_tracking()
        return score

    def _reset_rep_tracking(self):
        self._min_knee_angle = 180.0
        self._max_torso_lean = 0.0
        self._valgus_frames = 0
        self._frame_count = 0
