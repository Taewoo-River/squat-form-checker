"""Orchestration layer that ties pose estimation, rep counting, form
evaluation, feedback, and session tracking together.

ExerciseAnalyzer  -- abstract base (extend for push-ups, lunges, etc.)
SquatAnalyzer     -- concrete implementation for bodyweight squats
"""

from abc import ABC, abstractmethod

from .pose_estimator import PoseEstimator
from .rep_counter import RepCounter, RepPhase
from .form_evaluator import SquatFormEvaluator, RepScore
from .feedback_engine import FeedbackEngine
from .workout_session import WorkoutSession
from .utils import calculate_angle


class ExerciseAnalyzer(ABC):
    """Interface every exercise-specific analyzer must implement."""

    @abstractmethod
    def process_landmarks(self, landmarks: dict | None) -> None: ...

    @abstractmethod
    def get_rep_count(self) -> int: ...

    @abstractmethod
    def get_phase(self) -> RepPhase: ...

    @abstractmethod
    def get_feedback(self) -> list[str]: ...

    @abstractmethod
    def get_session_summary(self) -> dict: ...


class SquatAnalyzer(ExerciseAnalyzer):
    """End-to-end squat analysis: counts reps, scores form, coaches user."""

    def __init__(self, pose_estimator: PoseEstimator):
        self.pe = pose_estimator
        self.rep_counter = RepCounter()
        self.form_eval = SquatFormEvaluator()
        self.feedback = FeedbackEngine()
        self.session = WorkoutSession()

        self._live_cues: list[str] = []
        self._live_metrics: dict = {}
        self._last_rep_feedback = ""
        self._knee_angle = 180.0

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------
    def process_landmarks(self, landmarks: dict | None) -> None:
        if landmarks is None:
            self._live_cues = ["No pose detected"]
            return

        metrics = self.form_eval.update(landmarks, self.pe)
        self._live_metrics = metrics
        if not metrics:
            return

        knee_angle = metrics["knee_angle"]
        self._knee_angle = knee_angle

        rep_done = self.rep_counter.update(knee_angle)

        if rep_done:
            score: RepScore = self.form_eval.score_rep(
                self.rep_counter.rep_count, knee_angle
            )
            self.session.add_rep(score)
            self._last_rep_feedback = self.feedback.get_rep_feedback(score)

        self._live_cues = self.feedback.get_live_feedback(metrics)

    # ------------------------------------------------------------------
    # Accessors used by UI layers
    # ------------------------------------------------------------------
    def get_rep_count(self) -> int:
        return self.rep_counter.rep_count

    def get_phase(self) -> RepPhase:
        return self.rep_counter.phase

    def get_feedback(self) -> list[str]:
        return self._live_cues

    def get_last_rep_feedback(self) -> str:
        return self._last_rep_feedback

    def get_knee_angle(self) -> float:
        return self._knee_angle

    def get_live_metrics(self) -> dict:
        return self._live_metrics

    def get_session_summary(self) -> dict:
        self.session.end()
        return self.session.get_summary()

    def get_rep_scores(self) -> list[RepScore]:
        return list(self.session.reps)
