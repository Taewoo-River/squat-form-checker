"""Squat form checker package.

MediaPipe and other heavy dependencies are loaded lazily on first attribute access
so lightweight imports (e.g. ``squat_form_checker.utils``) stay fast — needed for
Streamlit to paint the UI before pose code runs.
"""

import importlib

_EXPORTS = {
    "PoseEstimator": ".pose_estimator",
    "RepCounter": ".rep_counter",
    "RepPhase": ".rep_counter",
    "SquatFormEvaluator": ".form_evaluator",
    "FormIssue": ".form_evaluator",
    "RepScore": ".form_evaluator",
    "FeedbackEngine": ".feedback_engine",
    "WorkoutSession": ".workout_session",
    "ExerciseAnalyzer": ".analyzer",
    "SquatAnalyzer": ".analyzer",
}

_attr_cache: dict[str, object] = {}


def __getattr__(name: str):
    if name in _attr_cache:
        return _attr_cache[name]
    mod_path = _EXPORTS.get(name)
    if mod_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_path, __name__)
    value = getattr(mod, name)
    _attr_cache[name] = value
    return value


__all__ = list(_EXPORTS.keys())
