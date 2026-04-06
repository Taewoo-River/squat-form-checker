import time
from .form_evaluator import RepScore


class WorkoutSession:
    """Accumulates per-rep results and produces a summary."""

    def __init__(self):
        self.start_time = time.time()
        self.end_time: float | None = None
        self.reps: list[RepScore] = []

    def add_rep(self, rep_score: RepScore):
        self.reps.append(rep_score)

    def end(self):
        if self.end_time is None:
            self.end_time = time.time()

    @property
    def total_reps(self) -> int:
        return len(self.reps)

    @property
    def good_reps(self) -> int:
        return sum(1 for r in self.reps if r.quality == "Good")

    @property
    def fair_reps(self) -> int:
        return sum(1 for r in self.reps if r.quality == "Fair")

    @property
    def poor_reps(self) -> int:
        return sum(1 for r in self.reps if r.quality == "Poor")

    @property
    def avg_score(self) -> float:
        if not self.reps:
            return 0.0
        return sum(r.overall_score for r in self.reps) / len(self.reps)

    @property
    def duration(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def get_summary(self) -> dict:
        issue_counts: dict[str, int] = {}
        for r in self.reps:
            for issue in r.issues:
                issue_counts[issue.value] = issue_counts.get(issue.value, 0) + 1

        return {
            "total_reps": self.total_reps,
            "good_reps": self.good_reps,
            "fair_reps": self.fair_reps,
            "poor_reps": self.poor_reps,
            "avg_score": round(self.avg_score, 1),
            "duration_sec": round(self.duration, 1),
            "issue_counts": issue_counts,
        }
