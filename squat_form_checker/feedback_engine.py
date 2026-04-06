from .form_evaluator import FormIssue, RepScore


class FeedbackEngine:
    """Converts form metrics into human-readable coaching cues."""

    REP_MESSAGES = {
        FormIssue.SHALLOW_DEPTH: "Go lower -- try to reach parallel.",
        FormIssue.KNEE_VALGUS: "Push your knees out over your toes.",
        FormIssue.EXCESSIVE_LEAN: "Keep your chest up -- less forward lean.",
        FormIssue.INCOMPLETE_LOCKOUT: "Stand fully upright at the top.",
    }

    def get_rep_feedback(self, rep_score: RepScore) -> str:
        if not rep_score.issues:
            return f"Great rep! (score {rep_score.overall_score:.0f})"
        parts = [self.REP_MESSAGES[i] for i in rep_score.issues]
        return " | ".join(parts)

    def get_live_feedback(self, metrics: dict) -> list[str]:
        """Return a list of short real-time coaching cues."""
        if not metrics:
            return ["Waiting for pose..."]

        cues: list[str] = []
        if metrics.get("torso_lean", 0) > 45:
            cues.append("Chest up!")
        if metrics.get("knee_spread_ratio", 1) < 0.85:
            cues.append("Knees out!")
        if not cues:
            cues.append("Good form")
        return cues
