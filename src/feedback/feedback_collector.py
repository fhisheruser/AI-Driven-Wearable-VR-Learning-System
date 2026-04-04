"""Feedback Collection module for adaptive learning.

Captures implicit and explicit feedback from learner interactions including
gaze tracking, interaction duration, follow-up queries, and verbal feedback.
Feeds signals back into the learning model for continuous adaptation.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSignal:
    """Processed feedback signal from a learner interaction."""
    timestamp: float = field(default_factory=time.time)
    comprehension_change: float = 0.0  # -1.0 to 1.0
    engagement_level: float = 0.5  # 0.0 to 1.0
    interaction_time: float = 0.0  # seconds
    quiz_score: float | None = None  # 0.0 to 1.0
    domain: str | None = None
    follow_up_asked: bool = False
    gaze_duration: float | None = None  # seconds on content
    verbal_feedback: str | None = None
    feedback_type: str = "implicit"  # "implicit" or "explicit"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "comprehension_change": self.comprehension_change,
            "engagement_level": self.engagement_level,
            "interaction_time": self.interaction_time,
            "quiz_score": self.quiz_score,
            "domain": self.domain,
            "follow_up_asked": self.follow_up_asked,
            "gaze_duration": self.gaze_duration,
            "verbal_feedback": self.verbal_feedback,
            "feedback_type": self.feedback_type,
        }


class FeedbackCollector:
    """Collects and processes learner feedback for the adaptive engine.

    Supports both implicit feedback (gaze, interaction patterns)
    and explicit feedback (verbal, quiz results).
    """

    def __init__(self):
        self.history: list[FeedbackSignal] = []

    def collect(self, feedback_data: dict) -> FeedbackSignal:
        """Process raw feedback data into a structured signal.

        Args:
            feedback_data: Dictionary with any combination of:
                - comprehension_change: float (-1 to 1)
                - engagement: float (0 to 1)
                - interaction_time: float (seconds)
                - quiz_score: float (0 to 1)
                - domain: str
                - follow_up: bool
                - gaze_duration: float (seconds)
                - verbal_feedback: str
                - rating: int (1-5, converted to engagement)
        """
        signal = FeedbackSignal()

        # Direct signals
        signal.comprehension_change = max(-1.0, min(1.0,
            feedback_data.get("comprehension_change", 0.0)))
        signal.interaction_time = max(0, feedback_data.get("interaction_time", 0.0))
        signal.domain = feedback_data.get("domain")
        signal.follow_up_asked = feedback_data.get("follow_up", False)
        signal.gaze_duration = feedback_data.get("gaze_duration")
        signal.verbal_feedback = feedback_data.get("verbal_feedback")

        # Quiz score
        quiz = feedback_data.get("quiz_score")
        if quiz is not None:
            signal.quiz_score = max(0.0, min(1.0, quiz))

        # Engagement: combine explicit rating with implicit signals
        if "rating" in feedback_data:
            signal.engagement_level = (feedback_data["rating"] - 1) / 4.0
            signal.feedback_type = "explicit"
        elif "engagement" in feedback_data:
            signal.engagement_level = max(0.0, min(1.0,
                feedback_data["engagement"]))
        else:
            # Infer engagement from implicit signals
            signal.engagement_level = self._infer_engagement(signal)

        # Infer comprehension change if not directly provided
        if signal.comprehension_change == 0.0 and signal.quiz_score is not None:
            signal.comprehension_change = (signal.quiz_score - 0.5) * 0.4

        if signal.verbal_feedback:
            signal.feedback_type = "explicit"

        self.history.append(signal)
        return signal

    def _infer_engagement(self, signal: FeedbackSignal) -> float:
        """Infer engagement level from implicit behavioral signals."""
        engagement = 0.5  # baseline

        # Longer interaction time suggests higher engagement (up to a point)
        if signal.interaction_time > 0:
            time_factor = min(signal.interaction_time / 120.0, 1.0)
            engagement += 0.2 * time_factor

        # Gaze duration on content indicates attention
        if signal.gaze_duration is not None and signal.gaze_duration > 0:
            gaze_factor = min(signal.gaze_duration / 60.0, 1.0)
            engagement += 0.15 * gaze_factor

        # Follow-up questions indicate engagement
        if signal.follow_up_asked:
            engagement += 0.15

        return min(1.0, engagement)

    def get_session_summary(self) -> dict:
        """Get summary statistics for the current feedback session."""
        if not self.history:
            return {"total_signals": 0}

        engagements = [s.engagement_level for s in self.history]
        comprehensions = [s.comprehension_change for s in self.history]
        quiz_scores = [s.quiz_score for s in self.history if s.quiz_score is not None]

        return {
            "total_signals": len(self.history),
            "avg_engagement": sum(engagements) / len(engagements),
            "avg_comprehension_change": sum(comprehensions) / len(comprehensions),
            "avg_quiz_score": sum(quiz_scores) / len(quiz_scores) if quiz_scores else None,
            "total_interaction_time": sum(s.interaction_time for s in self.history),
            "follow_up_rate": sum(1 for s in self.history if s.follow_up_asked) / len(self.history),
        }

    def reset(self):
        """Reset feedback history for a new session."""
        self.history.clear()
