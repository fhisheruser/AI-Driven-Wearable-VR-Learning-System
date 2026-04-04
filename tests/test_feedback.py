"""Tests for the feedback collection module."""

import pytest
from src.feedback.feedback_collector import FeedbackCollector, FeedbackSignal


@pytest.fixture
def collector():
    return FeedbackCollector()


class TestFeedbackCollector:

    def test_collect_basic_feedback(self, collector):
        signal = collector.collect({"engagement": 0.8, "interaction_time": 30.0})
        assert isinstance(signal, FeedbackSignal)
        assert signal.engagement_level == 0.8
        assert signal.interaction_time == 30.0

    def test_collect_quiz_score(self, collector):
        signal = collector.collect({"quiz_score": 0.85, "domain": "chemistry"})
        assert signal.quiz_score == 0.85
        assert signal.domain == "chemistry"

    def test_collect_rating(self, collector):
        signal = collector.collect({"rating": 4})
        assert signal.engagement_level == 0.75  # (4-1)/4
        assert signal.feedback_type == "explicit"

    def test_collect_verbal_feedback(self, collector):
        signal = collector.collect({"verbal_feedback": "Great explanation!"})
        assert signal.verbal_feedback == "Great explanation!"
        assert signal.feedback_type == "explicit"

    def test_infer_engagement_from_time(self, collector):
        signal = collector.collect({"interaction_time": 120.0})
        # 120s should give higher engagement than baseline 0.5
        assert signal.engagement_level > 0.5

    def test_infer_engagement_from_followup(self, collector):
        signal = collector.collect({"follow_up": True})
        assert signal.engagement_level > 0.5

    def test_comprehension_from_quiz(self, collector):
        signal = collector.collect({"quiz_score": 0.9})
        assert signal.comprehension_change > 0

    def test_session_summary(self, collector):
        collector.collect({"engagement": 0.7, "interaction_time": 20})
        collector.collect({"engagement": 0.9, "interaction_time": 40})
        summary = collector.get_session_summary()
        assert summary["total_signals"] == 2
        assert summary["avg_engagement"] == 0.8

    def test_reset(self, collector):
        collector.collect({"engagement": 0.5})
        collector.reset()
        assert len(collector.history) == 0

    def test_signal_to_dict(self, collector):
        signal = collector.collect({"engagement": 0.6, "domain": "physics"})
        d = signal.to_dict()
        assert d["engagement_level"] == 0.6
        assert d["domain"] == "physics"
