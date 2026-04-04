"""Tests for the evaluation metrics module."""

import pytest
from src.evaluation.metrics import (
    EvaluationEngine, ASRMetrics, LatencyMetrics,
    LearningEffectiveness, NASATLXScore,
)


class TestASRMetrics:

    def test_perfect_accuracy(self):
        m = ASRMetrics()
        m.update("the solar system", "the solar system")
        assert m.accuracy == 1.0
        assert m.word_error_rate == 0.0

    def test_word_errors(self):
        m = ASRMetrics()
        m.update("the solar system works", "the solar systems work")
        assert m.word_error_rate > 0

    def test_empty_input(self):
        m = ASRMetrics()
        # No words evaluated, WER is 0, so accuracy defaults to 1.0
        assert m.word_error_rate == 0.0


class TestLatencyMetrics:

    def test_mean_latency(self):
        m = LatencyMetrics()
        m.record(100)
        m.record(200)
        m.record(300)
        assert m.mean == 200.0

    def test_p95(self):
        m = LatencyMetrics()
        for i in range(100):
            m.record(i * 10)
        assert m.p95 >= 900

    def test_within_target(self):
        m = LatencyMetrics(target_ms=2000)
        m.record(1500)
        m.record(2500)
        assert m.within_target == 0.5


class TestLearningEffectiveness:

    def test_mean_gain(self):
        le = LearningEffectiveness()
        le.pre_scores = [50, 60, 70]
        le.post_scores = [70, 80, 90]
        assert le.mean_gain == 20.0

    def test_paired_t_test_significant(self):
        le = LearningEffectiveness()
        le.pre_scores = [50, 55, 60, 65, 70, 55, 60, 65, 50, 55]
        le.post_scores = [80, 85, 90, 85, 95, 85, 90, 85, 80, 85]
        result = le.paired_t_test()
        assert result["significant_at_95"] is True
        assert result["mean_gain"] > 0

    def test_paired_t_test_insufficient_data(self):
        le = LearningEffectiveness()
        le.pre_scores = [50]
        le.post_scores = [70]
        result = le.paired_t_test()
        assert result["n"] == 1


class TestEvaluationEngine:

    def test_generate_report(self):
        engine = EvaluationEngine()
        engine.record_asr_result("hello world", "hello world")
        engine.record_latency(500)
        engine.record_learning_scores(60, 85)
        engine.record_nasa_tlx({
            "mental_demand": 40, "physical_demand": 10,
            "temporal_demand": 30, "performance": 20,
            "effort": 35, "frustration": 15,
        })
        engine.record_engagement({"interaction_duration": 120, "query_count": 8})

        report = engine.generate_report()
        assert "asr" in report
        assert "latency" in report
        assert "learning_effectiveness" in report
        assert "cognitive_load" in report
        assert "engagement" in report
        assert report["asr"]["accuracy"] == 1.0
