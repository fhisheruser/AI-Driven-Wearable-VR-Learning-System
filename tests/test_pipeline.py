"""Tests for the full learning pipeline."""

import pytest
from src.pipeline import LearningPipeline, PipelineResponse


@pytest.fixture
def pipeline():
    p = LearningPipeline()
    p.initialize()
    return p


class TestPipelineIntegration:
    """End-to-end pipeline tests."""

    def test_process_text_returns_response(self, pipeline):
        response = pipeline.process_text("How does the solar system work?")
        assert isinstance(response, PipelineResponse)
        assert response.query_text == "How does the solar system work?"
        assert response.total_latency_ms > 0

    def test_process_text_detects_domain(self, pipeline):
        response = pipeline.process_text("Show me a water molecule")
        assert response.semantic_result.domain == "chemistry"

    def test_process_text_generates_scene(self, pipeline):
        response = pipeline.process_text("Explain the human heart")
        assert response.scene is not None
        assert len(response.scene.objects) > 0

    def test_process_text_physics(self, pipeline):
        response = pipeline.process_text("How does the solar system work?")
        assert response.semantic_result.domain == "physics"
        assert response.scene.title is not None

    def test_process_text_math(self, pipeline):
        response = pipeline.process_text("What is the Pythagorean theorem?")
        assert response.semantic_result.domain == "mathematics"

    def test_to_dict(self, pipeline):
        response = pipeline.process_text("Show me DNA")
        data = response.to_dict()
        assert "semantic" in data
        assert "knowledge" in data
        assert "scene" in data
        assert "adaptation" in data
        assert "latency" in data

    def test_learner_profile_updates(self, pipeline):
        pipeline.process_text("Explain atoms", learner_id="test_user")
        stats = pipeline.get_learner_stats("test_user")
        assert stats["total_interactions"] >= 1

    def test_session_reset(self, pipeline):
        pipeline.process_text("What is a cell?")
        pipeline.reset_session()
        # Should not raise
        response = pipeline.process_text("What is gravity?")
        assert response.semantic_result.domain == "physics"

    def test_feedback_processing(self, pipeline):
        pipeline.process_text("Show me a molecule")
        result = pipeline.process_feedback("default", {
            "comprehension_change": 0.3,
            "engagement": 0.8,
            "interaction_time": 30.0,
        })
        assert result["feedback_processed"] is True
