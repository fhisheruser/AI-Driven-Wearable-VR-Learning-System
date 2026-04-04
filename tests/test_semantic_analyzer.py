"""Tests for the NLU semantic analyzer."""

import pytest
from src.nlu.semantic_analyzer import SemanticAnalyzer, SemanticResult
from src.nlu.intent_classifier import LearnerIntent


@pytest.fixture
def analyzer():
    a = SemanticAnalyzer()
    a.load_models()
    return a


class TestSemanticAnalyzer:

    def test_analyze_returns_result(self, analyzer):
        result = analyzer.analyze("What is a water molecule?")
        assert isinstance(result, SemanticResult)

    def test_detects_chemistry_domain(self, analyzer):
        result = analyzer.analyze("Show me a water molecule")
        assert result.domain == "chemistry"

    def test_detects_biology_domain(self, analyzer):
        result = analyzer.analyze("Explain the cell and its nucleus")
        assert result.domain == "biology"

    def test_detects_physics_domain(self, analyzer):
        result = analyzer.analyze("How does gravity work?")
        assert result.domain == "physics"

    def test_detects_anatomy_domain(self, analyzer):
        result = analyzer.analyze("Explain the cardiovascular and respiratory skeletal muscular systems")
        assert result.domain == "anatomy"

    def test_detects_math_domain(self, analyzer):
        result = analyzer.analyze("Explain the derivative in calculus")
        assert result.domain == "mathematics"

    def test_detects_engineering_domain(self, analyzer):
        result = analyzer.analyze("How does a gear system work?")
        assert result.domain == "engineering"

    def test_extracts_entities(self, analyzer):
        result = analyzer.analyze("Show me the water molecule and its bonds")
        assert len(result.entities) > 0

    def test_extracts_keywords(self, analyzer):
        result = analyzer.analyze("Explain the structure of DNA")
        assert len(result.context_keywords) > 0

    def test_visual_representation_intent(self, analyzer):
        result = analyzer.analyze("Show me a 3D model of the solar system")
        assert result.visualization_type is not None

    def test_complexity_basic(self, analyzer):
        result = analyzer.analyze("What is a simple introduction to atoms?")
        assert result.complexity_hint == "basic"

    def test_complexity_advanced(self, analyzer):
        result = analyzer.analyze("Explain the advanced quantum molecular orbital theory")
        assert result.complexity_hint == "advanced"

    def test_context_continuity(self, analyzer):
        analyzer.analyze("Tell me about the cell")
        result = analyzer.analyze("Tell me more")
        # Should inherit domain from context
        assert result.domain == "biology"

    def test_reset_context(self, analyzer):
        analyzer.analyze("Explain DNA")
        analyzer.reset_context()
        result = analyzer.analyze("Tell me more")
        # No context to inherit
        assert result.domain is None or result.domain != "biology" or result.topic is None
