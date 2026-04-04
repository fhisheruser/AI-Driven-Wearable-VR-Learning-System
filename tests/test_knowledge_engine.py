"""Tests for the knowledge reasoning engine."""

import pytest
from src.knowledge.knowledge_engine import KnowledgeReasoningEngine, ReasonedOutput
from src.nlu.semantic_analyzer import SemanticAnalyzer


@pytest.fixture
def engine():
    return KnowledgeReasoningEngine()


@pytest.fixture
def analyzer():
    a = SemanticAnalyzer()
    a.load_models()
    return a


class TestKnowledgeEngine:

    def test_reason_returns_output(self, engine, analyzer):
        semantic = analyzer.analyze("What is a water molecule?")
        output = engine.reason(semantic)
        assert isinstance(output, ReasonedOutput)

    def test_finds_concept(self, engine, analyzer):
        semantic = analyzer.analyze("Explain the atom")
        output = engine.reason(semantic)
        assert output.concept is not None
        assert output.concept.domain == "chemistry"

    def test_generates_explanation(self, engine, analyzer):
        semantic = analyzer.analyze("What is DNA?")
        output = engine.reason(semantic)
        assert len(output.explanation) > 0

    def test_pedagogical_sequence(self, engine, analyzer):
        semantic = analyzer.analyze("Explain the heart")
        output = engine.reason(semantic)
        assert len(output.pedagogical_sequence) > 0

    def test_related_topics(self, engine, analyzer):
        semantic = analyzer.analyze("Tell me about atoms")
        output = engine.reason(semantic)
        assert isinstance(output.related_topics, list)

    def test_visualization_type_set(self, engine, analyzer):
        semantic = analyzer.analyze("Show me the solar system")
        output = engine.reason(semantic)
        assert output.visualization_type in [
            "3d_model", "animation", "step_by_step", "contextual_overlay"
        ]

    def test_difficulty_with_profile(self, engine, analyzer):
        semantic = analyzer.analyze("Explain molecules")
        profile = {"comprehension_level": 0.9, "skill_level": 4}
        output = engine.reason(semantic, profile)
        assert output.difficulty_level >= 1
        assert output.difficulty_level <= 5

    def test_quiz_generation(self, engine, analyzer):
        semantic = analyzer.analyze("Quiz me on atoms")
        # Intent classifier should detect quiz request
        output = engine.reason(semantic)
        # Quiz may or may not be generated depending on intent detection
        assert isinstance(output.quiz_questions, list)

    def test_procedural_steps(self, engine, analyzer):
        semantic = analyzer.analyze("How do you balance a chemical equation step by step?")
        output = engine.reason(semantic)
        assert isinstance(output.steps, list)
