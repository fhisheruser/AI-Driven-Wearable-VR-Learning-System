"""Tests for the adaptive learning engine (PPO)."""

import pytest
import numpy as np
from src.personalization.adaptive_engine import (
    AdaptiveLearningEngine, PPOAgent, LearningEnvironment, AdaptationAction,
)
from src.personalization.learner_profile import LearnerProfile


@pytest.fixture
def profile():
    return LearnerProfile(learner_id="test")


class TestLearningEnvironment:

    def test_reset(self, profile):
        env = LearningEnvironment()
        state = env.reset(profile)
        assert state.shape == (7,)
        assert all(0 <= s <= 1 for s in state)

    def test_step(self, profile):
        env = LearningEnvironment()
        env.reset(profile)
        action = np.zeros(5, dtype=np.float32)
        feedback = {"comprehension_change": 0.1, "engagement": 0.7}
        next_state, reward, done = env.step(action, feedback)
        assert next_state.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)


class TestPPOAgent:

    def test_select_action(self):
        agent = PPOAgent()
        state = np.random.randn(7).astype(np.float32)
        action, log_prob = agent.select_action(state)
        assert action.shape == (5,)
        assert all(-1 <= a <= 1 for a in action)

    def test_get_value(self):
        agent = PPOAgent()
        state = np.random.randn(7).astype(np.float32)
        value = agent.get_value(state)
        assert isinstance(value, float)

    def test_update_with_data(self):
        agent = PPOAgent()
        for _ in range(5):
            state = np.random.randn(7).astype(np.float32)
            action, log_prob = agent.select_action(state)
            value = agent.get_value(state)
            agent.store_transition(state, action, 1.0, log_prob, value)
        metrics = agent.update()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics


class TestAdaptiveLearningEngine:

    def test_get_adaptation(self, profile):
        engine = AdaptiveLearningEngine()
        adaptation = engine.get_adaptation(profile)
        assert isinstance(adaptation, AdaptationAction)
        assert adaptation.content_depth in ["overview", "standard", "detailed", "expert"]
        assert adaptation.pacing in ["slow", "moderate", "fast"]

    def test_process_feedback(self, profile):
        engine = AdaptiveLearningEngine()
        engine.initialize_session(profile)
        engine.get_adaptation(profile)
        metrics = engine.process_feedback({
            "comprehension_change": 0.2,
            "engagement": 0.8,
        })
        assert isinstance(metrics, dict)
