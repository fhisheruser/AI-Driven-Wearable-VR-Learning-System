"""Adaptive Learning Engine using Reinforcement Learning (PPO).

Implements a Proximal Policy Optimization-based personalization model
that dynamically adjusts content depth, pacing, and visualization
complexity based on learner interaction history and comprehension signals.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .learner_profile import LearnerProfile

logger = logging.getLogger(__name__)


@dataclass
class AdaptationAction:
    """An action taken by the adaptive engine to personalize learning."""
    difficulty_adjustment: int  # -2 to +2
    content_depth: str  # "overview", "standard", "detailed", "expert"
    visualization_complexity: str  # "simple", "moderate", "complex"
    pacing: str  # "slow", "moderate", "fast"
    reinforcement_strategy: str  # "repeat", "advance", "branch"
    explanation_style: str  # "concise", "detailed", "example_based"


class LearningEnvironment:
    """Gymnasium-compatible environment for the adaptive learning RL agent.

    State: [comprehension, engagement, difficulty, time_on_task, quiz_score,
            interaction_count, follow_up_rate]
    Action: [difficulty_adj, depth, complexity, pacing, strategy]
    Reward: comprehension improvement + engagement signal
    """

    def __init__(self):
        self.state_dim = 7
        self.action_dim = 5
        self._state = np.zeros(self.state_dim, dtype=np.float32)
        self._step_count = 0

    def reset(self, learner_profile: LearnerProfile) -> np.ndarray:
        """Reset environment with learner profile."""
        self._state = np.array([
            learner_profile.overall_comprehension,
            min(1.0, learner_profile.avg_interaction_duration / 300),  # Normalize to 5 min
            learner_profile.skill_level / 5.0,
            0.0,  # time_on_current_task
            self._get_avg_quiz(learner_profile),
            min(1.0, learner_profile.total_interactions / 100),
            learner_profile.follow_up_rate,
        ], dtype=np.float32)
        self._step_count = 0
        return self._state.copy()

    def step(self, action: np.ndarray, feedback: dict) -> tuple[np.ndarray, float, bool]:
        """Take a step in the environment.

        Args:
            action: Action vector from policy
            feedback: Feedback from learner interaction

        Returns:
            (next_state, reward, done)
        """
        comprehension_change = feedback.get("comprehension_change", 0.0)
        engagement = feedback.get("engagement", 0.5)
        quiz_result = feedback.get("quiz_score", None)
        interaction_time = feedback.get("interaction_time", 0.0)

        # Update state
        self._state[0] = np.clip(self._state[0] + comprehension_change, 0, 1)
        self._state[1] = engagement
        self._state[3] = min(1.0, interaction_time / 300)
        if quiz_result is not None:
            self._state[4] = 0.7 * self._state[4] + 0.3 * quiz_result
        self._state[5] = min(1.0, self._state[5] + 0.01)

        # Calculate reward
        reward = (
            2.0 * comprehension_change +
            1.0 * (engagement - 0.5) +
            (1.5 * quiz_result if quiz_result is not None else 0.0) -
            0.1 * abs(action[0])  # Penalize large difficulty changes
        )

        self._step_count += 1
        done = self._step_count >= 100

        return self._state.copy(), float(reward), done

    def _get_avg_quiz(self, profile: LearnerProfile) -> float:
        all_scores = []
        for dp in profile.domain_progress.values():
            all_scores.extend(dp.quiz_scores)
        return sum(all_scores) / len(all_scores) if all_scores else 0.5


class PPOAgent:
    """Proximal Policy Optimization agent for adaptive learning.

    Uses a simple neural network policy to map learner states to
    adaptation actions. Can be trained online or loaded from pre-trained weights.
    """

    def __init__(self, state_dim: int = 7, action_dim: int = 5,
                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 clip_range: float = 0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_range = clip_range

        # Simple policy network weights (2-layer MLP)
        self.hidden_size = 64
        np.random.seed(42)
        self.W1 = np.random.randn(state_dim, self.hidden_size).astype(np.float32) * 0.1
        self.b1 = np.zeros(self.hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(self.hidden_size, action_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(action_dim, dtype=np.float32)

        # Value network weights
        self.Wv1 = np.random.randn(state_dim, self.hidden_size).astype(np.float32) * 0.1
        self.bv1 = np.zeros(self.hidden_size, dtype=np.float32)
        self.Wv2 = np.random.randn(self.hidden_size, 1).astype(np.float32) * 0.1
        self.bv2 = np.zeros(1, dtype=np.float32)

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _forward_policy(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through policy network."""
        h = self._relu(state @ self.W1 + self.b1)
        return self._tanh(h @ self.W2 + self.b2)  # Actions in [-1, 1]

    def _forward_value(self, state: np.ndarray) -> float:
        """Forward pass through value network."""
        h = self._relu(state @ self.Wv1 + self.bv1)
        return float((h @ self.Wv2 + self.bv2)[0])

    def select_action(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Select an action using the current policy with exploration noise."""
        mean_action = self._forward_policy(state)

        # Add Gaussian exploration noise
        noise = np.random.randn(self.action_dim).astype(np.float32) * 0.1
        action = np.clip(mean_action + noise, -1, 1)

        # Compute log probability (Gaussian policy)
        log_prob = -0.5 * np.sum((action - mean_action) ** 2) / 0.1

        return action, float(log_prob)

    def get_value(self, state: np.ndarray) -> float:
        """Estimate state value."""
        return self._forward_value(state)

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, log_prob: float, value: float):
        """Store a transition for training."""
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def update(self) -> dict[str, float]:
        """Perform PPO update on stored experience.

        Returns training metrics.
        """
        if len(self.states) < 2:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        old_log_probs = np.array(self.log_probs)
        old_values = np.array(self.values)

        # Compute returns and advantages (GAE)
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        last_return = 0.0
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * last_return
            next_value = old_values[t + 1] if t + 1 < len(old_values) else 0.0
            delta = rewards[t] + self.gamma * next_value - old_values[t]
            advantages[t] = delta + self.gamma * 0.95 * last_advantage
            last_return = returns[t]
            last_advantage = advantages[t]

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # PPO update (simplified gradient step)
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for i in range(len(states)):
            # Current policy
            mean_action = self._forward_policy(states[i])
            new_log_prob = -0.5 * np.sum((actions[i] - mean_action) ** 2) / 0.1

            # Importance ratio
            ratio = np.exp(new_log_prob - old_log_probs[i])

            # Clipped surrogate objective
            surr1 = ratio * advantages[i]
            surr2 = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages[i]
            policy_loss = -min(surr1, surr2)

            # Value loss
            value_pred = self._forward_value(states[i])
            value_loss = (value_pred - returns[i]) ** 2

            total_policy_loss += policy_loss
            total_value_loss += value_loss

            # Simple gradient update for policy
            h = self._relu(states[i] @ self.W1 + self.b1)
            grad_action = -2 * (actions[i] - mean_action) * advantages[i]
            self.W2 -= self.learning_rate * np.outer(h, grad_action) * 0.01
            self.b2 -= self.learning_rate * grad_action * 0.01

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()

        n = max(len(states), 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
        }

    def save(self, path: str):
        """Save model weights."""
        np.savez(path,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 Wv1=self.Wv1, bv1=self.bv1, Wv2=self.Wv2, bv2=self.bv2)

    def load(self, path: str):
        """Load model weights."""
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.Wv1, self.bv1 = data["Wv1"], data["bv1"]
        self.Wv2, self.bv2 = data["Wv2"], data["bv2"]


class AdaptiveLearningEngine:
    """Main adaptive learning engine that uses PPO to personalize instruction.

    Dynamically adjusts content depth, pacing, and visualization complexity
    based on learner interaction history and comprehension signals.
    """

    def __init__(self, learning_rate: float = 0.0003, gamma: float = 0.99):
        self.env = LearningEnvironment()
        self.agent = PPOAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
        )
        self._current_state = None

    def initialize_session(self, learner_profile: LearnerProfile):
        """Initialize a learning session for a learner."""
        self._current_state = self.env.reset(learner_profile)

    def get_adaptation(self, learner_profile: LearnerProfile) -> AdaptationAction:
        """Get personalized adaptation parameters for the current interaction."""
        if self._current_state is None:
            self.initialize_session(learner_profile)

        action, log_prob = self.agent.select_action(self._current_state)
        value = self.agent.get_value(self._current_state)

        # Map continuous action to discrete adaptation parameters
        adaptation = self._map_action(action, learner_profile)

        # Store for training
        self.agent.store_transition(
            self._current_state, action, 0.0, log_prob, value
        )

        return adaptation

    def process_feedback(self, feedback: dict) -> dict[str, float]:
        """Process learner feedback and update the model.

        Args:
            feedback: dict with keys like 'comprehension_change', 'engagement',
                     'quiz_score', 'interaction_time'

        Returns:
            Training metrics
        """
        if self._current_state is None:
            return {}

        # Get last stored transition and update reward
        action = self.agent.actions[-1] if self.agent.actions else np.zeros(self.env.action_dim)

        next_state, reward, done = self.env.step(action, feedback)

        # Update stored reward
        if self.agent.rewards:
            self.agent.rewards[-1] = reward

        self._current_state = next_state

        # Periodic PPO update
        metrics = {}
        if len(self.agent.states) >= 32:
            metrics = self.agent.update()
            logger.info(f"PPO update: {metrics}")

        return metrics

    def _map_action(self, action: np.ndarray,
                    profile: LearnerProfile) -> AdaptationAction:
        """Map continuous action vector to discrete adaptation parameters."""
        # action[0]: difficulty adjustment (-1 to 1) -> (-2 to +2)
        difficulty_adj = int(np.round(action[0] * 2))
        difficulty_adj = max(-2, min(2, difficulty_adj))

        # action[1]: content depth
        depth_idx = int((action[1] + 1) / 2 * 3)
        depth_options = ["overview", "standard", "detailed", "expert"]
        content_depth = depth_options[min(depth_idx, 3)]

        # action[2]: visualization complexity
        viz_idx = int((action[2] + 1) / 2 * 2)
        viz_options = ["simple", "moderate", "complex"]
        viz_complexity = viz_options[min(viz_idx, 2)]

        # action[3]: pacing
        pace_idx = int((action[3] + 1) / 2 * 2)
        pace_options = ["slow", "moderate", "fast"]
        pacing = pace_options[min(pace_idx, 2)]

        # action[4]: reinforcement strategy
        strategy_idx = int((action[4] + 1) / 2 * 2)
        strategy_options = ["repeat", "advance", "branch"]
        strategy = strategy_options[min(strategy_idx, 2)]

        # Determine explanation style based on profile
        if profile.preferred_modality == "visual":
            explanation_style = "example_based"
        elif profile.overall_comprehension > 0.7:
            explanation_style = "concise"
        else:
            explanation_style = "detailed"

        return AdaptationAction(
            difficulty_adjustment=difficulty_adj,
            content_depth=content_depth,
            visualization_complexity=viz_complexity,
            pacing=pacing,
            reinforcement_strategy=strategy,
            explanation_style=explanation_style,
        )

    def save_model(self, path: str):
        """Save the PPO model."""
        self.agent.save(path)

    def load_model(self, path: str):
        """Load a pre-trained PPO model."""
        self.agent.load(path)
