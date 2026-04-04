"""Learner Profile management for personalized learning.

Maintains dynamic learner profiles including prior interactions,
comprehension levels, and preferred learning modalities.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DomainProgress:
    """Tracks learner progress in a specific domain."""
    domain: str
    concepts_explored: list[str] = field(default_factory=list)
    quiz_scores: list[float] = field(default_factory=list)
    interaction_count: int = 0
    total_time_seconds: float = 0.0
    difficulty_level: int = 1
    comprehension_score: float = 0.5

    @property
    def average_quiz_score(self) -> float:
        return sum(self.quiz_scores) / len(self.quiz_scores) if self.quiz_scores else 0.0


@dataclass
class LearnerProfile:
    """Complete learner profile for personalization."""
    learner_id: str
    name: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # Learning preferences
    preferred_modality: str = "visual"  # visual, auditory, kinesthetic
    preferred_complexity: str = "intermediate"
    preferred_pace: str = "moderate"  # slow, moderate, fast

    # Overall metrics
    total_interactions: int = 0
    total_session_time: float = 0.0
    overall_comprehension: float = 0.5
    skill_level: int = 2  # 1-5

    # Domain-specific progress
    domain_progress: dict[str, DomainProgress] = field(default_factory=dict)

    # Interaction history (recent)
    recent_queries: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    recent_intents: list[str] = field(default_factory=list)

    # Engagement metrics
    avg_interaction_duration: float = 0.0
    follow_up_rate: float = 0.0  # How often they ask follow-up questions
    quiz_participation_rate: float = 0.0

    def update_interaction(self, domain: str, topic: str, intent: str,
                           duration: float, comprehension_signal: float = 0.5):
        """Update profile with a new interaction."""
        self.total_interactions += 1
        self.last_active = time.time()
        self.total_session_time += duration

        # Update recent history
        self.recent_queries = (self.recent_queries + [topic])[-20:]
        self.recent_topics = (self.recent_topics + [topic])[-20:]
        self.recent_intents = (self.recent_intents + [intent])[-20:]

        # Update domain progress
        if domain not in self.domain_progress:
            self.domain_progress[domain] = DomainProgress(domain=domain)

        dp = self.domain_progress[domain]
        dp.interaction_count += 1
        dp.total_time_seconds += duration
        if topic and topic not in dp.concepts_explored:
            dp.concepts_explored.append(topic)

        # Update comprehension with exponential moving average
        alpha = 0.3
        dp.comprehension_score = alpha * comprehension_signal + (1 - alpha) * dp.comprehension_score
        self.overall_comprehension = alpha * comprehension_signal + (1 - alpha) * self.overall_comprehension

        # Update engagement metrics
        self.avg_interaction_duration = (
            (self.avg_interaction_duration * (self.total_interactions - 1) + duration)
            / self.total_interactions
        )

        # Adjust skill level based on comprehension
        if self.overall_comprehension > 0.8 and self.total_interactions > 10:
            self.skill_level = min(5, self.skill_level + 1)
        elif self.overall_comprehension < 0.3 and self.total_interactions > 5:
            self.skill_level = max(1, self.skill_level - 1)

    def record_quiz_result(self, domain: str, score: float):
        """Record a quiz score for a domain."""
        if domain not in self.domain_progress:
            self.domain_progress[domain] = DomainProgress(domain=domain)
        self.domain_progress[domain].quiz_scores.append(score)

    def to_dict(self) -> dict:
        """Convert profile to dictionary for serialization."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LearnerProfile":
        """Create profile from dictionary."""
        domain_progress = {}
        for domain, dp_data in data.pop("domain_progress", {}).items():
            domain_progress[domain] = DomainProgress(**dp_data)
        profile = cls(**data)
        profile.domain_progress = domain_progress
        return profile


class LearnerProfileManager:
    """Manages persistence and retrieval of learner profiles."""

    def __init__(self, storage_path: str = "data/learner_profiles"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, LearnerProfile] = {}

    def get_profile(self, learner_id: str) -> LearnerProfile:
        """Get or create a learner profile."""
        if learner_id in self._cache:
            return self._cache[learner_id]

        file_path = self.storage_path / f"{learner_id}.json"
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
            profile = LearnerProfile.from_dict(data)
        else:
            profile = LearnerProfile(learner_id=learner_id)

        self._cache[learner_id] = profile
        return profile

    def save_profile(self, profile: LearnerProfile):
        """Save a learner profile to disk."""
        file_path = self.storage_path / f"{profile.learner_id}.json"
        with open(file_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        self._cache[profile.learner_id] = profile

    def delete_profile(self, learner_id: str):
        """Delete a learner profile."""
        file_path = self.storage_path / f"{learner_id}.json"
        if file_path.exists():
            file_path.unlink()
        self._cache.pop(learner_id, None)

    def list_profiles(self) -> list[str]:
        """List all learner profile IDs."""
        return [f.stem for f in self.storage_path.glob("*.json")]
