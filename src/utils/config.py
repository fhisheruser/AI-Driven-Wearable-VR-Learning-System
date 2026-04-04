"""Configuration loader for the AI-VR Learning System."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ASRConfig(BaseModel):
    model_size: str = "base"
    language: str = "en"
    beam_size: int = 5
    word_error_rate_target: float = 0.05
    vad_threshold: float = 0.5


class NLUConfig(BaseModel):
    model_name: str = "distilbert-base-uncased"
    max_sequence_length: int = 128
    intent_confidence_threshold: float = 0.6
    supported_intents: list[str] = [
        "conceptual_explanation",
        "procedural_question",
        "visual_representation",
        "elaboration",
        "quiz_request",
        "topic_change",
    ]


class KnowledgeConfig(BaseModel):
    domains: list[str] = ["chemistry", "biology", "physics", "mathematics", "engineering", "anatomy"]
    curriculum_standards: list[str] = ["K12", "undergraduate", "graduate"]


class PersonalizationConfig(BaseModel):
    algorithm: str = "PPO"
    learning_rate: float = 0.0003
    gamma: float = 0.99
    clip_range: float = 0.2
    n_steps: int = 128
    batch_size: int = 64
    difficulty_levels: int = 5
    adaptation_rate: float = 0.1


class VisualizationConfig(BaseModel):
    target_fps: int = 30
    max_latency_ms: int = 2000
    render_quality: str = "medium"
    supported_types: list[str] = ["3d_model", "animation", "step_by_step", "contextual_overlay", "video"]


class DisplayConfig(BaseModel):
    fov_degrees: int = 52
    resolution: list[int] = [1280, 720]
    transparency: float = 0.85
    depth_range: list[float] = [0.5, 10.0]


class EvaluationConfig(BaseModel):
    participant_count: int = 60
    asr_accuracy_target: float = 0.95
    latency_target_seconds: float = 2.0
    fps_target: int = 30
    confidence_level: float = 0.95


class SystemConfig(BaseModel):
    asr: ASRConfig = ASRConfig()
    nlu: NLUConfig = NLUConfig()
    knowledge: KnowledgeConfig = KnowledgeConfig()
    personalization: PersonalizationConfig = PersonalizationConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    display: DisplayConfig = DisplayConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def load_config(config_path: str | None = None) -> SystemConfig:
    """Load system configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get(
            "CONFIG_PATH",
            str(Path(__file__).parent.parent.parent / "configs" / "system_config.yaml"),
        )

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return SystemConfig(**data)

    return SystemConfig()
