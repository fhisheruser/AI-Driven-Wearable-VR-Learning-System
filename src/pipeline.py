"""Main Pipeline Orchestrator for the AI-Driven VR Learning System.

Chains all modules together following the paper's workflow:
1. Student asks a question (speech input)
2. Speech recognition converts audio to text (ASR)
3. AI model performs semantic analysis (NLU)
4. Relevant educational content is retrieved/generated (Knowledge Engine)
5. VR visualization is rendered (Visualization Engine)
6. Content is projected into student's field of view (Display)
7. Feedback and follow-up interaction supported (Adaptive Engine)
"""

import logging
import time
from dataclasses import dataclass, field

from .asr.speech_recognizer import SpeechRecognizer
from .nlu.semantic_analyzer import SemanticAnalyzer, SemanticResult
from .knowledge.knowledge_engine import KnowledgeReasoningEngine, ReasonedOutput
from .personalization.adaptive_engine import AdaptiveLearningEngine, AdaptationAction
from .personalization.learner_profile import LearnerProfile, LearnerProfileManager
from .visualization.visualization_engine import VisualizationEngine, VisualizationScene
from .feedback.feedback_collector import FeedbackCollector, FeedbackSignal
from .utils.config import SystemConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineResponse:
    """Complete response from the learning pipeline."""
    query_text: str
    semantic_result: SemanticResult
    reasoned_output: ReasonedOutput
    adaptation: AdaptationAction
    scene: VisualizationScene
    total_latency_ms: float
    asr_latency_ms: float = 0.0
    pipeline_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query_text": self.query_text,
            "semantic": {
                "intent": self.semantic_result.intent.value,
                "intent_confidence": self.semantic_result.intent_confidence,
                "domain": self.semantic_result.domain,
                "topic": self.semantic_result.topic,
                "entities": self.semantic_result.entities,
                "visualization_type": self.semantic_result.visualization_type,
                "complexity": self.semantic_result.complexity_hint,
            },
            "knowledge": {
                "explanation": self.reasoned_output.explanation,
                "content_depth": self.reasoned_output.content_depth,
                "visualization_type": self.reasoned_output.visualization_type,
                "difficulty_level": self.reasoned_output.difficulty_level,
                "pedagogical_sequence": self.reasoned_output.pedagogical_sequence,
                "related_topics": self.reasoned_output.related_topics,
                "quiz_questions": self.reasoned_output.quiz_questions,
                "steps": self.reasoned_output.steps,
            },
            "adaptation": {
                "difficulty_adjustment": self.adaptation.difficulty_adjustment,
                "content_depth": self.adaptation.content_depth,
                "visualization_complexity": self.adaptation.visualization_complexity,
                "pacing": self.adaptation.pacing,
                "reinforcement_strategy": self.adaptation.reinforcement_strategy,
                "explanation_style": self.adaptation.explanation_style,
            },
            "scene": self.scene.to_dict(),
            "latency": {
                "total_ms": self.total_latency_ms,
                "asr_ms": self.asr_latency_ms,
            },
            "metadata": self.pipeline_metadata,
        }


class LearningPipeline:
    """End-to-end pipeline orchestrating all system modules.

    Implements the algorithm from the paper:
    - Input acquisition and speech recognition
    - Semantic analysis and intent classification
    - Adaptive reasoning based on learner profile
    - VR/AR content generation
    - Feedback integration and continuous adaptation
    """

    def __init__(self, config: SystemConfig | None = None):
        self.config = config or load_config()

        # Initialize all modules
        self.speech_recognizer = SpeechRecognizer(
            model_size=self.config.asr.model_size,
            language=self.config.asr.language,
            beam_size=self.config.asr.beam_size,
        )
        self.semantic_analyzer = SemanticAnalyzer(
            confidence_threshold=self.config.nlu.intent_confidence_threshold,
        )
        self.knowledge_engine = KnowledgeReasoningEngine()
        self.adaptive_engine = AdaptiveLearningEngine(
            learning_rate=self.config.personalization.learning_rate,
            gamma=self.config.personalization.gamma,
        )
        self.visualization_engine = VisualizationEngine()
        self.profile_manager = LearnerProfileManager()
        self.feedback_collector = FeedbackCollector()

        self._initialized = False

    def initialize(self):
        """Load all models and prepare the pipeline."""
        logger.info("Initializing AI Learning Pipeline...")
        self.speech_recognizer.load_model()
        self.semantic_analyzer.load_models()
        self._initialized = True
        logger.info("Pipeline initialized successfully")

    def process_text(self, text: str, learner_id: str = "default") -> PipelineResponse:
        """Process a text query through the full pipeline.

        This is the main entry point for text-based interaction,
        bypassing the ASR module.
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Get learner profile
        profile = self.profile_manager.get_profile(learner_id)

        # Step 1: Semantic analysis
        semantic_result = self.semantic_analyzer.analyze(text)
        logger.info(f"Intent: {semantic_result.intent.value}, "
                     f"Domain: {semantic_result.domain}, "
                     f"Topic: {semantic_result.topic}")

        # Step 2: Knowledge reasoning
        learner_data = {
            "comprehension_level": profile.overall_comprehension,
            "skill_level": profile.skill_level,
        }
        reasoned_output = self.knowledge_engine.reason(semantic_result, learner_data)

        # Step 3: Adaptive personalization
        adaptation = self.adaptive_engine.get_adaptation(profile)

        # Step 4: Generate visualization
        concept_data = None
        if reasoned_output.concept:
            concept_data = {
                "name": reasoned_output.concept.name,
                "domain": reasoned_output.concept.domain,
                "description": reasoned_output.concept.description,
            }
        scene = self.visualization_engine.generate_scene(
            reasoned_output.visualization_params, concept_data
        )

        # Step 5: Update learner profile
        elapsed = time.time() - start_time
        profile.update_interaction(
            domain=semantic_result.domain or "general",
            topic=semantic_result.topic or text[:50],
            intent=semantic_result.intent.value,
            duration=elapsed,
            comprehension_signal=0.5,
        )
        self.profile_manager.save_profile(profile)

        total_latency = (time.time() - start_time) * 1000

        return PipelineResponse(
            query_text=text,
            semantic_result=semantic_result,
            reasoned_output=reasoned_output,
            adaptation=adaptation,
            scene=scene,
            total_latency_ms=total_latency,
            pipeline_metadata={
                "learner_id": learner_id,
                "models_loaded": self._initialized,
            },
        )

    def process_audio(self, audio_bytes: bytes, learner_id: str = "default",
                      sample_rate: int = 16000) -> PipelineResponse:
        """Process audio input through the full pipeline (ASR + NLU + Knowledge + Viz)."""
        if not self._initialized:
            self.initialize()

        # Step 1: Speech recognition
        asr_result = self.speech_recognizer.transcribe_bytes(audio_bytes, sample_rate)

        if not asr_result["is_speech"] or not asr_result["text"]:
            # Return empty response for non-speech
            empty_response = self.process_text("", learner_id)
            empty_response.asr_latency_ms = asr_result["latency_ms"]
            return empty_response

        # Step 2-6: Process recognized text through the rest of the pipeline
        response = self.process_text(asr_result["text"], learner_id)
        response.asr_latency_ms = asr_result["latency_ms"]
        return response

    def process_feedback(self, learner_id: str, feedback_data: dict) -> dict:
        """Process learner feedback for adaptive learning.

        Args:
            learner_id: The learner's ID
            feedback_data: Feedback signals (comprehension_change, engagement,
                          quiz_score, interaction_time, verbal_feedback)
        """
        profile = self.profile_manager.get_profile(learner_id)

        # Collect and analyze feedback
        signal = self.feedback_collector.collect(feedback_data)

        # Update adaptive engine
        rl_feedback = {
            "comprehension_change": signal.comprehension_change,
            "engagement": signal.engagement_level,
            "quiz_score": signal.quiz_score,
            "interaction_time": signal.interaction_time,
        }
        metrics = self.adaptive_engine.process_feedback(rl_feedback)

        # Update learner profile with quiz results
        if signal.quiz_score is not None and signal.domain:
            profile.record_quiz_result(signal.domain, signal.quiz_score)
            self.profile_manager.save_profile(profile)

        return {
            "feedback_processed": True,
            "signal": signal.to_dict(),
            "rl_metrics": metrics,
        }

    def reset_session(self, learner_id: str = "default"):
        """Reset conversation context for a new learning session."""
        self.semantic_analyzer.reset_context()
        profile = self.profile_manager.get_profile(learner_id)
        self.adaptive_engine.initialize_session(profile)

    def get_learner_stats(self, learner_id: str) -> dict:
        """Get statistics for a learner."""
        profile = self.profile_manager.get_profile(learner_id)
        return profile.to_dict()
