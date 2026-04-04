"""Knowledge Reasoning Engine for educational content retrieval and generation.

Maps identified intent to structured educational content using domain ontology
aligned with standard curricula as described in the paper. Determines appropriate
content depth, visualization complexity, and pedagogical sequencing.
"""

import logging
from dataclasses import dataclass, field

from ..nlu.intent_classifier import LearnerIntent
from ..nlu.semantic_analyzer import SemanticResult
from .domain_ontology import Concept, DomainOntology

logger = logging.getLogger(__name__)


@dataclass
class ReasonedOutput:
    """Output from the knowledge reasoning engine."""
    concept: Concept | None
    explanation: str
    content_depth: str  # "overview", "standard", "detailed", "expert"
    visualization_type: str  # "3d_model", "animation", "step_by_step", "contextual_overlay"
    visualization_params: dict = field(default_factory=dict)
    pedagogical_sequence: list[str] = field(default_factory=list)
    difficulty_level: int = 1
    related_topics: list[str] = field(default_factory=list)
    quiz_questions: list[dict] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


class KnowledgeReasoningEngine:
    """Integrates domain knowledge, curriculum alignment, and learner-specific
    data to generate appropriate educational content."""

    def __init__(self):
        self.ontology = DomainOntology()

    def reason(self, semantic_result: SemanticResult,
               learner_profile: dict | None = None) -> ReasonedOutput:
        """Process semantic analysis result and generate reasoned educational content.

        This implements the adaptive reasoning described in the paper:
        - Determines appropriate content depth
        - Selects visualization complexity
        - Plans pedagogical sequencing
        """
        # Find the most relevant concept
        concept = self.ontology.find_concept(
            semantic_result.original_text,
            domain=semantic_result.domain,
        )

        # Determine content depth based on learner profile and complexity hint
        content_depth = self._determine_content_depth(
            semantic_result, learner_profile
        )

        # Determine difficulty level
        difficulty = self._determine_difficulty(
            semantic_result, learner_profile, concept
        )

        # Select visualization type
        viz_type = self._select_visualization(semantic_result, concept)

        # Generate visualization parameters
        viz_params = self._generate_visualization_params(
            viz_type, concept, semantic_result
        )

        # Build pedagogical sequence
        sequence = self._build_pedagogical_sequence(
            concept, semantic_result.intent, content_depth
        )

        # Generate explanation
        explanation = self._generate_explanation(
            concept, semantic_result, content_depth
        )

        # Generate steps if procedural
        steps = self._generate_steps(concept, semantic_result)

        # Find related topics
        related = self._find_related_topics(concept)

        # Generate quiz questions if requested
        quiz = []
        if semantic_result.intent == LearnerIntent.QUIZ_REQUEST:
            quiz = self._generate_quiz(concept, difficulty)

        return ReasonedOutput(
            concept=concept,
            explanation=explanation,
            content_depth=content_depth,
            visualization_type=viz_type,
            visualization_params=viz_params,
            pedagogical_sequence=sequence,
            difficulty_level=difficulty,
            related_topics=related,
            quiz_questions=quiz,
            steps=steps,
        )

    def _determine_content_depth(self, semantic_result: SemanticResult,
                                  learner_profile: dict | None) -> str:
        """Determine appropriate content depth based on learner context."""
        complexity = semantic_result.complexity_hint

        # Check learner profile for historical performance
        if learner_profile:
            comprehension = learner_profile.get("comprehension_level", 0.5)
            if comprehension > 0.8:
                depth_boost = 1
            elif comprehension < 0.3:
                depth_boost = -1
            else:
                depth_boost = 0
        else:
            depth_boost = 0

        depth_map = {
            "basic": ["overview", "standard", "standard"],
            "intermediate": ["standard", "standard", "detailed"],
            "advanced": ["standard", "detailed", "expert"],
        }

        depths = depth_map.get(complexity, ["standard", "standard", "standard"])
        index = max(0, min(2, 1 + depth_boost))
        return depths[index]

    def _determine_difficulty(self, semantic_result: SemanticResult,
                               learner_profile: dict | None,
                               concept: Concept | None) -> int:
        """Determine appropriate difficulty level (1-5)."""
        base_difficulty = concept.difficulty_level if concept else 2

        if learner_profile:
            learner_level = learner_profile.get("skill_level", 2)
            # Blend concept difficulty with learner level
            difficulty = int(0.6 * base_difficulty + 0.4 * learner_level)
        else:
            difficulty = base_difficulty

        # Adjust based on complexity hint
        if semantic_result.complexity_hint == "advanced":
            difficulty = min(5, difficulty + 1)
        elif semantic_result.complexity_hint == "basic":
            difficulty = max(1, difficulty - 1)

        return max(1, min(5, difficulty))

    def _select_visualization(self, semantic_result: SemanticResult,
                               concept: Concept | None) -> str:
        """Select the most appropriate visualization type."""
        # Use semantic analysis suggestion first
        if semantic_result.visualization_type:
            return semantic_result.visualization_type

        # Use concept's hint
        if concept:
            return concept.visualization_hint

        # Default based on intent
        intent_viz_map = {
            LearnerIntent.CONCEPTUAL_EXPLANATION: "3d_model",
            LearnerIntent.PROCEDURAL_QUESTION: "step_by_step",
            LearnerIntent.VISUAL_REPRESENTATION: "3d_model",
            LearnerIntent.ELABORATION: "animation",
            LearnerIntent.QUIZ_REQUEST: "contextual_overlay",
            LearnerIntent.TOPIC_CHANGE: "3d_model",
        }

        return intent_viz_map.get(semantic_result.intent, "3d_model")

    def _generate_visualization_params(self, viz_type: str,
                                        concept: Concept | None,
                                        semantic_result: SemanticResult) -> dict:
        """Generate parameters for the visualization engine."""
        params = {
            "type": viz_type,
            "quality": "medium",
            "interactive": True,
            "labels": True,
        }

        if concept:
            params["title"] = concept.name
            params["domain"] = concept.domain
            params["concept_id"] = concept.id

            # Domain-specific visualization parameters
            if concept.domain == "chemistry":
                params.update({
                    "show_bonds": True,
                    "show_electrons": True,
                    "atom_style": "ball_and_stick",
                    "color_scheme": "element",
                })
            elif concept.domain == "biology":
                params.update({
                    "cross_section": False,
                    "show_labels": True,
                    "transparency": 0.3,
                    "highlight_parts": True,
                })
            elif concept.domain == "physics":
                params.update({
                    "show_forces": True,
                    "show_vectors": True,
                    "animate": True,
                    "time_scale": 1.0,
                })
            elif concept.domain == "anatomy":
                params.update({
                    "cross_section": True,
                    "show_labels": True,
                    "layer_mode": "all",
                    "color_coded": True,
                })
            elif concept.domain == "mathematics":
                params.update({
                    "show_grid": True,
                    "show_axes": True,
                    "animate_proof": True,
                    "step_through": True,
                })
            elif concept.domain == "engineering":
                params.update({
                    "show_stress": True,
                    "show_dimensions": True,
                    "exploded_view": False,
                    "animate_mechanism": True,
                })

        if viz_type == "step_by_step":
            params["auto_advance"] = False
            params["step_duration_ms"] = 3000

        return params

    def _build_pedagogical_sequence(self, concept: Concept | None,
                                     intent: LearnerIntent,
                                     content_depth: str) -> list[str]:
        """Plan the pedagogical sequence for content delivery."""
        sequence = []

        if concept:
            # Add prerequisites first
            prereqs = self.ontology.get_prerequisites(concept.id)
            if prereqs and content_depth in ("detailed", "expert"):
                sequence.append(f"Review prerequisite: {prereqs[0].name}")

            # Introduction
            sequence.append(f"Introduction to {concept.name}")

            # Core content based on intent
            if intent == LearnerIntent.CONCEPTUAL_EXPLANATION:
                sequence.append(f"Core explanation of {concept.name}")
                if content_depth in ("detailed", "expert"):
                    sequence.append("Detailed mechanisms and properties")
                sequence.append("Visual demonstration")
            elif intent == LearnerIntent.PROCEDURAL_QUESTION:
                sequence.append("Step-by-step procedure")
                sequence.append("Visual walkthrough")
                sequence.append("Practice exercise")
            elif intent == LearnerIntent.VISUAL_REPRESENTATION:
                sequence.append("3D visualization")
                sequence.append("Interactive exploration")
                sequence.append("Key feature annotation")

            # Summary
            sequence.append("Summary and key takeaways")

            # Related topics suggestion
            related = self.ontology.get_related(concept.id)
            if related:
                sequence.append(f"Explore related: {related[0].name}")
        else:
            sequence = [
                "Topic introduction",
                "Core explanation",
                "Visual demonstration",
                "Summary",
            ]

        return sequence

    def _generate_explanation(self, concept: Concept | None,
                               semantic_result: SemanticResult,
                               content_depth: str) -> str:
        """Generate an educational explanation."""
        if concept and concept.explanation:
            explanation = concept.explanation

            if content_depth == "overview":
                # Truncate to first sentence
                sentences = explanation.split(". ")
                explanation = sentences[0] + "."
            elif content_depth == "expert":
                # Add technical detail
                explanation += f" This concept is fundamental in {concept.domain} "
                explanation += f"and relates to: {', '.join(concept.keywords[:5])}."

            return explanation

        # Generate a generic response when concept is not found
        topic = semantic_result.topic or "the requested topic"
        return (
            f"Let me explain {topic}. This is an important concept "
            f"in {semantic_result.domain or 'this field'}. "
            f"I'll provide a visualization to help you understand it better."
        )

    def _generate_steps(self, concept: Concept | None,
                        semantic_result: SemanticResult) -> list[str]:
        """Generate procedural steps if applicable."""
        if concept and concept.steps:
            return concept.steps

        if semantic_result.intent == LearnerIntent.PROCEDURAL_QUESTION:
            return [
                "Understand the fundamental concept",
                "Identify the key components",
                "Apply the relevant principles",
                "Verify your understanding",
            ]

        return []

    def _find_related_topics(self, concept: Concept | None) -> list[str]:
        """Find related topics for further exploration."""
        if not concept:
            return []

        related = self.ontology.get_related(concept.id)
        return [c.name for c in related]

    def _generate_quiz(self, concept: Concept | None,
                       difficulty: int) -> list[dict]:
        """Generate quiz questions for knowledge assessment."""
        if not concept:
            return []

        questions = []

        # Generate conceptual question
        questions.append({
            "type": "multiple_choice",
            "question": f"Which of the following best describes {concept.name}?",
            "options": [
                concept.description,
                f"A type of {concept.domain} measurement",
                f"An instrument used in {concept.domain}",
                f"A theoretical framework in {concept.domain}",
            ],
            "correct_answer": 0,
            "difficulty": difficulty,
            "explanation": concept.explanation[:100] + "..." if concept.explanation else "",
        })

        # Generate keyword-based question
        if len(concept.keywords) >= 3:
            questions.append({
                "type": "multiple_choice",
                "question": f"Which term is most closely related to {concept.name}?",
                "options": concept.keywords[:4] if len(concept.keywords) >= 4
                           else concept.keywords + ["entropy"],
                "correct_answer": 0,
                "difficulty": difficulty,
                "explanation": f"{concept.keywords[0]} is a key aspect of {concept.name}.",
            })

        return questions
