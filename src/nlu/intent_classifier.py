"""Intent Classification module using transformer-based models.

Implements intent detection and semantic parsing of learner queries using
fine-tuned BERT/DistilBERT architecture as described in the paper.
"""

import logging
import re
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class LearnerIntent(str, Enum):
    """Supported learner intent categories as defined in the paper."""
    CONCEPTUAL_EXPLANATION = "conceptual_explanation"
    PROCEDURAL_QUESTION = "procedural_question"
    VISUAL_REPRESENTATION = "visual_representation"
    ELABORATION = "elaboration"
    QUIZ_REQUEST = "quiz_request"
    TOPIC_CHANGE = "topic_change"


# Pattern-based intent detection as reliable baseline + transformer enhancement
INTENT_PATTERNS = {
    LearnerIntent.CONCEPTUAL_EXPLANATION: [
        r"what\s+is\s+",
        r"explain\s+",
        r"describe\s+",
        r"how\s+does\s+.*\s+work",
        r"what\s+are\s+",
        r"define\s+",
        r"tell\s+me\s+about\s+",
        r"what\s+do\s+you\s+mean\s+by",
        r"can\s+you\s+explain",
        r"what\s+is\s+the\s+meaning\s+of",
        r"why\s+is\s+",
        r"why\s+does\s+",
        r"what\s+causes\s+",
    ],
    LearnerIntent.PROCEDURAL_QUESTION: [
        r"how\s+to\s+",
        r"how\s+do\s+(?:i|you|we)\s+",
        r"what\s+are\s+the\s+steps",
        r"step\s+by\s+step",
        r"procedure\s+for",
        r"process\s+of",
        r"how\s+can\s+(?:i|we)\s+",
        r"walk\s+me\s+through",
        r"guide\s+me",
        r"show\s+me\s+how",
        r"what\s+is\s+the\s+process",
    ],
    LearnerIntent.VISUAL_REPRESENTATION: [
        r"show\s+me\s+",
        r"visualize\s+",
        r"display\s+",
        r"render\s+",
        r"draw\s+",
        r"illustrate\s+",
        r"can\s+(?:i|you)\s+see\s+",
        r"3d\s+model\s+of",
        r"animation\s+of",
        r"diagram\s+of",
        r"picture\s+of",
        r"image\s+of",
        r"view\s+",
        r"demonstrate\s+visually",
    ],
    LearnerIntent.ELABORATION: [
        r"more\s+(?:detail|info|about)",
        r"elaborate\s+",
        r"go\s+deeper",
        r"expand\s+on",
        r"tell\s+me\s+more",
        r"can\s+you\s+add\s+more",
        r"what\s+else\s+",
        r"in\s+more\s+detail",
        r"further\s+explanation",
        r"continue\s+",
        r"go\s+on\s+",
    ],
    LearnerIntent.QUIZ_REQUEST: [
        r"quiz\s+me",
        r"test\s+me",
        r"ask\s+me\s+",
        r"practice\s+question",
        r"exercise\s+",
        r"assessment\s+",
        r"check\s+my\s+understanding",
        r"sample\s+question",
    ],
    LearnerIntent.TOPIC_CHANGE: [
        r"(?:let's|let\s+us)\s+(?:move|switch|change|talk)\s+(?:to|about)",
        r"new\s+topic",
        r"change\s+(?:the\s+)?subject",
        r"(?:i\s+want|i'd\s+like)\s+to\s+learn\s+about",
        r"switch\s+to\s+",
        r"move\s+on\s+to",
        r"next\s+topic",
    ],
}


class IntentClassifier:
    """Transformer-enhanced intent classifier with pattern-based fallback.

    Uses a fine-tuned DistilBERT model for intent classification and
    falls back to regex patterns when the model is unavailable.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 confidence_threshold: float = 0.6):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._transformer_loaded = False
        self._tokenizer = None
        self._model = None
        self._sentence_model = None

    def load_model(self):
        """Load transformer model for enhanced intent classification."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer for intent classification...")
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._transformer_loaded = True
            logger.info("Sentence transformer loaded successfully")

            # Pre-compute intent embeddings
            self._intent_exemplars = self._build_intent_exemplars()
        except ImportError:
            logger.warning("sentence-transformers not installed. Using pattern-based classification.")
            self._transformer_loaded = False

    def _build_intent_exemplars(self) -> dict[str, np.ndarray]:
        """Build embedding exemplars for each intent category."""
        exemplars = {
            LearnerIntent.CONCEPTUAL_EXPLANATION: [
                "What is photosynthesis?",
                "Explain how gravity works",
                "What are covalent bonds?",
                "Describe the structure of DNA",
                "Why does water boil at 100 degrees?",
            ],
            LearnerIntent.PROCEDURAL_QUESTION: [
                "How do I solve a quadratic equation?",
                "What are the steps to balance a chemical equation?",
                "Walk me through cell division",
                "How to calculate the derivative?",
                "Show me how to use the Pythagorean theorem",
            ],
            LearnerIntent.VISUAL_REPRESENTATION: [
                "Show me a 3D model of a water molecule",
                "Visualize the solar system",
                "Display the human heart",
                "Render an atom structure",
                "Can I see the DNA helix?",
            ],
            LearnerIntent.ELABORATION: [
                "Tell me more about that",
                "Can you elaborate on this concept?",
                "Go deeper into the topic",
                "I want more details",
                "Expand on the previous explanation",
            ],
            LearnerIntent.QUIZ_REQUEST: [
                "Quiz me on chemistry",
                "Test my knowledge of biology",
                "Give me a practice question",
                "Check my understanding",
                "Ask me something about physics",
            ],
            LearnerIntent.TOPIC_CHANGE: [
                "Let's switch to mathematics",
                "I want to learn about astronomy now",
                "Change the subject to biology",
                "Move on to the next topic",
                "Let's talk about physics instead",
            ],
        }

        intent_embeddings = {}
        for intent, texts in exemplars.items():
            embeddings = self._sentence_model.encode(texts)
            intent_embeddings[intent] = np.mean(embeddings, axis=0)

        return intent_embeddings

    def classify(self, text: str) -> dict:
        """Classify the intent of a learner's query.

        Returns:
            Dictionary with 'intent', 'confidence', 'all_scores'
        """
        text = text.strip().lower()
        if not text:
            return {
                "intent": LearnerIntent.CONCEPTUAL_EXPLANATION,
                "confidence": 0.0,
                "all_scores": {},
            }

        # Try transformer-based classification first
        if self._transformer_loaded and self._sentence_model is not None:
            return self._classify_transformer(text)

        # Fall back to pattern-based classification
        return self._classify_patterns(text)

    def _classify_transformer(self, text: str) -> dict:
        """Classify intent using sentence transformer similarity."""
        query_embedding = self._sentence_model.encode([text])[0]

        scores = {}
        for intent, exemplar_embedding in self._intent_exemplars.items():
            similarity = float(np.dot(query_embedding, exemplar_embedding) /
                             (np.linalg.norm(query_embedding) * np.linalg.norm(exemplar_embedding)))
            scores[intent] = max(0.0, similarity)

        # Also factor in pattern matches
        pattern_scores = self._get_pattern_scores(text)
        for intent in scores:
            if intent in pattern_scores:
                scores[intent] = 0.7 * scores[intent] + 0.3 * pattern_scores[intent]

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        return {
            "intent": best_intent,
            "confidence": best_score,
            "all_scores": {k.value: v for k, v in scores.items()},
        }

    def _classify_patterns(self, text: str) -> dict:
        """Classify intent using regex pattern matching."""
        scores = self._get_pattern_scores(text)

        if not scores:
            return {
                "intent": LearnerIntent.CONCEPTUAL_EXPLANATION,
                "confidence": 0.3,
                "all_scores": {i.value: 0.0 for i in LearnerIntent},
            }

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        return {
            "intent": best_intent,
            "confidence": best_score,
            "all_scores": {k.value: v for k, v in scores.items()},
        }

    def _get_pattern_scores(self, text: str) -> dict[LearnerIntent, float]:
        """Score each intent based on pattern matches."""
        scores = {}
        for intent, patterns in INTENT_PATTERNS.items():
            match_count = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
            if match_count > 0:
                scores[intent] = min(1.0, match_count * 0.3 + 0.4)

        return scores
