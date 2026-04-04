"""Semantic Analysis module for parsing learner queries.

Performs semantic parsing, entity extraction, and context management
using transformer-based NLU as described in the paper.
"""

import logging
import re
from dataclasses import dataclass, field

from .intent_classifier import IntentClassifier, LearnerIntent

logger = logging.getLogger(__name__)


# Domain-specific entity patterns
DOMAIN_ENTITIES = {
    "chemistry": {
        "topics": ["molecule", "atom", "electron", "proton", "neutron", "ion",
                    "bond", "covalent", "ionic", "metallic", "reaction", "acid",
                    "base", "ph", "periodic table", "element", "compound",
                    "solution", "solvent", "solute", "oxidation", "reduction",
                    "catalyst", "equilibrium", "photosynthesis", "organic",
                    "inorganic", "polymer", "isomer", "crystal"],
        "structures": ["water molecule", "benzene ring", "dna", "protein",
                        "amino acid", "glucose", "methane", "ethanol",
                        "carbon dioxide", "sodium chloride", "h2o", "co2",
                        "nacl", "hcl", "h2so4"],
    },
    "biology": {
        "topics": ["cell", "mitochondria", "nucleus", "ribosome", "chromosome",
                    "dna", "rna", "gene", "protein", "enzyme", "membrane",
                    "photosynthesis", "respiration", "mitosis", "meiosis",
                    "evolution", "ecosystem", "organism", "tissue", "organ"],
        "structures": ["human heart", "brain", "lung", "kidney", "liver",
                        "stomach", "eye", "ear", "skeleton", "muscle",
                        "neuron", "blood cell", "plant cell", "animal cell"],
    },
    "physics": {
        "topics": ["force", "gravity", "momentum", "energy", "velocity",
                    "acceleration", "mass", "wave", "frequency", "wavelength",
                    "electromagnetic", "quantum", "photon", "electron",
                    "magnetism", "electricity", "circuit", "resistance",
                    "thermodynamics", "entropy", "optics", "nuclear"],
        "structures": ["solar system", "atom", "planetary model", "wave",
                        "electric circuit", "magnetic field", "light spectrum",
                        "pendulum", "lever", "pulley", "inclined plane"],
    },
    "mathematics": {
        "topics": ["equation", "function", "variable", "derivative", "integral",
                    "limit", "matrix", "vector", "polynomial", "logarithm",
                    "trigonometry", "geometry", "algebra", "calculus",
                    "probability", "statistics", "set", "graph", "theorem"],
        "structures": ["parabola", "circle", "sphere", "cube", "cylinder",
                        "cone", "triangle", "pythagorean", "sine wave",
                        "coordinate plane", "unit circle", "number line"],
    },
    "anatomy": {
        "topics": ["skeletal", "muscular", "nervous", "cardiovascular",
                    "respiratory", "digestive", "endocrine", "immune",
                    "lymphatic", "urinary", "reproductive", "integumentary"],
        "structures": ["heart", "brain", "lungs", "spine", "skull", "femur",
                        "ribcage", "pelvis", "hand bones", "knee joint",
                        "hip joint", "shoulder", "elbow"],
    },
    "engineering": {
        "topics": ["stress", "strain", "torque", "friction", "elasticity",
                    "fluid dynamics", "thermodynamics", "circuit", "signal",
                    "control system", "mechanism", "gear", "bearing"],
        "structures": ["bridge", "truss", "beam", "gear system", "engine",
                        "turbine", "generator", "motor", "circuit board",
                        "heat exchanger", "hydraulic system"],
    },
}


@dataclass
class SemanticResult:
    """Result of semantic analysis on a learner query."""
    original_text: str
    intent: LearnerIntent
    intent_confidence: float
    domain: str | None
    topic: str | None
    entities: list[str]
    visualization_type: str | None
    complexity_hint: str  # "basic", "intermediate", "advanced"
    context_keywords: list[str]
    requires_followup: bool


@dataclass
class ConversationContext:
    """Maintains conversational continuity across interactions."""
    current_domain: str | None = None
    current_topic: str | None = None
    interaction_history: list[dict] = field(default_factory=list)
    topic_depth: int = 0  # How deep into a topic the learner has gone
    max_history: int = 20

    def update(self, result: SemanticResult):
        """Update context with new interaction."""
        if result.domain:
            self.current_domain = result.domain
        if result.topic:
            if result.topic == self.current_topic:
                self.topic_depth += 1
            else:
                self.current_topic = result.topic
                self.topic_depth = 0

        self.interaction_history.append({
            "text": result.original_text,
            "intent": result.intent.value,
            "domain": result.domain,
            "topic": result.topic,
        })

        if len(self.interaction_history) > self.max_history:
            self.interaction_history = self.interaction_history[-self.max_history:]


class SemanticAnalyzer:
    """Full semantic analysis pipeline combining intent classification,
    entity extraction, domain detection, and context management."""

    def __init__(self, confidence_threshold: float = 0.6):
        self.intent_classifier = IntentClassifier(
            confidence_threshold=confidence_threshold
        )
        self.context = ConversationContext()

    def load_models(self):
        """Load all NLU models."""
        self.intent_classifier.load_model()

    def analyze(self, text: str) -> SemanticResult:
        """Perform complete semantic analysis on learner query.

        This is the main entry point for the NLU pipeline:
        1. Classify intent
        2. Detect domain
        3. Extract entities and topics
        4. Determine visualization type
        5. Assess complexity
        6. Update conversation context
        """
        # Step 1: Intent classification
        intent_result = self.intent_classifier.classify(text)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # Step 2: Domain detection
        domain = self._detect_domain(text)

        # Step 3: Entity and topic extraction
        entities = self._extract_entities(text, domain)
        topic = self._extract_topic(text, domain, entities)

        # Step 4: Determine visualization type
        viz_type = self._determine_visualization_type(intent, text, domain)

        # Step 5: Assess complexity
        complexity = self._assess_complexity(text, intent)

        # Step 6: Extract context keywords
        keywords = self._extract_keywords(text)

        # Step 7: Check if follow-up is needed
        requires_followup = confidence < self.intent_classifier.confidence_threshold

        # Use context for missing domain/topic
        if domain is None and self.context.current_domain:
            domain = self.context.current_domain
        if topic is None and self.context.current_topic:
            topic = self.context.current_topic

        result = SemanticResult(
            original_text=text,
            intent=intent,
            intent_confidence=confidence,
            domain=domain,
            topic=topic,
            entities=entities,
            visualization_type=viz_type,
            complexity_hint=complexity,
            context_keywords=keywords,
            requires_followup=requires_followup,
        )

        # Update conversation context
        self.context.update(result)

        return result

    def _detect_domain(self, text: str) -> str | None:
        """Detect the educational domain from the query text."""
        text_lower = text.lower()
        domain_scores = {}

        for domain, data in DOMAIN_ENTITIES.items():
            score = 0
            all_terms = data["topics"] + data["structures"]
            for term in all_terms:
                if term.lower() in text_lower:
                    score += len(term.split())  # Weight multi-word matches higher
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return None

    def _extract_entities(self, text: str, domain: str | None) -> list[str]:
        """Extract domain-specific entities from the query."""
        text_lower = text.lower()
        entities = []

        domains_to_check = [domain] if domain else DOMAIN_ENTITIES.keys()

        for d in domains_to_check:
            if d not in DOMAIN_ENTITIES:
                continue
            data = DOMAIN_ENTITIES[d]
            for term in data["topics"] + data["structures"]:
                if term.lower() in text_lower and term not in entities:
                    entities.append(term)

        return entities

    def _extract_topic(self, text: str, domain: str | None,
                       entities: list[str]) -> str | None:
        """Extract the main topic from the query."""
        if entities:
            # Return the most specific (longest) entity as the topic
            return max(entities, key=len)

        # Try to extract topic from common query patterns
        patterns = [
            r"(?:about|of|is|are)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\?|$|\.)",
            r"(?:explain|describe|visualize|show)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\?|$|\.)",
            r"how\s+does\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+work",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 2 and len(topic) < 100:
                    return topic

        return None

    def _determine_visualization_type(self, intent: LearnerIntent,
                                       text: str, domain: str | None) -> str | None:
        """Determine the appropriate visualization type for the content."""
        text_lower = text.lower()

        if intent == LearnerIntent.VISUAL_REPRESENTATION:
            if any(kw in text_lower for kw in ["3d", "model", "structure", "shape"]):
                return "3d_model"
            if any(kw in text_lower for kw in ["animation", "animate", "motion", "movement"]):
                return "animation"
            if any(kw in text_lower for kw in ["diagram", "chart", "graph"]):
                return "contextual_overlay"
            return "3d_model"  # Default for visual requests

        if intent == LearnerIntent.PROCEDURAL_QUESTION:
            return "step_by_step"

        if intent == LearnerIntent.CONCEPTUAL_EXPLANATION:
            # Concepts with spatial components get 3D models
            if domain in ("chemistry", "biology", "anatomy", "physics"):
                return "3d_model"
            return "animation"

        return None

    def _assess_complexity(self, text: str, intent: LearnerIntent) -> str:
        """Assess the complexity level needed for the response."""
        text_lower = text.lower()

        advanced_indicators = [
            "advanced", "complex", "detailed", "in-depth", "mechanism",
            "molecular", "quantum", "differential", "integral",
            "thermodynamic", "electromagnetic",
        ]
        basic_indicators = [
            "simple", "basic", "easy", "introduction", "beginner",
            "what is", "overview",
        ]

        advanced_score = sum(1 for kw in advanced_indicators if kw in text_lower)
        basic_score = sum(1 for kw in basic_indicators if kw in text_lower)

        # Factor in topic depth from context
        depth_bonus = min(self.context.topic_depth * 0.3, 1.0)

        if intent == LearnerIntent.ELABORATION:
            depth_bonus += 0.5

        if advanced_score > basic_score or depth_bonus > 0.8:
            return "advanced"
        elif basic_score > advanced_score and depth_bonus < 0.3:
            return "basic"
        return "intermediate"

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract important keywords from the query."""
        # Remove common stop words and extract content words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "its", "this", "that", "these", "those", "i", "me",
            "my", "we", "our", "you", "your", "he", "she", "they",
            "what", "how", "why", "when", "where", "which", "who",
            "and", "or", "but", "not", "so", "if", "then", "than",
            "about", "tell", "show", "explain", "please", "help",
        }

        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]

        return list(dict.fromkeys(keywords))  # Deduplicate preserving order

    def reset_context(self):
        """Reset conversation context for a new session."""
        self.context = ConversationContext()
