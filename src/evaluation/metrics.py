"""Evaluation metrics for the AI-Driven VR Learning System.

Implements the evaluation framework from the paper:
- ASR accuracy (target >= 95%)
- End-to-end response latency (target <= 2 seconds)
- Visualization frame rate (target >= 30 fps)
- Learning effectiveness via pre/post test scores
- Cognitive load assessment (NASA-TLX)
- Engagement metrics
- Statistical significance testing (paired t-test, 95% confidence)
"""

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ASRMetrics:
    """Speech recognition evaluation metrics."""
    total_words: int = 0
    correct_words: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def word_error_rate(self) -> float:
        if self.total_words == 0:
            return 0.0
        return (self.substitutions + self.insertions + self.deletions) / self.total_words

    @property
    def accuracy(self) -> float:
        return 1.0 - self.word_error_rate

    def update(self, reference: str, hypothesis: str):
        """Update metrics by comparing reference and hypothesis transcriptions."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        self.total_words += len(ref_words)

        # Simple word-level comparison using edit distance
        d = self._edit_distance(ref_words, hyp_words)
        s, i, dl = self._classify_errors(ref_words, hyp_words, d)
        self.substitutions += s
        self.insertions += i
        self.deletions += dl
        self.correct_words += len(ref_words) - s - dl

    def _edit_distance(self, ref: list[str], hyp: list[str]) -> list[list[int]]:
        n, m = len(ref), len(hyp)
        d = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        return d

    def _classify_errors(self, ref: list[str], hyp: list[str],
                          d: list[list[int]]) -> tuple[int, int, int]:
        i, j = len(ref), len(hyp)
        subs, ins, dels = 0, 0, 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
                subs += 1
                i -= 1
                j -= 1
            elif j > 0 and d[i][j] == d[i][j-1] + 1:
                ins += 1
                j -= 1
            else:
                dels += 1
                i -= 1
        return subs, ins, dels


@dataclass
class LatencyMetrics:
    """Response latency evaluation metrics."""
    measurements: list[float] = field(default_factory=list)
    target_ms: float = 2000.0

    def record(self, latency_ms: float):
        self.measurements.append(latency_ms)

    @property
    def mean(self) -> float:
        return sum(self.measurements) / len(self.measurements) if self.measurements else 0.0

    @property
    def p95(self) -> float:
        if not self.measurements:
            return 0.0
        sorted_m = sorted(self.measurements)
        idx = int(0.95 * len(sorted_m))
        return sorted_m[min(idx, len(sorted_m) - 1)]

    @property
    def within_target(self) -> float:
        if not self.measurements:
            return 0.0
        return sum(1 for m in self.measurements if m <= self.target_ms) / len(self.measurements)


@dataclass
class NASATLXScore:
    """NASA Task Load Index cognitive load assessment."""
    mental_demand: int = 0  # 0-100
    physical_demand: int = 0
    temporal_demand: int = 0
    performance: int = 0
    effort: int = 0
    frustration: int = 0

    @property
    def overall_score(self) -> float:
        return (self.mental_demand + self.physical_demand + self.temporal_demand +
                self.performance + self.effort + self.frustration) / 6.0


@dataclass
class LearningEffectiveness:
    """Pre/post test learning effectiveness metrics."""
    pre_scores: list[float] = field(default_factory=list)
    post_scores: list[float] = field(default_factory=list)

    @property
    def mean_gain(self) -> float:
        if not self.pre_scores or not self.post_scores:
            return 0.0
        n = min(len(self.pre_scores), len(self.post_scores))
        gains = [self.post_scores[i] - self.pre_scores[i] for i in range(n)]
        return sum(gains) / n

    def paired_t_test(self) -> dict:
        """Perform paired t-test on pre/post scores (95% confidence).

        Returns t-statistic, p-value estimate, and significance.
        """
        n = min(len(self.pre_scores), len(self.post_scores))
        if n < 2:
            return {"t_statistic": 0.0, "significant": False, "n": n}

        diffs = [self.post_scores[i] - self.pre_scores[i] for i in range(n)]
        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
        std_d = math.sqrt(var_d) if var_d > 0 else 0.001

        t_stat = mean_d / (std_d / math.sqrt(n))

        # Approximate p-value using t-distribution
        # For df >= 30, t > 2.0 is significant at 0.05 level
        df = n - 1
        t_critical = 2.045 if df < 30 else 1.96
        significant = abs(t_stat) > t_critical

        return {
            "t_statistic": round(t_stat, 4),
            "degrees_of_freedom": df,
            "mean_gain": round(mean_d, 4),
            "std_gain": round(std_d, 4),
            "significant_at_95": significant,
            "n": n,
        }


class EvaluationEngine:
    """Comprehensive evaluation engine implementing the paper's framework.

    Tracks:
    - ASR accuracy (target >= 95%)
    - End-to-end latency (target <= 2s)
    - Learning effectiveness (pre/post test)
    - Cognitive load (NASA-TLX)
    - Engagement metrics
    """

    def __init__(self):
        self.asr_metrics = ASRMetrics()
        self.latency_metrics = LatencyMetrics()
        self.learning_effectiveness = LearningEffectiveness()
        self.nasa_tlx_scores: list[NASATLXScore] = []
        self.engagement_data: list[dict] = []

    def record_asr_result(self, reference: str, hypothesis: str):
        """Record an ASR evaluation result."""
        self.asr_metrics.update(reference, hypothesis)

    def record_latency(self, latency_ms: float):
        """Record a pipeline response latency."""
        self.latency_metrics.record(latency_ms)

    def record_learning_scores(self, pre_score: float, post_score: float):
        """Record pre/post test scores for a participant."""
        self.learning_effectiveness.pre_scores.append(pre_score)
        self.learning_effectiveness.post_scores.append(post_score)

    def record_nasa_tlx(self, scores: dict):
        """Record a NASA-TLX cognitive load assessment."""
        self.nasa_tlx_scores.append(NASATLXScore(**scores))

    def record_engagement(self, data: dict):
        """Record engagement metrics (interaction duration, query frequency, gaze)."""
        self.engagement_data.append(data)

    def generate_report(self) -> dict:
        """Generate a comprehensive evaluation report."""
        report = {
            "asr": {
                "accuracy": round(self.asr_metrics.accuracy, 4),
                "word_error_rate": round(self.asr_metrics.word_error_rate, 4),
                "target_met": self.asr_metrics.accuracy >= 0.95,
                "total_words_evaluated": self.asr_metrics.total_words,
            },
            "latency": {
                "mean_ms": round(self.latency_metrics.mean, 2),
                "p95_ms": round(self.latency_metrics.p95, 2),
                "within_target_rate": round(self.latency_metrics.within_target, 4),
                "target_met": self.latency_metrics.mean <= 2000,
                "total_measurements": len(self.latency_metrics.measurements),
            },
            "learning_effectiveness": {
                "mean_gain": round(self.learning_effectiveness.mean_gain, 4),
                "statistical_test": self.learning_effectiveness.paired_t_test(),
                "participants": min(
                    len(self.learning_effectiveness.pre_scores),
                    len(self.learning_effectiveness.post_scores),
                ),
            },
        }

        if self.nasa_tlx_scores:
            avg_tlx = sum(s.overall_score for s in self.nasa_tlx_scores) / len(self.nasa_tlx_scores)
            report["cognitive_load"] = {
                "mean_nasa_tlx": round(avg_tlx, 2),
                "assessments": len(self.nasa_tlx_scores),
            }

        if self.engagement_data:
            durations = [d.get("interaction_duration", 0) for d in self.engagement_data]
            queries = [d.get("query_count", 0) for d in self.engagement_data]
            report["engagement"] = {
                "avg_interaction_duration": round(sum(durations) / len(durations), 2) if durations else 0,
                "avg_query_count": round(sum(queries) / len(queries), 2) if queries else 0,
                "sessions": len(self.engagement_data),
            }

        return report
