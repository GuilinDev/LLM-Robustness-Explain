"""
Robustness Metrics for LLM-Generated Recommendation Explanations.

Evaluates how stable explanations are under user behavior perturbations.
Inspired by XAI robustness evaluation in image classification domain.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import re
import json


@dataclass
class RobustnessResult:
    """Result of robustness evaluation."""
    semantic_similarity: float  # 0-1, higher is more robust
    keyword_stability: float  # Jaccard coefficient of key terms
    structural_consistency: float  # Explanation structure similarity
    length_stability: float  # Length variation ratio
    overall_robustness: float  # Weighted average
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            'semantic_similarity': self.semantic_similarity,
            'keyword_stability': self.keyword_stability,
            'structural_consistency': self.structural_consistency,
            'length_stability': self.length_stability,
            'overall_robustness': self.overall_robustness,
            'details': self.details,
        }


class TextSimilarity:
    """Text similarity metrics without external dependencies."""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    @staticmethod
    def get_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        """Get n-grams from tokens."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def cosine_similarity_bow(text1: str, text2: str) -> float:
        """Compute cosine similarity using bag-of-words."""
        tokens1 = TextSimilarity.tokenize(text1)
        tokens2 = TextSimilarity.tokenize(text2)

        # Build vocabulary
        vocab = set(tokens1) | set(tokens2)
        if not vocab:
            return 1.0

        # Build vectors
        vec1 = Counter(tokens1)
        vec2 = Counter(tokens2)

        # Compute dot product and magnitudes
        dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in vocab)
        mag1 = np.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = np.sqrt(sum(v ** 2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    @staticmethod
    def bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
        """Simplified BLEU score calculation."""
        ref_tokens = TextSimilarity.tokenize(reference)
        cand_tokens = TextSimilarity.tokenize(candidate)

        if len(cand_tokens) == 0:
            return 0.0

        # Calculate precision for each n-gram order
        precisions = []
        for n in range(1, min(max_n + 1, len(cand_tokens) + 1)):
            ref_ngrams = Counter(TextSimilarity.get_ngrams(ref_tokens, n))
            cand_ngrams = Counter(TextSimilarity.get_ngrams(cand_tokens, n))

            overlap = sum((ref_ngrams & cand_ngrams).values())
            total = sum(cand_ngrams.values())

            if total > 0:
                precisions.append(overlap / total)
            else:
                precisions.append(0.0)

        if not precisions or all(p == 0 for p in precisions):
            return 0.0

        # Geometric mean of precisions
        log_precisions = [np.log(p) if p > 0 else -np.inf for p in precisions]
        avg_log_precision = np.mean([lp for lp in log_precisions if lp > -np.inf])

        if avg_log_precision == -np.inf:
            return 0.0

        # Brevity penalty
        bp = 1.0
        if len(cand_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(cand_tokens))

        return bp * np.exp(avg_log_precision)


class RobustnessEvaluator:
    """
    Evaluates robustness of LLM-generated explanations.

    Compares explanations generated from original vs perturbed user histories.
    """

    # Keywords indicating recommendation reasoning
    REASONING_KEYWORDS = {
        'because', 'since', 'based on', 'due to', 'given', 'considering',
        'as you', 'similar to', 'matches', 'aligns with', 'reflects',
        'your preference', 'your history', 'past purchases', 'previous',
        'interested in', 'frequently', 'often', 'typically', 'usually',
    }

    # Product-related keywords
    PRODUCT_KEYWORDS = {
        'category', 'brand', 'price', 'quality', 'feature', 'rating',
        'review', 'popular', 'trending', 'new', 'similar', 'related',
    }

    def __init__(
        self,
        semantic_weight: float = 0.4,
        keyword_weight: float = 0.25,
        structure_weight: float = 0.2,
        length_weight: float = 0.15
    ):
        """
        Initialize evaluator.

        Args:
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword stability
            structure_weight: Weight for structural consistency
            length_weight: Weight for length stability
        """
        self.weights = {
            'semantic': semantic_weight,
            'keyword': keyword_weight,
            'structure': structure_weight,
            'length': length_weight,
        }
        assert abs(sum(self.weights.values()) - 1.0) < 0.01, "Weights must sum to 1"

    def evaluate(
        self,
        original_explanation: str,
        perturbed_explanation: str,
        perturbation_type: Optional[str] = None,
        severity: Optional[int] = None
    ) -> RobustnessResult:
        """
        Evaluate robustness between original and perturbed explanations.

        Args:
            original_explanation: Explanation from original user history
            perturbed_explanation: Explanation from perturbed history
            perturbation_type: Type of perturbation applied
            severity: Severity level (1-5)

        Returns:
            RobustnessResult with metrics
        """
        # 1. Semantic Similarity (using multiple methods)
        semantic_sim = self._compute_semantic_similarity(
            original_explanation, perturbed_explanation
        )

        # 2. Keyword Stability
        keyword_stability = self._compute_keyword_stability(
            original_explanation, perturbed_explanation
        )

        # 3. Structural Consistency
        structural_consistency = self._compute_structural_consistency(
            original_explanation, perturbed_explanation
        )

        # 4. Length Stability
        length_stability = self._compute_length_stability(
            original_explanation, perturbed_explanation
        )

        # Compute overall robustness
        overall = (
            self.weights['semantic'] * semantic_sim +
            self.weights['keyword'] * keyword_stability +
            self.weights['structure'] * structural_consistency +
            self.weights['length'] * length_stability
        )

        return RobustnessResult(
            semantic_similarity=semantic_sim,
            keyword_stability=keyword_stability,
            structural_consistency=structural_consistency,
            length_stability=length_stability,
            overall_robustness=overall,
            details={
                'perturbation_type': perturbation_type,
                'severity': severity,
            }
        )

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using multiple approaches."""
        # Bag-of-words cosine similarity
        bow_sim = TextSimilarity.cosine_similarity_bow(text1, text2)

        # BLEU score (treats text1 as reference)
        bleu = TextSimilarity.bleu_score(text1, text2)

        # Jaccard on words
        tokens1 = set(TextSimilarity.tokenize(text1))
        tokens2 = set(TextSimilarity.tokenize(text2))
        jaccard = TextSimilarity.jaccard_similarity(tokens1, tokens2)

        # Weighted combination
        return 0.4 * bow_sim + 0.3 * bleu + 0.3 * jaccard

    def _compute_keyword_stability(self, text1: str, text2: str) -> float:
        """Compute stability of key reasoning and product keywords."""
        tokens1 = set(TextSimilarity.tokenize(text1))
        tokens2 = set(TextSimilarity.tokenize(text2))

        # Extract reasoning keywords present
        reasoning1 = tokens1 & self.REASONING_KEYWORDS
        reasoning2 = tokens2 & self.REASONING_KEYWORDS

        # Extract product keywords present
        product1 = tokens1 & self.PRODUCT_KEYWORDS
        product2 = tokens2 & self.PRODUCT_KEYWORDS

        # Jaccard on reasoning keywords
        reasoning_sim = TextSimilarity.jaccard_similarity(reasoning1, reasoning2)

        # Jaccard on product keywords
        product_sim = TextSimilarity.jaccard_similarity(product1, product2)

        # Extract entities (capitalized words, likely product names/brands)
        entities1 = {t for t in tokens1 if len(t) > 2}
        entities2 = {t for t in tokens2 if len(t) > 2}
        entity_sim = TextSimilarity.jaccard_similarity(entities1, entities2)

        return 0.35 * reasoning_sim + 0.35 * product_sim + 0.3 * entity_sim

    def _compute_structural_consistency(self, text1: str, text2: str) -> float:
        """Compute structural similarity of explanations."""
        # Sentence count similarity
        sent1 = len(re.split(r'[.!?]+', text1.strip()))
        sent2 = len(re.split(r'[.!?]+', text2.strip()))
        sent_sim = 1.0 - abs(sent1 - sent2) / max(sent1, sent2, 1)

        # Paragraph structure (newline count)
        para1 = text1.count('\n') + 1
        para2 = text2.count('\n') + 1
        para_sim = 1.0 - abs(para1 - para2) / max(para1, para2, 1)

        # Check for common structural elements
        has_list1 = bool(re.search(r'^\s*[-*\d]', text1, re.MULTILINE))
        has_list2 = bool(re.search(r'^\s*[-*\d]', text2, re.MULTILINE))
        list_match = 1.0 if has_list1 == has_list2 else 0.5

        # N-gram overlap on structure
        bigrams1 = set(TextSimilarity.get_ngrams(TextSimilarity.tokenize(text1), 2))
        bigrams2 = set(TextSimilarity.get_ngrams(TextSimilarity.tokenize(text2), 2))
        bigram_sim = TextSimilarity.jaccard_similarity(bigrams1, bigrams2)

        return 0.2 * sent_sim + 0.15 * para_sim + 0.15 * list_match + 0.5 * bigram_sim

    def _compute_length_stability(self, text1: str, text2: str) -> float:
        """Compute length stability between explanations."""
        len1 = len(text1)
        len2 = len(text2)

        if len1 == 0 and len2 == 0:
            return 1.0
        if len1 == 0 or len2 == 0:
            return 0.0

        # Ratio-based stability
        ratio = min(len1, len2) / max(len1, len2)

        # Word count stability
        words1 = len(TextSimilarity.tokenize(text1))
        words2 = len(TextSimilarity.tokenize(text2))
        word_ratio = min(words1, words2) / max(words1, words2) if max(words1, words2) > 0 else 1.0

        return 0.5 * ratio + 0.5 * word_ratio


class RobustnessAnalyzer:
    """
    Comprehensive analyzer for explanation robustness across perturbations.

    Runs full evaluation grid and aggregates results.
    """

    def __init__(self, evaluator: Optional[RobustnessEvaluator] = None):
        self.evaluator = evaluator or RobustnessEvaluator()
        self.results: List[Dict[str, Any]] = []

    def add_result(
        self,
        model: str,
        perturbation_type: str,
        severity: int,
        original_explanation: str,
        perturbed_explanation: str
    ):
        """Add a single evaluation result."""
        result = self.evaluator.evaluate(
            original_explanation,
            perturbed_explanation,
            perturbation_type,
            severity
        )

        self.results.append({
            'model': model,
            'perturbation_type': perturbation_type,
            'severity': severity,
            'robustness': result.to_dict(),
        })

    def get_summary_by_model(self) -> Dict[str, Dict[str, float]]:
        """Get robustness summary grouped by model."""
        model_results = {}

        for r in self.results:
            model = r['model']
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(r['robustness'])

        summary = {}
        for model, results in model_results.items():
            summary[model] = {
                'overall_robustness': np.mean([r['overall_robustness'] for r in results]),
                'semantic_similarity': np.mean([r['semantic_similarity'] for r in results]),
                'keyword_stability': np.mean([r['keyword_stability'] for r in results]),
                'structural_consistency': np.mean([r['structural_consistency'] for r in results]),
                'length_stability': np.mean([r['length_stability'] for r in results]),
                'n_samples': len(results),
            }

        return summary

    def get_summary_by_perturbation(self) -> Dict[str, Dict[str, float]]:
        """Get robustness summary grouped by perturbation type."""
        ptype_results = {}

        for r in self.results:
            ptype = r['perturbation_type']
            if ptype not in ptype_results:
                ptype_results[ptype] = []
            ptype_results[ptype].append(r['robustness'])

        summary = {}
        for ptype, results in ptype_results.items():
            summary[ptype] = {
                'overall_robustness': np.mean([r['overall_robustness'] for r in results]),
                'std': np.std([r['overall_robustness'] for r in results]),
            }

        return summary

    def get_summary_by_severity(self) -> Dict[int, float]:
        """Get robustness degradation by severity level."""
        severity_results = {}

        for r in self.results:
            severity = r['severity']
            if severity not in severity_results:
                severity_results[severity] = []
            severity_results[severity].append(r['robustness']['overall_robustness'])

        return {s: np.mean(vals) for s, vals in sorted(severity_results.items())}

    def get_detailed_results(self) -> List[Dict[str, Any]]:
        """Get all detailed results."""
        return self.results

    def generate_report(self) -> str:
        """Generate text report of robustness analysis."""
        report = []
        report.append("=" * 60)
        report.append("ROBUSTNESS ANALYSIS REPORT")
        report.append("=" * 60)

        # By Model
        report.append("\n## Robustness by Model")
        model_summary = self.get_summary_by_model()
        for model, metrics in sorted(model_summary.items(), key=lambda x: -x[1]['overall_robustness']):
            report.append(f"\n{model}:")
            report.append(f"  Overall Robustness: {metrics['overall_robustness']:.4f}")
            report.append(f"  Semantic Similarity: {metrics['semantic_similarity']:.4f}")
            report.append(f"  Keyword Stability: {metrics['keyword_stability']:.4f}")
            report.append(f"  Samples: {metrics['n_samples']}")

        # By Perturbation
        report.append("\n\n## Robustness by Perturbation Type")
        ptype_summary = self.get_summary_by_perturbation()
        for ptype, metrics in sorted(ptype_summary.items(), key=lambda x: -x[1]['overall_robustness']):
            report.append(f"  {ptype}: {metrics['overall_robustness']:.4f} (±{metrics['std']:.4f})")

        # By Severity
        report.append("\n\n## Robustness by Severity Level")
        severity_summary = self.get_summary_by_severity()
        for severity, robustness in severity_summary.items():
            report.append(f"  Severity {severity}: {robustness:.4f}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


def demo():
    """Demo robustness evaluation."""
    print("=== Robustness Metrics Demo ===\n")

    # Sample explanations
    original = """Based on your shopping history showing strong interest in Electronics,
    particularly smartphones and accessories, we recommend this tablet.
    Your previous purchases in the Electronics category had an average rating of 4.5,
    and this item shares similar features with products you've enjoyed before."""

    perturbed_mild = """Given your shopping history with interest in Electronics,
    especially smartphones and accessories, we suggest this tablet.
    Your past purchases in Electronics had an average rating around 4.5,
    and this item has similar characteristics to products you've liked."""

    perturbed_severe = """This tablet might interest you. It's a popular Electronics item
    that many customers have rated highly. Consider it for your next purchase."""

    evaluator = RobustnessEvaluator()

    # Evaluate mild perturbation
    result_mild = evaluator.evaluate(original, perturbed_mild, "noise", 1)
    print("Mild Perturbation (Severity 1):")
    print(f"  Overall Robustness: {result_mild.overall_robustness:.4f}")
    print(f"  Semantic Similarity: {result_mild.semantic_similarity:.4f}")
    print(f"  Keyword Stability: {result_mild.keyword_stability:.4f}")

    # Evaluate severe perturbation
    result_severe = evaluator.evaluate(original, perturbed_severe, "noise", 5)
    print("\nSevere Perturbation (Severity 5):")
    print(f"  Overall Robustness: {result_severe.overall_robustness:.4f}")
    print(f"  Semantic Similarity: {result_severe.semantic_similarity:.4f}")
    print(f"  Keyword Stability: {result_severe.keyword_stability:.4f}")

    # Test analyzer
    print("\n\n=== Analyzer Demo ===")
    analyzer = RobustnessAnalyzer()

    # Simulate multiple results
    models = ['qwen2.5:72b', 'llama3.1:70b', 'qwen2.5:14b']
    perturbations = ['noise', 'shuffle', 'dilution']

    for model in models:
        for ptype in perturbations:
            for severity in [1, 3, 5]:
                # Simulate degradation based on severity
                if severity == 1:
                    perturbed = perturbed_mild
                elif severity == 3:
                    perturbed = perturbed_mild[:len(perturbed_mild)//2] + perturbed_severe[len(perturbed_severe)//2:]
                else:
                    perturbed = perturbed_severe

                analyzer.add_result(model, ptype, severity, original, perturbed)

    print(analyzer.generate_report())


if __name__ == "__main__":
    demo()
