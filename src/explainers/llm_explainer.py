"""
LLM-based Explanation Generator for Recommendation Systems.

Generates natural language explanations for why items are recommended,
using local Ollama models (qwen2.5:72b, llama3.1:70b, etc.)
"""

import sys
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import time

# Add shared module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'))

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class ExplanationStyle(Enum):
    """Different styles of explanations."""
    CONCISE = "concise"  # Brief, to the point
    DETAILED = "detailed"  # Full explanation with reasoning
    CONVERSATIONAL = "conversational"  # Friendly, casual tone
    TECHNICAL = "technical"  # Focus on features and attributes


@dataclass
class ExplanationConfig:
    """Configuration for explanation generation."""
    model_name: str = "qwen2.5:72b"
    style: ExplanationStyle = ExplanationStyle.DETAILED
    max_length: int = 200  # Max words in explanation
    include_evidence: bool = True  # Include supporting evidence
    temperature: float = 0.7
    timeout: int = 60


@dataclass
class Explanation:
    """Represents a generated explanation."""
    text: str
    model: str
    user_history_hash: str
    recommendation: str
    style: ExplanationStyle
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'model': self.model,
            'user_history_hash': self.user_history_hash,
            'recommendation': self.recommendation,
            'style': self.style.value,
            'generation_time': self.generation_time,
            'metadata': self.metadata,
        }


class BaseExplainer(ABC):
    """Abstract base class for explanation generators."""

    def __init__(self, config: Optional[ExplanationConfig] = None):
        self.config = config or ExplanationConfig()

    @abstractmethod
    def generate(
        self,
        user_history: List[Dict],
        recommendation: Dict,
        item_features: Optional[Dict[str, Dict]] = None
    ) -> Explanation:
        """Generate explanation for a recommendation."""
        pass

    @abstractmethod
    def generate_batch(
        self,
        user_history: List[Dict],
        recommendations: List[Dict],
        item_features: Optional[Dict[str, Dict]] = None
    ) -> List[Explanation]:
        """Generate explanations for multiple recommendations."""
        pass

    def _hash_history(self, history: List[Dict]) -> str:
        """Create hash of user history for tracking."""
        history_str = json.dumps(history, sort_keys=True)
        return hashlib.md5(history_str.encode()).hexdigest()[:12]


class LLMExplainer(BaseExplainer):
    """LLM-based explanation generator using Ollama."""

    # Prompt templates for different styles
    STYLE_PROMPTS = {
        ExplanationStyle.CONCISE: """Generate a brief 1-2 sentence explanation for why this item is recommended.
Be direct and focus on the most important reason.""",

        ExplanationStyle.DETAILED: """Generate a detailed explanation for why this item is recommended.
Include:
1. The main reason based on user preferences
2. How it relates to past purchases/views
3. Any unique features that match the user's interests""",

        ExplanationStyle.CONVERSATIONAL: """Generate a friendly, conversational explanation for this recommendation.
Write as if you're a helpful shopping assistant talking to a friend.
Be warm and personable while explaining the recommendation.""",

        ExplanationStyle.TECHNICAL: """Generate a technical explanation focusing on item attributes and features.
Reference specific product categories, attributes, and how they match the user's demonstrated preferences.
Use precise terminology related to the product domain.""",
    }

    def __init__(self, config: Optional[ExplanationConfig] = None):
        super().__init__(config)
        self._verify_ollama()

    def _verify_ollama(self):
        """Verify Ollama is available."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama module not available. Install with: pip install ollama")

        try:
            models = ollama.list()
            available = [m.model for m in models.models] if hasattr(models, 'models') else []
            if self.config.model_name not in available and f"{self.config.model_name}:latest" not in available:
                # Check for partial match
                matching = [m for m in available if self.config.model_name.split(':')[0] in m]
                if not matching:
                    print(f"Warning: Model {self.config.model_name} not found. Available: {available}")
        except Exception as e:
            print(f"Warning: Could not verify Ollama models: {e}")

    def _build_prompt(
        self,
        user_history: List[Dict],
        recommendation: Dict,
        item_features: Optional[Dict[str, Dict]] = None
    ) -> str:
        """Build the prompt for explanation generation."""
        # Format user history
        history_summary = self._summarize_history(user_history)

        # Get item details
        rec_id = recommendation.get('item_id', 'Unknown')
        rec_features = {}
        if item_features and rec_id in item_features:
            rec_features = item_features[rec_id]

        rec_category = rec_features.get('category', recommendation.get('category', 'Unknown'))
        rec_details = json.dumps(rec_features, indent=2) if rec_features else "No additional details"

        # Style-specific instruction
        style_instruction = self.STYLE_PROMPTS.get(
            self.config.style,
            self.STYLE_PROMPTS[ExplanationStyle.DETAILED]
        )

        prompt = f"""You are an AI assistant explaining product recommendations in an e-commerce system.

USER'S SHOPPING HISTORY:
{history_summary}

RECOMMENDED ITEM:
- Item ID: {rec_id}
- Category: {rec_category}
- Details: {rec_details}

TASK:
{style_instruction}

Generate the explanation now. Do not include any meta-commentary, just provide the explanation directly."""

        return prompt

    def _summarize_history(self, history: List[Dict], max_items: int = 10) -> str:
        """Summarize user history for the prompt."""
        if not history:
            return "No previous interactions recorded."

        # Take most recent items
        recent = history[-max_items:]

        # Group by category
        categories = {}
        for item in recent:
            cat = item.get('category', 'Unknown')
            if cat is None:
                cat = 'Unknown'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)

        summary_parts = []
        for cat, items in categories.items():
            item_ids = [i.get('item_id', 'unknown') for i in items[:3]]
            ratings = [i.get('rating') for i in items if i.get('rating') is not None]
            valid_ratings = [r for r in ratings if isinstance(r, (int, float))]
            avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else None

            summary = f"- {cat}: {len(items)} items (e.g., {', '.join(item_ids[:3])})"
            if avg_rating:
                summary += f", avg rating: {avg_rating:.1f}"
            summary_parts.append(summary)

        return "\n".join(summary_parts)

    def generate(
        self,
        user_history: List[Dict],
        recommendation: Dict,
        item_features: Optional[Dict[str, Dict]] = None
    ) -> Explanation:
        """Generate explanation for a single recommendation."""
        start_time = time.time()

        prompt = self._build_prompt(user_history, recommendation, item_features)

        try:
            response = ollama.chat(
                model=self.config.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful e-commerce recommendation assistant.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_length * 5,  # Rough char estimate
                }
            )
            explanation_text = response.message.content.strip()
        except Exception as e:
            explanation_text = f"[Error generating explanation: {str(e)}]"

        generation_time = time.time() - start_time

        return Explanation(
            text=explanation_text,
            model=self.config.model_name,
            user_history_hash=self._hash_history(user_history),
            recommendation=recommendation.get('item_id', str(recommendation)),
            style=self.config.style,
            generation_time=generation_time,
            metadata={
                'history_length': len(user_history),
                'prompt_length': len(prompt),
            }
        )

    def generate_batch(
        self,
        user_history: List[Dict],
        recommendations: List[Dict],
        item_features: Optional[Dict[str, Dict]] = None
    ) -> List[Explanation]:
        """Generate explanations for multiple recommendations."""
        explanations = []
        for rec in recommendations:
            exp = self.generate(user_history, rec, item_features)
            explanations.append(exp)
        return explanations


class MockExplainer(BaseExplainer):
    """Mock explainer for testing without LLM."""

    TEMPLATES = {
        ExplanationStyle.CONCISE: "Recommended because you frequently purchase {category} items.",
        ExplanationStyle.DETAILED: "Based on your shopping history showing interest in {category} products, we recommend this item. Your previous purchases in this category had an average rating of {avg_rating:.1f}, and this item has similar characteristics.",
        ExplanationStyle.CONVERSATIONAL: "Hey! Since you seem to love {category} stuff, I thought you'd really like this one. It's similar to other things you've bought before!",
        ExplanationStyle.TECHNICAL: "Recommendation based on category affinity score: {category} (user preference weight: {weight:.2f}). Historical interaction count: {count}.",
    }

    def generate(
        self,
        user_history: List[Dict],
        recommendation: Dict,
        item_features: Optional[Dict[str, Dict]] = None
    ) -> Explanation:
        """Generate mock explanation."""
        start_time = time.time()

        # Analyze history
        categories = {}
        total_rating = 0
        rating_count = 0
        for item in user_history:
            cat = item.get('category', 'Unknown')
            if cat is not None:  # Handle missing values from perturbation
                categories[cat] = categories.get(cat, 0) + 1
            rating = item.get('rating')
            if rating is not None and isinstance(rating, (int, float)):
                total_rating += rating
                rating_count += 1

        # Get most common category
        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else 'Unknown'
        avg_rating = total_rating / rating_count if rating_count > 0 else 3.5

        # Generate from template
        template = self.TEMPLATES.get(self.config.style, self.TEMPLATES[ExplanationStyle.DETAILED])
        explanation_text = template.format(
            category=top_category,
            avg_rating=avg_rating,
            weight=categories.get(top_category, 1) / len(user_history) if user_history else 0.5,
            count=len(user_history),
        )

        generation_time = time.time() - start_time

        return Explanation(
            text=explanation_text,
            model="mock",
            user_history_hash=self._hash_history(user_history),
            recommendation=recommendation.get('item_id', str(recommendation)),
            style=self.config.style,
            generation_time=generation_time,
            metadata={'mock': True}
        )

    def generate_batch(
        self,
        user_history: List[Dict],
        recommendations: List[Dict],
        item_features: Optional[Dict[str, Dict]] = None
    ) -> List[Explanation]:
        """Generate mock explanations for batch."""
        return [self.generate(user_history, rec, item_features) for rec in recommendations]


class MultiModelExplainer:
    """
    Generates explanations using multiple LLM models for comparison.

    Used in robustness evaluation to compare explanation stability across models.
    """

    def __init__(
        self,
        models: List[str] = None,
        style: ExplanationStyle = ExplanationStyle.DETAILED
    ):
        self.models = models or ['qwen2.5:72b', 'llama3.1:70b', 'qwen2.5:14b']
        self.style = style
        self.explainers: Dict[str, LLMExplainer] = {}

        for model in self.models:
            config = ExplanationConfig(model_name=model, style=style)
            try:
                self.explainers[model] = LLMExplainer(config)
            except Exception as e:
                print(f"Warning: Could not initialize {model}: {e}")

    def generate_all(
        self,
        user_history: List[Dict],
        recommendation: Dict,
        item_features: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Explanation]:
        """Generate explanations from all models."""
        results = {}
        for model, explainer in self.explainers.items():
            try:
                results[model] = explainer.generate(user_history, recommendation, item_features)
            except Exception as e:
                print(f"Error with {model}: {e}")
        return results


def demo():
    """Demo explanation generation."""
    # Sample data
    user_history = [
        {'item_id': 'phone_1', 'category': 'Electronics', 'rating': 5},
        {'item_id': 'phone_2', 'category': 'Electronics', 'rating': 4},
        {'item_id': 'case_1', 'category': 'Phone Accessories', 'rating': 4},
        {'item_id': 'laptop_1', 'category': 'Electronics', 'rating': 5},
        {'item_id': 'headphones_1', 'category': 'Electronics', 'rating': 3},
    ]

    recommendation = {'item_id': 'tablet_1', 'category': 'Electronics'}

    item_features = {
        'tablet_1': {
            'category': 'Electronics',
            'brand': 'Apple',
            'price': 799,
            'features': ['Retina display', 'M1 chip', '256GB storage'],
        }
    }

    print("=== Explanation Generation Demo ===\n")

    # Test mock explainer
    print("1. Mock Explainer (no LLM needed):")
    for style in ExplanationStyle:
        config = ExplanationConfig(style=style)
        explainer = MockExplainer(config)
        exp = explainer.generate(user_history, recommendation, item_features)
        print(f"\n  [{style.value}]: {exp.text}")

    # Test LLM explainer if available
    if OLLAMA_AVAILABLE:
        print("\n\n2. LLM Explainer (using qwen2.5:14b for demo):")
        try:
            config = ExplanationConfig(model_name='qwen2.5:14b', style=ExplanationStyle.DETAILED)
            explainer = LLMExplainer(config)
            exp = explainer.generate(user_history, recommendation, item_features)
            print(f"\n  Generated in {exp.generation_time:.2f}s:")
            print(f"  {exp.text}")
        except Exception as e:
            print(f"  LLM explainer error: {e}")
    else:
        print("\n\n2. LLM Explainer: Ollama not available")


if __name__ == "__main__":
    demo()
