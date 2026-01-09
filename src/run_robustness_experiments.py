"""
Experiment Runner for XAI Robustness Evaluation.

Runs comprehensive robustness evaluation of LLM-generated explanations
under various user behavior perturbations.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import time

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from perturbations import (
    PerturbationType, PerturbationFactory, apply_perturbation_grid
)
from explainers import (
    ExplanationConfig, ExplanationStyle, LLMExplainer, MockExplainer
)
from evaluation import RobustnessEvaluator, RobustnessAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def create_sample_data(n_users: int = 10, n_items: int = 100) -> Dict[str, Any]:
    """Create synthetic sample data for testing."""
    import random
    random.seed(RANDOM_SEED)

    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Toys']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

    # Create items
    items = {}
    for i in range(n_items):
        items[f'item_{i}'] = {
            'item_id': f'item_{i}',
            'category': random.choice(categories),
            'brand': random.choice(brands),
            'price': random.randint(10, 500),
            'rating': round(random.uniform(3.0, 5.0), 1),
            'is_new': random.random() < 0.2,
        }

    # Create user histories
    users = {}
    for u in range(n_users):
        history_length = random.randint(10, 50)
        # Simulate user preference (biased toward 1-2 categories)
        preferred_cats = random.sample(categories, 2)

        history = []
        for _ in range(history_length):
            # 70% from preferred, 30% random
            if random.random() < 0.7:
                cat = random.choice(preferred_cats)
            else:
                cat = random.choice(categories)

            cat_items = [iid for iid, idata in items.items() if idata['category'] == cat]
            if cat_items:
                item_id = random.choice(cat_items)
            else:
                item_id = random.choice(list(items.keys()))

            history.append({
                'item_id': item_id,
                'category': items[item_id]['category'],
                'rating': random.randint(3, 5),
                'timestamp': len(history),
            })

        users[f'user_{u}'] = {
            'history': history,
            'preferred_categories': preferred_cats,
        }

    return {
        'items': items,
        'users': users,
        'categories': categories,
    }


def run_single_user_evaluation(
    user_history: List[Dict],
    recommendation: Dict,
    item_features: Dict[str, Dict],
    models: List[str],
    use_llm: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run robustness evaluation for a single user.

    Generates explanations from original and perturbed histories,
    then measures robustness.
    """
    results = {
        'models': {},
        'perturbations': {},
        'raw_results': [],
    }

    analyzer = RobustnessAnalyzer()
    evaluator = RobustnessEvaluator()

    # Apply all perturbations
    all_items = list(item_features.keys())
    perturbation_grid = apply_perturbation_grid(user_history, all_items)

    for model_name in models:
        logger.info(f"Evaluating model: {model_name}")

        # Initialize explainer
        if use_llm:
            config = ExplanationConfig(model_name=model_name, style=ExplanationStyle.DETAILED)
            try:
                explainer = LLMExplainer(config)
            except Exception as e:
                logger.warning(f"Could not init LLM {model_name}: {e}. Using mock.")
                explainer = MockExplainer(config)
        else:
            config = ExplanationConfig(style=ExplanationStyle.DETAILED)
            explainer = MockExplainer(config)

        # Generate original explanation
        original_exp = explainer.generate(user_history, recommendation, item_features)
        original_text = original_exp.text

        # Evaluate each perturbation
        for ptype_str, severity_dict in perturbation_grid.items():
            for severity, perturbed_history in severity_dict.items():
                # Generate explanation from perturbed history
                perturbed_exp = explainer.generate(perturbed_history, recommendation, item_features)
                perturbed_text = perturbed_exp.text

                # Evaluate robustness
                result = evaluator.evaluate(
                    original_text, perturbed_text,
                    ptype_str, severity
                )

                # Store result
                analyzer.add_result(
                    model_name, ptype_str, severity,
                    original_text, perturbed_text
                )

                results['raw_results'].append({
                    'model': model_name,
                    'perturbation': ptype_str,
                    'severity': severity,
                    'original_explanation': original_text[:200] + '...' if len(original_text) > 200 else original_text,
                    'perturbed_explanation': perturbed_text[:200] + '...' if len(perturbed_text) > 200 else perturbed_text,
                    'robustness': result.to_dict(),
                })

    # Aggregate results
    results['models'] = analyzer.get_summary_by_model()
    results['perturbations'] = analyzer.get_summary_by_perturbation()
    results['severity_curve'] = analyzer.get_summary_by_severity()
    results['report'] = analyzer.generate_report()

    return results


def run_full_experiment(
    data: Dict[str, Any],
    models: List[str],
    use_llm: bool = False,
    n_users: int = 5,
    output_dir: str = './results'
) -> Dict[str, Any]:
    """
    Run full robustness experiment across multiple users.
    """
    logger.info("Starting full robustness experiment...")

    all_results = []
    items = data['items']
    users = list(data['users'].items())[:n_users]

    for user_id, user_data in users:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {user_id}")
        logger.info(f"{'='*50}")

        history = user_data['history']

        # Create a recommendation (random item from non-history)
        history_items = {h['item_id'] for h in history}
        candidate_items = [iid for iid in items.keys() if iid not in history_items]
        if candidate_items:
            rec_id = np.random.choice(candidate_items)
        else:
            rec_id = np.random.choice(list(items.keys()))

        recommendation = items[rec_id]

        # Run evaluation
        user_results = run_single_user_evaluation(
            user_history=history,
            recommendation=recommendation,
            item_features=items,
            models=models,
            use_llm=use_llm,
            output_dir=output_dir
        )

        user_results['user_id'] = user_id
        user_results['recommendation'] = rec_id
        all_results.append(user_results)

    # Aggregate across users
    final_results = aggregate_results(all_results)
    final_results['individual_users'] = all_results

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'robustness_results.json')
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # Save report
    report_path = os.path.join(output_dir, 'robustness_report.txt')
    with open(report_path, 'w') as f:
        f.write(generate_full_report(final_results))
    logger.info(f"Report saved to {report_path}")

    return final_results


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across multiple users."""
    model_robustness = {}
    perturbation_robustness = {}
    severity_robustness = {}

    for user_result in all_results:
        # Aggregate by model
        for model, metrics in user_result.get('models', {}).items():
            if model not in model_robustness:
                model_robustness[model] = []
            model_robustness[model].append(metrics['overall_robustness'])

        # Aggregate by perturbation
        for ptype, metrics in user_result.get('perturbations', {}).items():
            if ptype not in perturbation_robustness:
                perturbation_robustness[ptype] = []
            perturbation_robustness[ptype].append(metrics['overall_robustness'])

        # Aggregate by severity
        for severity, robustness in user_result.get('severity_curve', {}).items():
            severity = int(severity)
            if severity not in severity_robustness:
                severity_robustness[severity] = []
            severity_robustness[severity].append(robustness)

    # Compute final aggregates
    final_model = {
        model: {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'n_samples': len(vals),
        }
        for model, vals in model_robustness.items()
    }

    final_perturbation = {
        ptype: {
            'mean': np.mean(vals),
            'std': np.std(vals),
        }
        for ptype, vals in perturbation_robustness.items()
    }

    final_severity = {
        severity: np.mean(vals)
        for severity, vals in sorted(severity_robustness.items())
    }

    return {
        'model_summary': final_model,
        'perturbation_summary': final_perturbation,
        'severity_summary': final_severity,
        'timestamp': datetime.now().isoformat(),
        'n_users': len(all_results),
    }


def generate_full_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive report."""
    report = []
    report.append("=" * 70)
    report.append("XAI ROBUSTNESS EVALUATION REPORT")
    report.append("LLM-Generated Explanations under User Behavior Perturbations")
    report.append("=" * 70)
    report.append(f"\nTimestamp: {results.get('timestamp', 'N/A')}")
    report.append(f"Number of Users: {results.get('n_users', 'N/A')}")

    # Model Rankings
    report.append("\n\n" + "=" * 50)
    report.append("MODEL ROBUSTNESS RANKING")
    report.append("=" * 50)

    model_summary = results.get('model_summary', {})
    sorted_models = sorted(model_summary.items(), key=lambda x: -x[1]['mean'])

    for rank, (model, metrics) in enumerate(sorted_models, 1):
        report.append(f"\n{rank}. {model}")
        report.append(f"   Robustness Score: {metrics['mean']:.4f} ± {metrics['std']:.4f}")

    # Perturbation Analysis
    report.append("\n\n" + "=" * 50)
    report.append("PERTURBATION TYPE ANALYSIS")
    report.append("=" * 50)

    ptype_summary = results.get('perturbation_summary', {})
    sorted_ptypes = sorted(ptype_summary.items(), key=lambda x: -x[1]['mean'])

    report.append("\nRobustness by perturbation type (higher = more robust):")
    for ptype, metrics in sorted_ptypes:
        report.append(f"  {ptype:15s}: {metrics['mean']:.4f} ± {metrics['std']:.4f}")

    # Severity Curve
    report.append("\n\n" + "=" * 50)
    report.append("SEVERITY DEGRADATION CURVE")
    report.append("=" * 50)

    severity_summary = results.get('severity_summary', {})
    report.append("\nRobustness vs Perturbation Severity:")
    for severity in range(1, 6):
        robustness = severity_summary.get(severity, severity_summary.get(str(severity), 0))
        bar = "█" * int(robustness * 20)
        report.append(f"  Level {severity}: {robustness:.4f} {bar}")

    # Key Findings
    report.append("\n\n" + "=" * 50)
    report.append("KEY FINDINGS")
    report.append("=" * 50)

    if sorted_models:
        best_model = sorted_models[0][0]
        worst_model = sorted_models[-1][0]
        report.append(f"\n1. Most robust model: {best_model} ({sorted_models[0][1]['mean']:.4f})")
        report.append(f"2. Least robust model: {worst_model} ({sorted_models[-1][1]['mean']:.4f})")

    if sorted_ptypes:
        least_damaging = sorted_ptypes[0][0]
        most_damaging = sorted_ptypes[-1][0]
        report.append(f"3. Least damaging perturbation: {least_damaging}")
        report.append(f"4. Most damaging perturbation: {most_damaging}")

    if severity_summary:
        s1 = severity_summary.get(1, severity_summary.get('1', 0))
        s5 = severity_summary.get(5, severity_summary.get('5', 0))
        degradation = ((s1 - s5) / s1 * 100) if s1 > 0 else 0
        report.append(f"5. Robustness degradation (severity 1→5): {degradation:.1f}%")

    report.append("\n" + "=" * 70)
    return "\n".join(report)


def generate_latex_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX table for paper."""
    table = "\\begin{table}[h]\n\\centering\n"
    table += "\\caption{Robustness of LLM-Generated Explanations}\n"
    table += "\\label{tab:robustness}\n"
    table += "\\begin{tabular}{l" + "c" * 5 + "c}\n\\hline\n"
    table += "Model & Noise & Shuffle & Dilution & Drift & Missing & Overall \\\\ \\hline\n"

    model_summary = results.get('model_summary', {})
    ptype_summary = results.get('perturbation_summary', {})

    for model, metrics in sorted(model_summary.items(), key=lambda x: -x[1]['mean']):
        row = f"{model.replace(':', '-')} & "
        for ptype in ['noise', 'shuffle', 'dilution', 'drift', 'missing']:
            pmetrics = ptype_summary.get(ptype, {'mean': 0})
            row += f"{pmetrics['mean']:.3f} & "
        row += f"\\textbf{{{metrics['mean']:.3f}}} \\\\\n"
        table += row

    table += "\\hline\n\\end{tabular}\n\\end{table}"
    return table


def main():
    parser = argparse.ArgumentParser(description='Run XAI Robustness Experiments')
    parser.add_argument('--output_dir', type=str, default='../experiments/results',
                       help='Output directory')
    parser.add_argument('--use_llm', action='store_true',
                       help='Use actual LLM (default: mock explainer)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['qwen2.5:72b', 'llama3.1:70b', 'qwen2.5:14b'],
                       help='Models to evaluate')
    parser.add_argument('--n_users', type=int, default=5,
                       help='Number of users to evaluate')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode')

    args = parser.parse_args()

    # Create sample data
    logger.info("Creating sample data...")
    data = create_sample_data(n_users=20, n_items=200)
    logger.info(f"Created {len(data['items'])} items and {len(data['users'])} users")

    # Adjust for quick mode
    if args.quick:
        args.n_users = 2
        args.models = ['mock']
        args.use_llm = False

    # Run experiment
    results = run_full_experiment(
        data=data,
        models=args.models,
        use_llm=args.use_llm,
        n_users=args.n_users,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)

    model_summary = results.get('model_summary', {})
    print("\nModel Robustness Ranking:")
    for model, metrics in sorted(model_summary.items(), key=lambda x: -x[1]['mean']):
        print(f"  {model}: {metrics['mean']:.4f}")

    # Generate LaTeX
    latex = generate_latex_table(results)
    latex_path = os.path.join(args.output_dir, 'robustness_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
