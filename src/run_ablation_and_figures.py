"""
Ablation Experiments and Figure Generation for Paper B.
Runs experiments with smaller models and generates publication-ready figures.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from run_robustness_experiments import create_sample_data, run_full_experiment

# Set style for publication
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
})

def run_ablation_experiments(output_dir: str, n_users: int = 5):
    """Run ablation experiments with different model sizes."""
    print("=" * 60)
    print("RUNNING ABLATION EXPERIMENTS - Model Size Analysis")
    print("=" * 60)

    # Small models for ablation
    small_models = ['qwen2.5:7b', 'llama3.1:8b']

    # Create data
    data = create_sample_data(n_users=20, n_items=200)

    # Run experiments
    results = run_full_experiment(
        data=data,
        models=small_models,
        use_llm=True,
        n_users=n_users,
        output_dir=output_dir
    )

    return results


def load_all_results(results_dir: str) -> dict:
    """Load results from main experiment and combine with ablation."""
    # Load main results (large models)
    main_path = os.path.join(results_dir, 'robustness_results.json')
    with open(main_path, 'r') as f:
        main_results = json.load(f)

    # Load ablation results if exists
    ablation_path = os.path.join(results_dir, 'ablation_results.json')
    if os.path.exists(ablation_path):
        with open(ablation_path, 'r') as f:
            ablation_results = json.load(f)
        # Merge
        for model, metrics in ablation_results.get('model_summary', {}).items():
            main_results['model_summary'][model] = metrics

    return main_results


def generate_model_comparison_figure(results: dict, output_path: str):
    """Generate bar chart comparing model robustness by size."""
    model_summary = results.get('model_summary', {})

    # Order models by size - include all possible models
    model_order = [
        ('qwen2.5:7b', 'Qwen2.5\n7B', '#74b9ff'),
        ('qwen2.5:14b', 'Qwen2.5\n14B', '#0984e3'),
        ('llama3.1:8b', 'LLaMA3.1\n8B', '#fab1a0'),
        ('llama3.1:70b', 'LLaMA3.1\n70B', '#d63031'),
    ]

    models = []
    scores = []
    stds = []
    colors = []

    for model_key, model_name, color in model_order:
        if model_key in model_summary:
            models.append(model_name)
            scores.append(model_summary[model_key]['mean'])
            stds.append(model_summary[model_key]['std'])
            colors.append(color)

    if not models:
        print("No model data available for figure")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(models))
    bars = ax.bar(x, scores, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Robustness Score')
    ax.set_title('Model Size vs Explanation Robustness')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0.45, 0.60)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_perturbation_figure(results: dict, output_path: str):
    """Generate bar chart of perturbation type analysis."""
    ptype_summary = results.get('perturbation_summary', {})

    # Order by robustness
    ptypes = ['missing', 'noise', 'drift', 'shuffle', 'dilution']
    labels = ['Missing\nValues', 'Noise\nInjection', 'Category\nDrift', 'Temporal\nShuffle', 'Behavior\nDilution']

    scores = []
    stds = []
    for ptype in ptypes:
        if ptype in ptype_summary:
            scores.append(ptype_summary[ptype]['mean'])
            stds.append(ptype_summary[ptype]['std'])
        else:
            scores.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(ptypes))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(ptypes)))

    bars = ax.bar(x, scores, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Perturbation Type')
    ax.set_ylabel('Robustness Score')
    ax.set_title('Explanation Robustness by Perturbation Type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.50, 0.55)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_severity_figure(results: dict, output_path: str):
    """Generate line chart of severity degradation."""
    severity_summary = results.get('severity_summary', {})

    levels = [1, 2, 3, 4, 5]
    scores = []
    for level in levels:
        score = severity_summary.get(level, severity_summary.get(str(level), 0.5))
        scores.append(score)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(levels, scores, 'o-', color='#2c3e50', linewidth=2, markersize=8, markerfacecolor='#3498db')
    ax.fill_between(levels, [s - 0.01 for s in scores], [s + 0.01 for s in scores],
                    alpha=0.2, color='#3498db')

    ax.set_xlabel('Perturbation Severity Level')
    ax.set_ylabel('Robustness Score')
    ax.set_title('Robustness Degradation with Increasing Severity')
    ax.set_xticks(levels)
    ax.set_ylim(0.50, 0.55)

    # Add annotations
    degradation = (scores[0] - scores[-1]) / scores[0] * 100
    ax.annotate(f'Degradation: {degradation:.1f}%',
                xy=(3, min(scores)), xytext=(3.5, min(scores) - 0.01),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_ablation', action='store_true', help='Run ablation experiments')
    parser.add_argument('--n_users', type=int, default=5, help='Number of users for ablation')
    parser.add_argument('--figures_only', action='store_true', help='Only generate figures')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'experiments', 'results')
    figures_dir = os.path.join(script_dir, '..', 'paper', 'figures')
    results_dir = os.path.abspath(results_dir)
    figures_dir = os.path.abspath(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Results dir: {results_dir}")
    print(f"Figures dir: {figures_dir}")

    # Run ablation if requested
    if args.run_ablation and not args.figures_only:
        ablation_results = run_ablation_experiments(
            output_dir=results_dir + '/ablation',
            n_users=args.n_users
        )
        # Save ablation results
        ablation_path = os.path.join(results_dir, 'ablation_results.json')
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=2, default=str)
        print(f"Ablation results saved to {ablation_path}")

    # Load all results
    print("\nLoading results...")
    results = load_all_results(results_dir)

    # Generate figures
    print("\nGenerating figures...")

    generate_model_comparison_figure(
        results,
        os.path.join(figures_dir, 'model_comparison.pdf')
    )

    generate_perturbation_figure(
        results,
        os.path.join(figures_dir, 'perturbation_analysis.pdf')
    )

    generate_severity_figure(
        results,
        os.path.join(figures_dir, 'severity_curve.pdf')
    )

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
