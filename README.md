# RobustExplain: Evaluating Robustness of LLM-Based Explanation Agents for Recommendation

This repository contains the implementation of **RobustExplain**, a systematic evaluation framework for assessing the robustness of LLM-generated recommendation explanations under user behavior perturbations.

## Overview

RobustExplain provides:
- **Five perturbation types** modeling realistic user behavior variations:
  - Noise Injection: Random interactions simulating accidental clicks
  - Temporal Shuffle: Randomized interaction order
  - Behavior Dilution: Injected interactions from different categories
  - Category Drift: Shifted user preferences
  - Missing Values: Incomplete metadata
- **Multi-dimensional robustness metrics**:
  - Semantic Similarity
  - Keyword Stability
  - Structural Consistency
  - Length Stability
- **Support for multiple LLM models** via Ollama

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RobustExplain.git
cd RobustExplain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull models
# See https://ollama.ai for installation instructions
ollama pull qwen2.5:14b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

## Project Structure

```
RobustExplain/
├── src/
│   ├── evaluation/
│   │   └── robustness_metrics.py    # Robustness evaluation metrics
│   ├── explainers/
│   │   └── llm_explainer.py         # LLM-based explanation generator
│   ├── perturbations/
│   │   └── perturbation_factory.py  # Perturbation implementations
│   ├── run_robustness_experiments.py # Main experiment script
│   └── run_ablation_and_figures.py   # Ablation studies and visualization
├── data/                             # Dataset directory
├── results/                          # Experiment results
├── requirements.txt
└── README.md
```

## Usage

### Running Experiments

```bash
# Run main robustness evaluation
python src/run_robustness_experiments.py \
    --n_users 20 \
    --models qwen2.5:14b,llama3.1:8b,qwen2.5:7b \
    --output_dir results/

# Run ablation studies
python src/run_ablation_and_figures.py \
    --n_users 10 \
    --run_ablation
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n_users` | Number of users to evaluate | 20 |
| `--models` | Comma-separated list of models | qwen2.5:14b |
| `--output_dir` | Output directory for results | results/ |
| `--severity_levels` | Number of severity levels | 5 |

## Perturbation Types

### 1. Noise Injection
Adds random interactions simulating accidental clicks or exploratory browsing.

### 2. Temporal Shuffle
Randomly permutes interaction order within the user history.

### 3. Behavior Dilution
Injects interactions from categories different from the user's dominant preferences.

### 4. Category Drift
Shifts user preferences toward different categories.

### 5. Missing Values
Removes ratings, timestamps, or category information from interactions.

## Metrics

The robustness score is computed as a weighted combination:

```
ρ = 0.4 × Semantic + 0.25 × Keyword + 0.2 × Structural + 0.15 × Length
```

## Results

Our experiments with three LLM models reveal:
- Overall robustness scores around 0.50, indicating moderate sensitivity to perturbations
- Larger models (14B) demonstrate ~5.5% higher robustness than smaller models (7-8B)
- Robustness remains stable across severity levels with only 1.7% degradation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was conducted as part of research on explainable recommendation systems with large language models.
