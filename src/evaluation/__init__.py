"""
SignTalk AI: Comprehensive Evaluation Framework.
"""

from src.evaluation.classification_metrics import (
    compute_classification_metrics,
    format_metrics_table
)
from src.evaluation.translation_metrics import (
    compute_token_accuracy,
    compute_sequence_accuracy,
    compute_bleu,
    compute_rouge_l,
    compute_chrf,
    compute_wer
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    analyze_top_confusions,
    plot_confusion_matrix
)
from src.evaluation.confidence_analysis import (
    compute_expected_calibration_error,
    compute_brier_score,
    analyze_threshold_rejection,
    summarize_confidence_distribution
)
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.benchmark import (
    benchmark_model_inference,
    get_hardware_environment,
    count_parameters
)
from src.evaluation.robustness import evaluate_robustness_suite
from src.evaluation.ablation import (
    run_modality_ablation,
    run_graph_topology_ablation,
    run_sequence_length_ablation
)
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.report import (
    generate_model_summary_markdown,
    generate_comparison_table
)

__all__ = [
    "compute_classification_metrics",
    "format_metrics_table",
    "compute_token_accuracy",
    "compute_sequence_accuracy",
    "compute_bleu",
    "compute_rouge_l",
    "compute_chrf",
    "compute_wer",
    "compute_metrics",
    "compute_confusion_matrix",
    "analyze_top_confusions",
    "plot_confusion_matrix",
    "compute_expected_calibration_error",
    "compute_brier_score",
    "analyze_threshold_rejection",
    "summarize_confidence_distribution",
    "ErrorAnalyzer",
    "benchmark_model_inference",
    "get_hardware_environment",
    "count_parameters",
    "evaluate_robustness_suite",
    "run_modality_ablation",
    "run_graph_topology_ablation",
    "run_sequence_length_ablation",
    "ModelEvaluator",
    "generate_model_summary_markdown",
    "generate_comparison_table"
]
