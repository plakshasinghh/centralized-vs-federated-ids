"""Comparative analysis between centralized and federated models.

Intended to consume ExperimentResult objects (from src.reporting) produced by
both the centralized baseline runs and the federated CNN+LSTM run, and
generate side-by-side comparison charts/tables into reports/figures/comparisons.
"""


def compare_models(results: dict):
    """Compare multiple named ExperimentResult-like dicts on shared metrics.

    Args:
        results: mapping of model_name -> metrics dict (accuracy, auc, f1, etc.)
    """
    raise NotImplementedError("Fill in comparison logic here.")
