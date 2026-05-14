import numpy as np

from sir.comparison import ComparisonResult


def test_comparison_result_is_constructible():
    cr = ComparisonResult(
        cfg=None,
        interventions_resolved=[],
        baseline_results=[],
        treatment_results=[],
        baseline_healthcare=None,
        treatment_healthcare=None,
        intervention_samples=None,
    )
    assert cr.baseline_results == []
    assert cr.treatment_results == []
