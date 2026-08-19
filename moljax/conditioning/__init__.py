"""Matrix-free conditioning diagnostics."""

from .field_of_values import FieldOfValuesResult, numerical_range
from .non_normality import (
    PreconditionerAssessment,
    RateEstimates,
    assess_preconditioner,
    clustering_rate,
    crouzeix_palencia_envelope,
    enclosing_disk_rate,
    estimate_rates,
    right_real_outliers,
    traced_boundary_rate,
)
from .pseudospectra import (
    PseudospectraResult,
    arnoldi,
    epsilon_zero,
    pseudospectrum_dense,
    reduced_pseudospectrum,
    ritz_values,
)

__all__ = [
    "FieldOfValuesResult",
    "PseudospectraResult",
    "PreconditionerAssessment",
    "RateEstimates",
    "arnoldi",
    "assess_preconditioner",
    "clustering_rate",
    "crouzeix_palencia_envelope",
    "epsilon_zero",
    "enclosing_disk_rate",
    "estimate_rates",
    "numerical_range",
    "pseudospectrum_dense",
    "reduced_pseudospectrum",
    "right_real_outliers",
    "ritz_values",
    "traced_boundary_rate",
]
