"""Smoke tests for the optional conditioning-report plotting API."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg", force=True)

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from moljax.conditioning import (
    FieldOfValuesResult,
    PseudospectraResult,
    plot_numerical_range,
    plot_pseudospectrum,
    plot_rate_scaling,
    plot_residual_envelope,
)


def _field_of_values() -> FieldOfValuesResult:
    """Return a small synthetic numerical-range result for plotting."""
    angles = np.linspace(0.0, 2.0 * math.pi, 9, endpoint=False)
    boundary = 2.0 + 0.25 * np.exp(1j * angles)
    return FieldOfValuesResult(
        boundary=jnp.asarray(boundary),
        center=2.0 + 0.0j,
        radius=0.25,
        disk_rate=0.125,
        origin_enclosed=False,
        cp_prefactor=1.0 + math.sqrt(2.0),
    )


def _pseudospectra() -> PseudospectraResult:
    """Return a compact synthetic pseudospectrum for plotting."""
    real = jnp.linspace(0.5, 2.5, 4)
    imag = jnp.linspace(-1.0, 1.0, 3)
    sigma_min = jnp.asarray([[0.8, 0.4, 0.8, 1.2], [0.5, 0.1, 0.5, 1.0], [0.8, 0.4, 0.8, 1.2]])
    return PseudospectraResult(
        real_grid=real,
        imag_grid=imag,
        sigma_min=sigma_min,
        ritz_values=jnp.asarray([1.2 + 0.1j, 1.8 - 0.2j]),
        epsilon_zero=0.1,
    )


def test_conditioning_figures_return_populated_figures():
    """Each public plotting helper returns a matplotlib figure with axes."""
    figures = [
        plot_numerical_range(_field_of_values()),
        plot_pseudospectrum(_pseudospectra()),
        plot_rate_scaling([16, 64], [0.4, 0.2], [0.3, 0.15], [0.2, 0.1], measured=[8, 5]),
        plot_residual_envelope([1.0, 0.2, 0.04], 0.25),
    ]
    try:
        assert all(isinstance(figure, Figure) and figure.axes for figure in figures)
    finally:
        for figure in figures:
            plt.close(figure)
