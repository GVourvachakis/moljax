"""Unit tests for generic resolution-aware device dispatch."""

import math

import numpy as np

from moljax.core.device_dispatch import (
    DevicePrimitives,
    DispatchPolicy,
    FFTPrimitiveCostModel,
    PowerLawCostModel,
    TimingSample,
    fit_device_calibration,
    leave_one_size_out,
)


def test_power_law_calibration_recovers_known_curve():
    """The baseline model should recover a noiseless synthetic power law."""
    samples = [
        TimingSample("synthetic", "baseline", size, 2.5e-6 * size**1.3)
        for size in (100, 200, 400, 800)
    ]
    calibration = fit_device_calibration(
        samples,
        workload="synthetic",
        models={"baseline": PowerLawCostModel()},
    )
    prediction = calibration.predict(1600)["baseline"]
    assert math.isclose(prediction, 2.5e-6 * 1600**1.3, rel_tol=1e-10)


def test_fft_primitive_calibration_recovers_known_curve():
    """The non-negative primitive model should recover its synthetic curve."""
    primitives = DevicePrimitives(fft_seconds=2.0e-5, launch_seconds=3.0e-6)
    model = FFTPrimitiveCostModel(m0=100.0)
    samples = []
    for size in (100, 200, 400, 800):
        scaled = size / 100.0
        seconds = 4.0 * scaled * np.log1p(scaled) * primitives.fft_seconds
        seconds += 2.0 * primitives.launch_seconds
        samples.append(TimingSample("synthetic", "gpu", size, seconds))
    calibration = fit_device_calibration(
        samples,
        workload="synthetic",
        models={"gpu": model},
        primitives={"gpu": primitives},
    )
    assert math.isclose(calibration.predict(400)["gpu"], samples[2].seconds, rel_tol=1e-10)


def test_policy_applies_hysteresis_and_safe_fallback():
    """Selection should retain a near-tied prior backend and use fallback safely."""
    samples = [
        TimingSample("synthetic", "baseline", 100, 1.0),
        TimingSample("synthetic", "baseline", 200, 2.0),
        TimingSample("synthetic", "gpu", 100, 0.99),
        TimingSample("synthetic", "gpu", 200, 1.98),
    ]
    calibration = fit_device_calibration(
        samples,
        workload="synthetic",
        models={"baseline": PowerLawCostModel(), "gpu": PowerLawCostModel()},
    )
    policy = DispatchPolicy(fallback_backend="baseline", hysteresis_fraction=0.05)
    decision = policy.select(
        calibration,
        size=100,
        eligible_backends=("baseline", "gpu"),
        previous_backend="baseline",
    )
    assert decision.backend == "baseline"
    assert decision.hysteresis_applied
    fallback = policy.select(
        calibration,
        size=100,
        eligible_backends=("baseline",),
    )
    assert fallback.backend == "baseline"


def test_leave_one_size_out_computes_known_regret():
    """Held-out dispatch must choose only measured options and report regret."""
    samples = []
    for size, cpu, gpu in ((10, 1.0, 2.0), (20, 2.0, 1.0), (40, 4.0, 0.5)):
        samples.extend(
            (
                TimingSample("synthetic", "cpu", size, cpu),
                TimingSample("synthetic", "gpu", size, gpu),
            )
        )
    result = leave_one_size_out(
        samples,
        models={"cpu": PowerLawCostModel(), "gpu": PowerLawCostModel()},
        primitives=None,
        policy=DispatchPolicy(fallback_backend="cpu"),
    )
    assert len(result.holdouts) == 3
    assert result.overall_mean_regret >= 1.0
    assert all(holdout.regret >= 1.0 for holdout in result.holdouts)
