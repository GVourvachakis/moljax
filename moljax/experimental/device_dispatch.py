"""Resolution-aware execution-device selection from measured calibrations.

This module is generic across problem size, dimension, and datasets when an
appropriate cost model is supplied.  The included FFT-primitive model uses a
measured FFT and a measured trivial dispatch on each device.  Workloads that
are not FFT dominated should provide their own :class:`CostModel` rather than
assuming that this model applies to them.

Calibration is deliberately explicit: callers provide timing observations,
device primitives, and the backends eligible for a decision.  A policy only
chooses among eligible calibrated backends, uses a caller-supplied optional
feature multiplier, and falls back to a nominated baseline if prediction is
unavailable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import Protocol

import numpy as np

FeatureMultiplier = Callable[[str, str, int], float]


@dataclass(frozen=True)
class DevicePrimitives:
    """Measured device-local primitive costs in seconds."""

    fft_seconds: float
    launch_seconds: float

    def __post_init__(self) -> None:
        if self.fft_seconds <= 0.0 or self.launch_seconds <= 0.0:
            raise ValueError("Primitive timings must be positive")


@dataclass(frozen=True)
class TimingSample:
    """One measured workload/backend/size wall-time observation."""

    workload: str
    backend: str
    size: int
    seconds: float

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.seconds <= 0.0:
            raise ValueError("seconds must be positive")


@dataclass(frozen=True)
class CostModelFit:
    """Fitted model coefficients and in-sample diagnostic information."""

    name: str
    parameters: Mapping[str, float]
    r2: float
    available: bool
    reason: str = ""


class CostModel(Protocol):
    """Pluggable wall-time model used by :func:`fit_device_calibration`."""

    name: str

    def fit(
        self,
        sizes: np.ndarray,
        seconds: np.ndarray,
        primitives: DevicePrimitives | None,
    ) -> CostModelFit:
        """Fit the model to positive sizes and observed wall times."""

    def predict(
        self,
        fit: CostModelFit,
        size: int,
        primitives: DevicePrimitives | None,
    ) -> float | None:
        """Predict wall time for one size, or ``None`` when unusable."""


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total == 0.0:
        return 1.0 if np.allclose(observed, predicted) else 0.0
    return 1.0 - float(np.sum((observed - predicted) ** 2)) / total


@dataclass(frozen=True)
class PowerLawCostModel:
    """Baseline wall-time model ``beta * M**gamma``."""

    name: str = "power_law"

    def fit(
        self,
        sizes: np.ndarray,
        seconds: np.ndarray,
        primitives: DevicePrimitives | None = None,
    ) -> CostModelFit:
        del primitives
        if sizes.size < 2:
            return CostModelFit(self.name, {}, float("nan"), False, "requires two sizes")
        gamma, log_beta = np.polyfit(np.log(sizes), np.log(seconds), deg=1)
        beta = float(np.exp(log_beta))
        predicted = beta * sizes**gamma
        return CostModelFit(
            self.name,
            {"beta": beta, "gamma": float(gamma)},
            _r_squared(seconds, predicted),
            bool(isfinite(beta) and beta > 0.0 and isfinite(float(gamma))),
        )

    def predict(
        self,
        fit: CostModelFit,
        size: int,
        primitives: DevicePrimitives | None = None,
    ) -> float | None:
        del primitives
        if not fit.available or size <= 0:
            return None
        try:
            value = float(fit.parameters["beta"]) * float(size) ** float(fit.parameters["gamma"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return value if isfinite(value) and value > 0.0 else None


@dataclass(frozen=True)
class FFTPrimitiveCostModel:
    """FFT-dominated model normalized by measured device primitives.

    The fitted form is ``alpha * f(M) * T_fft + beta * T_launch``, where
    ``f(M) = (M / M0) * log(1 + M / M0)``.  The non-negative fit makes the
    model portable across devices once their own primitives are measured.
    """

    m0: float = 10_000.0
    name: str = "fft_primitive"

    def _basis(self, sizes: np.ndarray, primitives: DevicePrimitives) -> np.ndarray:
        scaled = sizes / self.m0
        return np.column_stack(
            (
                scaled * np.log1p(scaled) * primitives.fft_seconds,
                np.full_like(sizes, primitives.launch_seconds),
            )
        )

    @staticmethod
    def _nonnegative_fit(basis: np.ndarray, seconds: np.ndarray) -> np.ndarray:
        unconstrained, *_ = np.linalg.lstsq(basis, seconds, rcond=None)
        candidates = [np.zeros(2, dtype=np.float64)]
        if np.all(unconstrained >= 0.0):
            candidates.append(unconstrained)
        for column in range(2):
            coefficient = max(
                0.0,
                float(
                    np.dot(basis[:, column], seconds) / np.dot(basis[:, column], basis[:, column])
                ),
            )
            candidate = np.zeros(2, dtype=np.float64)
            candidate[column] = coefficient
            candidates.append(candidate)
        return min(
            candidates, key=lambda candidate: float(np.sum((basis @ candidate - seconds) ** 2))
        )

    def fit(
        self,
        sizes: np.ndarray,
        seconds: np.ndarray,
        primitives: DevicePrimitives | None,
    ) -> CostModelFit:
        if primitives is None:
            return CostModelFit(self.name, {}, float("nan"), False, "missing device primitives")
        if sizes.size < 2:
            return CostModelFit(self.name, {}, float("nan"), False, "requires two sizes")
        basis = self._basis(sizes, primitives)
        alpha, beta = self._nonnegative_fit(basis, seconds)
        predicted = basis @ np.asarray([alpha, beta])
        return CostModelFit(
            self.name,
            {"alpha": float(alpha), "beta": float(beta), "m0": self.m0},
            _r_squared(seconds, predicted),
            bool(np.all(np.isfinite(predicted)) and np.all(predicted > 0.0)),
        )

    def predict(
        self,
        fit: CostModelFit,
        size: int,
        primitives: DevicePrimitives | None,
    ) -> float | None:
        if not fit.available or primitives is None or size <= 0:
            return None
        try:
            m0 = float(fit.parameters["m0"])
            scaled = float(size) / m0
            value = float(fit.parameters["alpha"]) * scaled * log(1.0 + scaled)
            value *= primitives.fft_seconds
            value += float(fit.parameters["beta"]) * primitives.launch_seconds
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return value if isfinite(value) and value > 0.0 else None


@dataclass(frozen=True)
class BackendCalibration:
    """One backend's fitted model and the primitives used to evaluate it."""

    backend: str
    model: CostModel
    fit: CostModelFit
    primitives: DevicePrimitives | None

    def predict(self, size: int) -> float | None:
        """Evaluate this backend's fitted model."""
        return self.model.predict(self.fit, size, self.primitives)


@dataclass(frozen=True)
class DeviceCalibration:
    """Calibrated cost curves for one named workload class."""

    workload: str
    backends: Mapping[str, BackendCalibration]

    def predict(
        self,
        size: int,
        *,
        eligible_backends: Sequence[str] | None = None,
        feature_multiplier: FeatureMultiplier | None = None,
    ) -> dict[str, float]:
        """Return finite predictions for eligible calibrated backends."""
        eligible = set(eligible_backends) if eligible_backends is not None else None
        result: dict[str, float] = {}
        for backend, calibration in self.backends.items():
            if eligible is not None and backend not in eligible:
                continue
            value = calibration.predict(size)
            if value is None:
                continue
            if feature_multiplier is not None:
                multiplier = float(feature_multiplier(self.workload, backend, size))
                value *= multiplier
            if isfinite(value) and value > 0.0:
                result[backend] = value
        return result


def fit_device_calibration(
    samples: Sequence[TimingSample],
    *,
    workload: str,
    models: Mapping[str, CostModel],
    primitives: Mapping[str, DevicePrimitives] | None = None,
) -> DeviceCalibration:
    """Fit per-backend cost models for one workload class.

    ``models`` selects a pluggable model for each backend.  Backends absent
    from it use :class:`PowerLawCostModel`, making a conventional baseline
    available without imposing the FFT model on unrelated workload types.
    """
    grouped: dict[str, list[TimingSample]] = {}
    for sample in samples:
        if sample.workload == workload:
            grouped.setdefault(sample.backend, []).append(sample)
    calibrations: dict[str, BackendCalibration] = {}
    for backend, backend_samples in grouped.items():
        ordered = sorted(backend_samples, key=lambda sample: sample.size)
        sizes = np.asarray([sample.size for sample in ordered], dtype=np.float64)
        seconds = np.asarray([sample.seconds for sample in ordered], dtype=np.float64)
        model = models.get(backend, PowerLawCostModel())
        primitive = None if primitives is None else primitives.get(backend)
        calibrations[backend] = BackendCalibration(
            backend=backend,
            model=model,
            fit=model.fit(sizes, seconds, primitive),
            primitives=primitive,
        )
    return DeviceCalibration(workload=workload, backends=calibrations)


@dataclass(frozen=True)
class DispatchDecision:
    """One transparent backend-selection result."""

    backend: str
    predicted_seconds: Mapping[str, float]
    calibration_used: bool
    hysteresis_applied: bool
    fallback_used: bool


@dataclass(frozen=True)
class DispatchPolicy:
    """Select the least-cost eligible backend with a safe baseline fallback."""

    fallback_backend: str = "baseline"
    hysteresis_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.hysteresis_fraction < 1.0:
            raise ValueError("hysteresis_fraction must lie in [0, 1)")

    def select(
        self,
        calibration: DeviceCalibration,
        *,
        size: int,
        eligible_backends: Sequence[str],
        previous_backend: str | None = None,
        feature_multiplier: FeatureMultiplier | None = None,
    ) -> DispatchDecision:
        """Select an eligible calibrated backend for one resolution.

        Eligibility is supplied by the caller so selection can be limited to
        backends actually available, or actually measured for held-out
        validation.  Missing predictions select the nominated fallback when it
        is eligible; otherwise selection fails explicitly rather than guessing.
        """
        eligible = tuple(dict.fromkeys(eligible_backends))
        predictions = calibration.predict(
            size,
            eligible_backends=eligible,
            feature_multiplier=feature_multiplier,
        )
        if not predictions:
            if self.fallback_backend in eligible:
                return DispatchDecision(
                    self.fallback_backend,
                    {},
                    calibration_used=False,
                    hysteresis_applied=False,
                    fallback_used=True,
                )
            raise ValueError("No eligible backend has a usable calibration")
        selected = min(predictions, key=predictions.__getitem__)
        hysteresis_applied = False
        if previous_backend in predictions and previous_backend != selected:
            previous = predictions[previous_backend]
            best = predictions[selected]
            fallback_prediction = predictions.get(self.fallback_backend, float("inf"))
            if (
                previous <= (1.0 + self.hysteresis_fraction) * best
                and previous <= fallback_prediction
            ):
                selected = previous_backend
                hysteresis_applied = True
        return DispatchDecision(
            selected,
            predictions,
            calibration_used=True,
            hysteresis_applied=hysteresis_applied,
            fallback_used=False,
        )


def q_squared(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Return held-out time-domain Q-squared, ``1 - SSE / SST``."""
    observed_array = np.asarray(observed, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if observed_array.shape != predicted_array.shape or observed_array.size == 0:
        raise ValueError("observed and predicted must be non-empty matching arrays")
    total = float(np.sum((observed_array - np.mean(observed_array)) ** 2))
    return (
        1.0
        if total == 0.0
        else 1.0 - float(np.sum((observed_array - predicted_array) ** 2)) / total
    )


def l2_error_seconds(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Return the L2 prediction error in seconds."""
    observed_array = np.asarray(observed, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if observed_array.shape != predicted_array.shape:
        raise ValueError("observed and predicted must have matching shapes")
    return float(np.linalg.norm(observed_array - predicted_array))


@dataclass(frozen=True)
class HoldoutResult:
    """One leave-one-resolution-out dispatch decision."""

    workload: str
    size: int
    selected_backend: str
    fastest_backend: str
    regret: float
    predicted_seconds: float
    actual_seconds: float


@dataclass(frozen=True)
class ModelSummary:
    """Held-out prediction quality for one workload/backend pair."""

    workload: str
    backend: str
    n_sizes: int
    q2: float
    l2_error_seconds: float


@dataclass(frozen=True)
class CaseSummary:
    """Held-out regret summary for one workload class."""

    workload: str
    n_sizes: int
    mean_regret: float
    max_regret: float


@dataclass(frozen=True)
class LeaveOneSizeOutResult:
    """All held-out decisions and aggregate calibration-quality diagnostics."""

    holdouts: tuple[HoldoutResult, ...]
    model_summaries: tuple[ModelSummary, ...]
    case_summaries: tuple[CaseSummary, ...]
    overall_mean_regret: float
    overall_max_regret: float


def leave_one_size_out(
    samples: Sequence[TimingSample],
    *,
    models: Mapping[str, CostModel],
    primitives: Mapping[str, DevicePrimitives] | None,
    policy: DispatchPolicy,
) -> LeaveOneSizeOutResult:
    """Regenerate held-out regret, Q-squared, and L2 diagnostics.

    Every size is held out as a complete workload case: all backend timings at
    that resolution are excluded before fitting.  The policy is then restricted
    to backends actually observed at the held-out size.
    """
    workloads = sorted({sample.workload for sample in samples})
    holdouts: list[HoldoutResult] = []
    summaries: list[ModelSummary] = []
    case_summaries: list[CaseSummary] = []
    all_regrets: list[float] = []
    for workload in workloads:
        workload_samples = [sample for sample in samples if sample.workload == workload]
        sizes = sorted({sample.size for sample in workload_samples})
        observed_by_backend: dict[str, list[float]] = {}
        predicted_by_backend: dict[str, list[float]] = {}
        regrets: list[float] = []
        for held_size in sizes:
            training = [
                sample
                for sample in samples
                if not (sample.workload == workload and sample.size == held_size)
            ]
            calibration = fit_device_calibration(
                training,
                workload=workload,
                models=models,
                primitives=primitives,
            )
            actual = {
                sample.backend: sample.seconds
                for sample in workload_samples
                if sample.size == held_size
            }
            decision = policy.select(
                calibration,
                size=held_size,
                eligible_backends=tuple(actual),
            )
            selected_actual = actual[decision.backend]
            fastest_backend = min(actual, key=actual.__getitem__)
            regret = selected_actual / actual[fastest_backend]
            holdouts.append(
                HoldoutResult(
                    workload,
                    held_size,
                    decision.backend,
                    fastest_backend,
                    regret,
                    float(decision.predicted_seconds.get(decision.backend, float("nan"))),
                    selected_actual,
                )
            )
            regrets.append(regret)
            all_regrets.append(regret)
            transparent = calibration.predict(held_size)
            for backend, actual_seconds in actual.items():
                if backend in transparent:
                    observed_by_backend.setdefault(backend, []).append(actual_seconds)
                    predicted_by_backend.setdefault(backend, []).append(transparent[backend])
        for backend in sorted(observed_by_backend):
            observed = observed_by_backend[backend]
            predicted = predicted_by_backend[backend]
            summaries.append(
                ModelSummary(
                    workload,
                    backend,
                    len(observed),
                    q_squared(observed, predicted),
                    l2_error_seconds(observed, predicted),
                )
            )
        case_summaries.append(
            CaseSummary(workload, len(sizes), float(np.mean(regrets)), float(np.max(regrets)))
        )
    return LeaveOneSizeOutResult(
        tuple(holdouts),
        tuple(summaries),
        tuple(case_summaries),
        float(np.mean(all_regrets)),
        float(np.max(all_regrets)),
    )
