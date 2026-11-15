"""Opt-in controller presets, legacy parity, and driver cache behaviour."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moljax import FilterParams as PublicFilterParams
from moljax.core.dt_policy import (
    CONTROLLER_PRESETS,
    ControllerState,
    FilterParams,
    PIDParams,
    filter_controller_dt,
    pid_controller_dt,
)
from moljax.core.fft_integrators import _ETD_ADAPTIVE_DRIVERS, etd_integrate_adaptive
from moljax.core.fft_operators import DiffusionOperator
from moljax.core.grid import Grid1D, Grid2D
from moljax.core.model import create_gray_scott_periodic_fft
from moljax.core.stepping import (
    _ADAPTIVE_DRIVERS,
    _ADAPTIVE_IMEX_DRIVERS,
    IntegratorType,
    adaptive_integrate,
    adaptive_integrate_imex,
    clear_compiled_drivers,
)


def test_presets_and_characteristic_roots():
    assert PublicFilterParams is FilterParams
    expected = {
        "PI30": (1 / 3, 0, 0),
        "PI42": (0.6, -0.2, 0),
        "PI33": (2 / 3, -1 / 3, 0),
        "PI34": (0.7, -0.4, 0),
        "H312PID": (1 / 18, 1 / 9, 1 / 18),
        "moljax_pi": (1.1, -0.4, 0),
    }
    assert set(CONTROLLER_PRESETS) == set(expected)
    for name, coefficients in expected.items():
        p = CONTROLLER_PRESETS[name]
        assert (p.b1, p.b2, p.b3) == pytest.approx(coefficients)
        roots = np.roots([1, -(1 - p.b1), p.b2, p.b3])
        assert max(abs(roots)) < 1
        if name == "moljax_pi":
            np.testing.assert_allclose(sorted(roots), [-0.684, 0, 0.584], atol=1e-3)
        if name == "PI34":
            np.testing.assert_allclose(sorted(roots), [-0.5, 0, 0.8], atol=1e-3)
    p = CONTROLLER_PRESETS["moljax_pi"]
    assert max(abs(np.roots([1, -(1 - 2 * p.b1), 2 * p.b2, 0]))) > 1


def test_from_pid_matches_legacy_on_2000_draws():
    rng = np.random.default_rng(1007)
    dt = jnp.asarray(10 ** rng.uniform(-11, 1, 2000))
    err = jnp.asarray(10 ** rng.uniform(-12, 5, 2000))
    prev = jnp.asarray(10 ** rng.uniform(-12, 5, 2000))
    prevprev = jnp.asarray(10 ** rng.uniform(-12, 5, 2000))
    order = jnp.asarray(rng.integers(1, 6, size=2000))
    state = ControllerState(prev, prevprev, jnp.zeros(2000, dtype=jnp.int32))
    pid = PIDParams()
    old_dt, old_state = pid_controller_dt(dt, err, state, pid, order)
    new_dt, new_state = filter_controller_dt(dt, err, state, FilterParams.from_pid(pid), order)
    np.testing.assert_allclose(new_dt, old_dt, rtol=1e-14, atol=0)
    for a, b in zip(new_state, old_state, strict=True):
        np.testing.assert_array_equal(a, b)
    # Both the growth and shrink limits are exercised.
    assert bool(jnp.any(new_dt == jnp.clip(dt * pid.min_factor, pid.dt_min, pid.dt_max)))
    assert bool(jnp.any(new_dt == jnp.clip(dt * pid.max_factor, pid.dt_min, pid.dt_max)))


def test_growth_ceiling_after_rejection():
    state = ControllerState(jnp.array(1.0), jnp.array(1.0), jnp.array(1))
    p = FilterParams(0.7, -0.4, 0.0, max_factor_after_reject=1.0)
    dt, new_state = filter_controller_dt(jnp.array(0.1), jnp.array(1e-8), state, p, 5)
    assert float(dt) == pytest.approx(0.1)
    assert int(new_state.consecutive_rejects) == 0


def test_float32_step_dtype():
    state = ControllerState(
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(0),
    )
    dt, _ = filter_controller_dt(
        jnp.array(0.01, dtype=jnp.float32),
        jnp.array(0.25, dtype=jnp.float32),
        state,
        CONTROLLER_PRESETS["PI34"],
        5,
    )
    assert dt.dtype == jnp.float32


@pytest.mark.parametrize("driver", ["mol", "imex", "etd"])
def test_drivers_default_and_cache(driver):
    clear_compiled_drivers()
    if driver == "etd":
        grid = Grid1D.uniform(16, 0, 1)
        y0 = {"u": jnp.ones(16)}
        op = {"u": DiffusionOperator(grid, 0.01)}

        def rhs(state, t):
            return {"u": -0.1 * state["u"]}

        def call(**kwargs):
            return etd_integrate_adaptive(
                y0,
                (0.0, 0.125),
                0.0625,
                op,
                rhs,
                method="etd1",
                max_steps=16,
                save_every=16,
                **kwargs,
            )

        cache = _ETD_ADAPTIVE_DRIVERS
    else:
        grid = Grid2D.uniform(8, 8, 0, 2.5, 0, 2.5, n_ghost=1)
        model, fft_cache, ds = create_gray_scott_periodic_fft(grid)
        y0 = model.create_initial_state(fill_values={"u": 1.0, "v": 0.0})
        if driver == "mol":

            def call(**kwargs):
                return adaptive_integrate(
                    model,
                    y0,
                    0.0,
                    0.01,
                    0.005,
                    method=IntegratorType.RK4,
                    max_steps=16,
                    save_every=16,
                    **kwargs,
                )

            cache = _ADAPTIVE_DRIVERS
        else:

            def call(**kwargs):
                return adaptive_integrate_imex(
                    model,
                    y0,
                    0.0,
                    0.01,
                    0.005,
                    fft_cache,
                    ds,
                    max_steps=16,
                    save_every=16,
                    **kwargs,
                )

            cache = _ADAPTIVE_IMEX_DRIVERS
    default = call()
    assert len(cache) == 1
    explicit_none = call(controller=None)
    assert len(cache) == 1
    for key in default.y_final:
        np.testing.assert_array_equal(default.y_final[key], explicit_none.y_final[key])
    for name, expected_size in (("PI34", 2), ("PI34", 2), ("H312PID", 3)):
        result = call(controller=name)
        assert int(result.status) == 0
        assert len(cache) == expected_size
    with pytest.raises(ValueError, match="unknown controller"):
        call(controller="unknown")
    clear_compiled_drivers()
