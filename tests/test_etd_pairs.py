"""Independent checks of Zoto--Bowman Tables 9 and 10 (arXiv:2303.12139v2)."""

import math
from fractions import Fraction
from functools import cache

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import moljax.core.fft_integrators as fi
from moljax.core.dt_policy import PIDParams
from moljax.core.etd_pairs import ERK32ZB, ERK43ZB, pair_step_at
from moljax.core.fft_operators import DiffusionOperator
from moljax.core.grid import Grid2D


@cache
def _phi(k, z):
    if k == 0:
        return math.exp(z)
    a = np.zeros((k + 1, k + 1))
    a[0, 0] = z
    for j in range(k):
        a[j, j + 1] = 1
    return expm(a)[0, k]


def _value(expression, z):
    return sum(float(coef) * _phi(k, float(arg) * z) for coef, k, arg in expression)


def _exact(expression):
    return sum((coef / math.factorial(k) for coef, k, _ in expression), Fraction())


@pytest.mark.parametrize("tableau", [ERK32ZB, ERK43ZB], ids=lambda t: t.name)
def test_row_sums_and_classical_orders(tableau):
    expressions = [expression for row in tableau.a for expression in row]
    expressions.extend(tableau.b)
    expressions.extend(tableau.bhat)
    assert max(k for expression in expressions for _, k, _ in expression) <= 3
    if tableau is ERK32ZB:
        assert set(tableau.c) <= {Fraction(i, 4) for i in range(5)}
    for z in np.r_[0.0, -np.logspace(-6, 4, 200)]:
        for i in range(1, len(tableau.c)):
            expected = float(tableau.c[i]) * _phi(1, float(tableau.c[i]) * z)
            actual = sum(_value(expression, z) for expression in tableau.a[i])
            assert abs(actual - expected) <= 1e-12 * abs(expected)
        for weights in (tableau.b, tableau.bhat):
            expected = _phi(1, z)
            actual = sum(_value(expression, z) for expression in weights)
            assert abs(actual - expected) <= 1e-12 * abs(expected)

    c = tableau.c
    a = [[_exact(expression) for expression in row] for row in tableau.a]
    ac = [sum((a[i][j] * c[j] for j in range(i)), Fraction()) for i in range(len(c))]
    ac2 = [sum((a[i][j] * c[j] ** 2 for j in range(i)), Fraction()) for i in range(len(c))]
    aac = [sum((a[i][j] * ac[j] for j in range(i)), Fraction()) for i in range(len(c))]
    for weights, order in (
        (tableau.b, tableau.advancing_order),
        (tableau.bhat, tableau.embedded_order),
    ):
        b = [_exact(expression) for expression in weights]
        conditions = {
            1: [sum(b) - 1],
            2: [sum(b[i] * c[i] for i in range(len(c))) - Fraction(1, 2)],
            3: [
                sum(b[i] * c[i] ** 2 for i in range(len(c))) - Fraction(1, 3),
                sum(b[i] * ac[i] for i in range(len(c))) - Fraction(1, 6),
            ],
            4: [
                sum(b[i] * c[i] ** 3 for i in range(len(c))) - Fraction(1, 4),
                sum(b[i] * c[i] * ac[i] for i in range(len(c))) - Fraction(1, 8),
                sum(b[i] * ac2[i] for i in range(len(c))) - Fraction(1, 12),
                sum(b[i] * aac[i] for i in range(len(c))) - Fraction(1, 24),
            ],
        }
        assert all(v == 0 for degree in range(1, order + 1) for v in conditions[degree])
        if order < 4:
            assert any(v != 0 for v in conditions[order + 1])


def _stiff_residuals(tableau, weights, z):
    c = [float(x) for x in tableau.c]
    a = [[_value(expression, z) for expression in row] for row in tableau.a]
    b = [_value(expression, z) for expression in weights]
    n = len(c)

    def psi(j, i):
        return c[i] ** j * _phi(j, c[i] * z) - sum(
            a[i][k] * c[k] ** (j - 1) / math.factorial(j - 1) for k in range(i)
        )

    return [
        _phi(1, z) - sum(b),
        _phi(2, z) - sum(b[i] * c[i] for i in range(n)),
        max(abs(psi(1, i)) for i in range(1, n)),
        _phi(3, z) - sum(b[i] * c[i] ** 2 / 2 for i in range(n)),
        sum(b[i] * psi(2, i) for i in range(1, n)),
        _phi(4, z) - sum(b[i] * c[i] ** 3 / 6 for i in range(n)),
        sum(b[i] * psi(3, i) for i in range(1, n)),
        # Table 4 prints psi_{2,i}; its nested j sum requires psi_{2,j}.
        sum(b[i] * sum(a[i][j] * psi(2, j) for j in range(1, i)) for i in range(1, n)),
        sum(b[i] * c[i] * psi(2, i) for i in range(1, n)),
    ]


@pytest.mark.parametrize("tableau", [ERK32ZB, ERK43ZB], ids=lambda t: t.name)
def test_table_four_stiff_psi_conditions(tableau):
    # The paper states that the high estimates satisfy their top-order
    # conditions weakly. Thus all required conditions hold at z=0; only
    # lower-order conditions hold as functions of z over the full grid.
    for high, weights, order in (
        (True, tableau.b, tableau.advancing_order),
        (False, tableau.bhat, tableau.embedded_order),
    ):
        required = {1: (0,), 2: (0, 1, 2), 3: (0, 1, 2, 3, 4), 4: tuple(range(9))}[order]
        assert max(abs(_stiff_residuals(tableau, weights, 0)[i]) for i in required) < 1e-11
        strong_order = (2 if tableau.name == "erk32zb" else 3) if high else order - 1
        strong = {1: (0,), 2: (0, 1, 2), 3: (0, 1, 2, 3, 4)}[strong_order]
        for z in np.r_[0.0, -np.logspace(-6, 4, 200)]:
            r = _stiff_residuals(tableau, weights, z)
            if order >= strong_order:
                assert max(abs(r[i]) for i in strong) < 1e-11


@pytest.mark.parametrize("tableau", [ERK32ZB, ERK43ZB], ids=lambda t: t.name)
def test_high_order_weak_stiff_conditions(tableau):
    # In the paper's weak formulation, the outer weights are evaluated at
    # zero; the inner stage functions retain their dependence on z.
    b0 = [_value(w, 0.0) for w in tableau.b]
    c = [float(x) for x in tableau.c]
    for z in np.r_[0.0, -np.logspace(-6, 4, 200)]:
        a = [[_value(expression, z) for expression in row] for row in tableau.a]

        def psi(j, i, z=z, a=a):
            return c[i] ** j * _phi(j, c[i] * z) - sum(
                a[i][k] * c[k] ** (j - 1) / math.factorial(j - 1) for k in range(i)
            )

        weak = [sum(b0[i] * psi(2, i) for i in range(1, len(c)))]
        if tableau is ERK43ZB:
            weak.extend(
                [
                    sum(b0[i] * psi(3, i) for i in range(1, len(c))),
                    sum(
                        b0[i] * sum(a[i][j] * psi(2, j) for j in range(1, i))
                        for i in range(1, len(c))
                    ),
                    sum(b0[i] * c[i] * psi(2, i) for i in range(1, len(c))),
                ]
            )
        assert max(abs(v) for v in weak) < 1e-11

    # Table 4 permits arbitrary, noncommuting J and K. Check its matrix
    # products as well as the scalar-grid reductions above.
    j = np.array([[0.2, 1.0], [-0.7, 0.3]])
    k = np.array([[0.5, -0.2], [1.1, 0.1]])
    zero = np.zeros((2, 2))
    for z in (-1e-6, -0.01, -1.0, -100.0):
        spectrum = (z, 2 * z)
        a = [[np.diag([_value(expr, zz) for zz in spectrum]) for expr in row] for row in tableau.a]

        def psi_matrix(degree, i, spectrum=spectrum, a=a):
            first = np.diag([c[i] ** degree * _phi(degree, c[i] * zz) for zz in spectrum])
            return first - sum(
                (a[i][q] * c[q] ** (degree - 1) / math.factorial(degree - 1) for q in range(i)),
                zero,
            )

        c5 = sum((b0[i] * j @ psi_matrix(2, i) for i in range(1, len(c))), zero)
        conditions = [c5]
        if tableau is ERK43ZB:
            conditions.extend(
                [
                    sum((b0[i] * j @ psi_matrix(3, i) for i in range(1, len(c))), zero),
                    sum(
                        (
                            b0[i]
                            * j
                            @ sum((a[i][q] @ j @ psi_matrix(2, q) for q in range(1, i)), zero)
                            for i in range(1, len(c))
                        ),
                        zero,
                    ),
                    sum((b0[i] * c[i] * k @ psi_matrix(2, i) for i in range(1, len(c))), zero),
                ]
            )
        assert max(np.linalg.norm(v, ord=np.inf) for v in conditions) < 1e-11


@pytest.mark.parametrize("tableau", [ERK32ZB, ERK43ZB], ids=lambda t: t.name)
@pytest.mark.parametrize(
    "lam,step_counts",
    [
        (-1.0, (40, 80, 160)),
        (-100.0, (40, 80, 160)),
        (-10000.0, (1280, 2560, 5120)),
    ],
)
def test_scalar_convergence(tableau, lam, step_counts):
    # y' = lambda*y - y² + f(t), exact y = 1 + 0.1 sin(t).

    def rhs(y, t):
        exact = 1 + 0.1 * math.sin(t)
        return -y * y + 0.1 * math.cos(t) - lam * exact + exact * exact

    for weights, order in (
        (tableau.b, tableau.advancing_order),
        (tableau.bhat, tableau.embedded_order),
    ):
        errors = []
        for steps in step_counts:
            h = 1 / steps
            y = 1.0
            for step in range(steps):
                z = lam * h
                stage = [y]
                for i in range(1, len(tableau.c)):
                    stage.append(
                        math.exp(float(tableau.c[i]) * z) * y
                        + h
                        * sum(
                            _value(tableau.a[i][j], z)
                            * rhs(stage[j], (step + float(tableau.c[j])) * h)
                            for j in range(i)
                        )
                    )
                y = math.exp(z) * y + h * sum(
                    _value(weight, z) * rhs(stage[i], (step + float(tableau.c[i])) * h)
                    for i, weight in enumerate(weights)
                )
            errors.append(abs(y - (1 + 0.1 * math.sin(1))))
        slope = math.log2(errors[1] / errors[2])
        if lam == -10000:
            assert all(b < a for a, b in zip(errors, errors[1:], strict=False))
            assert slope >= order - 0.7, (tableau.name, order, errors, slope)
        else:
            assert abs(slope - order) <= 0.3, (tableau.name, order, errors, slope)


def _fixed_pair(y, op, rhs, tableau, steps, t_end):
    h = t_end / steps
    for i in range(steps):
        times = tuple((i + float(c)) * h for c in tableau.c)
        y, _ = pair_step_at(y, times, h, op, rhs, tableau)
    return y


def test_manufactured_2d_order_and_estimator():
    grid = Grid2D.uniform(16, 16, 0, 1, 0, 1)
    op = DiffusionOperator(grid, 0.01)
    x = jnp.arange(16) / 16
    s = jnp.sin(2 * jnp.pi * x[:, None]) * jnp.sin(2 * jnp.pi * x[None, :])
    eigenvalue = float(jnp.vdot(s, op.matvec(s)) / jnp.vdot(s, s))
    assert np.max(np.abs(np.asarray(op.matvec(s) - eigenvalue * s))) < 1e-12

    def exact(t):
        return (1 + 0.5 * jnp.cos(t)) * s

    def rhs(y, t):
        target = exact(t)
        return {"u": -(y["u"] ** 3) - 0.5 * jnp.sin(t) * s - eigenvalue * target + target**3}

    y0 = {"u": exact(0)}
    errors = []
    for steps in (8, 16, 32):
        y = _fixed_pair(y0, {"u": op}, rhs, ERK32ZB, steps, 1.0)
        errors.append(float(jnp.max(jnp.abs(y["u"] - exact(1.0)))))
    assert abs(math.log2(errors[1] / errors[2]) - 3) < 0.3

    for tableau in (ERK32ZB, ERK43ZB):
        differences = []
        for h in (0.0125, 0.00625, 0.003125):
            times = tuple(float(c) * h for c in tableau.c)
            high, low = pair_step_at(y0, times, h, {"u": op}, rhs, tableau)
            differences.append(float(jnp.max(jnp.abs(high["u"] - low["u"]))))
        assert abs(math.log2(differences[1] / differences[2]) - (tableau.embedded_order + 1)) < 0.3


@pytest.mark.parametrize("tableau", [ERK32ZB, ERK43ZB], ids=lambda t: t.name)
def test_gray_scott_autonomous_self_convergence(tableau):
    grid = Grid2D.uniform(32, 32, 0, 1, 0, 1)
    ops = {"u": DiffusionOperator(grid, 2e-5), "v": DiffusionOperator(grid, 1e-5)}
    x = jnp.arange(32)
    spot = (x[:, None] - 16) ** 2 + (x[None, :] - 16) ** 2 < 5**2
    y0 = {"u": 1.0 - 0.4 * spot, "v": 0.2 * spot.astype(jnp.float64)}

    def rhs(y, _):
        uvv = y["u"] * y["v"] ** 2
        return {"u": -uvv + 0.035 * (1 - y["u"]), "v": uvv - 0.1 * y["v"]}

    ends = [_fixed_pair(y0, ops, rhs, tableau, n, 0.5) for n in (4, 8, 16, 32)]
    differences = [
        float(max(jnp.max(jnp.abs(a[k] - b[k])) for k in y0))
        for a, b in zip(ends, ends[1:], strict=False)
    ]
    assert abs(math.log2(differences[1] / differences[2]) - tableau.advancing_order) < 0.4


@pytest.mark.parametrize("method", ["erk32zb", "erk43zb"])
def test_adaptive_smoke_and_cache(method):
    fi.clear_compiled_drivers()
    grid = Grid2D.uniform(8, 8, 0, 1, 0, 1)
    ops = {"u": DiffusionOperator(grid, 0.01)}
    y0 = {"u": jnp.ones((8, 8))}

    def rhs(y, _):
        return {"u": -y["u"]}

    args = (y0, (0.0, 0.5), 0.01, ops, rhs)
    kwargs = dict(
        method=method,
        rhs_uses_time=False,
        pid_params=PIDParams(atol=1e-5, rtol=1e-5),
        max_steps=1000,
        save_every=100,
    )
    first = fi.etd_integrate_adaptive(*args, **kwargs)
    assert bool(first.success)
    assert len(fi._ETD_ADAPTIVE_DRIVERS) == 1
    second = fi.etd_integrate_adaptive(*args, **kwargs)
    assert bool(second.success)
    assert len(fi._ETD_ADAPTIVE_DRIVERS) == 1
    np.testing.assert_array_equal(np.asarray(first.y_final["u"]), np.asarray(second.y_final["u"]))
    if method == "erk43zb":
        with pytest.raises(ValueError, match="tick clock"):
            fi.etd_integrate_adaptive(*args, **(kwargs | {"rhs_uses_time": True}))


def test_compiled_cache_separates_embedded_methods():
    fi.clear_compiled_drivers()
    y = {"u": jnp.ones(4)}

    class ZeroOp:
        eigenvalues = jnp.zeros(4)

    ops = {"u": ZeroOp()}

    def rhs(y, _):
        return {"u": -y["u"]}

    for method, size in (("erk32zb", 1), ("erk43zb", 2), ("erk32zb", 2)):
        result = fi.etd_integrate_adaptive(
            y,
            (0.0, 0.1),
            0.01,
            ops,
            rhs,
            method=method,
            rhs_uses_time=False,
            max_steps=100,
            save_every=100,
        )
        assert bool(result.success)
        assert len(fi._ETD_ADAPTIVE_DRIVERS) == size


@pytest.mark.parametrize("method,expected", [("erk32zb", 3), ("erk43zb", 4)])
def test_embedded_controller_order(monkeypatch, method, expected):
    fi.clear_compiled_drivers()
    seen = []
    original = fi.propose_dt_etd

    def recording(*args, **kwargs):
        seen.append(args[4])
        return original(*args, **kwargs)

    monkeypatch.setattr(fi, "propose_dt_etd", recording)

    class ZeroOp:
        eigenvalues = jnp.zeros(4)

    def rhs(y, _):
        return {"u": -y["u"]}

    result = fi.etd_integrate_adaptive(
        {"u": jnp.ones(4)},
        (0.0, 0.1),
        0.01,
        {"u": ZeroOp()},
        rhs,
        method=method,
        rhs_uses_time=False,
        max_steps=100,
        save_every=100,
    )
    assert bool(result.success)
    assert seen == [expected]


@pytest.mark.parametrize("tableau", [ERK32ZB, ERK43ZB], ids=lambda t: t.name)
def test_adaptive_accepts_advancing_solution(tableau):
    y = {"u": jnp.ones(4)}

    class ZeroOp:
        eigenvalues = jnp.zeros(4)

    ops = {"u": ZeroOp()}

    def rhs(y, _):
        return {"u": -(y["u"] ** 2)}

    h = 0.125
    high, low = pair_step_at(y, tuple(float(c) * h for c in tableau.c), h, ops, rhs, tableau)
    assert not np.array_equal(np.asarray(high["u"]), np.asarray(low["u"]))
    result = fi.etd_integrate_adaptive(
        y,
        (0.0, h),
        h,
        ops,
        rhs,
        method=tableau.name,
        rhs_uses_time=False,
        pid_params=PIDParams(atol=1e8, rtol=1e8, dt_max=h),
        max_steps=1,
    )
    assert bool(result.success)
    np.testing.assert_array_equal(np.asarray(result.y_final["u"]), np.asarray(high["u"]))


@pytest.mark.parametrize(
    "shape,spectrum_shape,is_rfft",
    [((8,), (8,), False), ((8, 8), (8, 8), False), ((8, 8), (8, 5), True)],
)
def test_pair_uses_existing_fft_layouts(shape, spectrum_shape, is_rfft):
    class ZeroOp:
        eigenvalues = jnp.zeros(spectrum_shape)
        _is_rfft = is_rfft

    def rhs(y, _):
        return {"u": -y["u"]}

    y = {"u": jnp.ones(shape)}
    times = tuple(float(c) * 0.1 for c in ERK32ZB.c)
    high, _ = pair_step_at(y, times, 0.1, {"u": ZeroOp()}, rhs, ERK32ZB)
    np.testing.assert_allclose(np.asarray(high["u"]), math.exp(-0.1), rtol=1e-5)


def test_erk32_time_dependent_stages_use_tick_lattice():
    class ZeroOp:
        eigenvalues = jnp.zeros(4)

    def rhs(y, t):
        return {"u": jnp.full_like(y["u"], t)}

    y = {"u": jnp.ones(4)}
    h = 0.125
    expected, _ = pair_step_at(y, (0.0, h / 2, 3 * h / 4, h), h, {"u": ZeroOp()}, rhs, ERK32ZB)
    result = fi.etd_integrate_adaptive(
        y,
        (0.0, h),
        h,
        {"u": ZeroOp()},
        rhs,
        method="erk32zb",
        pid_params=PIDParams(atol=1e8, rtol=1e8, dt_max=h),
        max_steps=1,
    )
    assert bool(result.success)
    np.testing.assert_array_equal(np.asarray(result.y_final["u"]), np.asarray(expected["u"]))
