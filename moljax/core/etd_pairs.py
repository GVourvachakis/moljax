"""Zoto–Bowman embedded exponential RK tableaux (arXiv:2303.12139v2, Tables 9–10).

The paper writes ``y' = F - L_paper y`` and ``phi_k(-c h L_paper)``.
Here ``u' = L_moljax u + N(u)``, with ``L_moljax <= 0``, so each term is
``phi_k(c h L_moljax)``. A term is ``(coefficient, phi index, argument scale)``.

Table 9 prints an undefined ``-a23`` in the first weight of its advancing
row. That row has only three weights, and the authors' symbolic source
(``stiffode/expint``, ``symbolically_discover_ERK32ZB.wls``) has only
``a20, a21, a22``. The defined row is transcribed with ``a20=phi1-a21-a22``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import jax.numpy as jnp

from moljax.core.jit_kernels import phi1, phi2, phi3

Term = tuple[Fraction, int, Fraction]
Expression = tuple[Term, ...]


def _term(coef: Fraction | int, k: int, arg: Fraction | int = 1) -> Expression:
    return ((Fraction(coef), k, Fraction(arg)),)


def _add(*expressions: Expression) -> Expression:
    combined: dict[tuple[int, Fraction], Fraction] = {}
    for expression in expressions:
        for coef, k, arg in expression:
            combined[k, arg] = combined.get((k, arg), Fraction(0)) + coef
    return tuple((coef, k, arg) for (k, arg), coef in sorted(combined.items()) if coef)


def _scale(expression: Expression, factor: Fraction | int) -> Expression:
    return tuple((Fraction(factor) * coef, k, arg) for coef, k, arg in expression)


def _minus(left: Expression, *rights: Expression) -> Expression:
    return _add(left, *(_scale(right, -1) for right in rights))


@dataclass(frozen=True)
class EmbeddedTableau:
    """Stage abscissae, preceding-stage weights, and the two output weights."""

    name: str
    c: tuple[Fraction, ...]
    a: tuple[tuple[Expression, ...], ...]
    b: tuple[Expression, ...]
    bhat: tuple[Expression, ...]
    advancing_order: int
    embedded_order: int


def _erk32zb() -> EmbeddedTableau:
    f = Fraction
    p = _term
    a10 = _scale(p(1, 1, f(1, 2)), f(1, 2))
    a21 = _add(_scale(p(1, 2, f(3, 4)), f(9, 8)), _scale(p(1, 2, f(1, 2)), f(3, 8)))
    a20 = _minus(_scale(p(1, 1, f(3, 4)), f(3, 4)), a21)
    b1 = _add(_scale(p(1, 2), f(3, 4)), _scale(p(1, 3), -f(1, 4)))
    b2 = _add(_scale(p(1, 2), f(5, 6)), _scale(p(1, 3), f(1, 6)))
    b0 = _minus(p(1, 1), b1, b2)
    bh0 = _add(
        _scale(p(1, 1), f(29, 18)),
        _scale(p(1, 1, f(3, 4)), f(7, 6)),
        _scale(p(1, 1, f(1, 2)), f(9, 14)),
        _scale(p(1, 2), f(3, 4)),
        _scale(p(1, 2, f(3, 4)), f(2, 7)),
        _scale(p(1, 2, f(1, 2)), f(1, 12)),
        _scale(p(1, 3), -f(8083, 420)),
        _scale(p(1, 3, f(1, 2)), f(11, 30)),
    )
    bh1 = _add(
        _scale(p(1, 1), -f(1, 9)),
        _scale(p(1, 1, f(3, 4)), -f(1, 6)),
        _scale(p(1, 2), -f(1, 2)),
        _scale(p(1, 2, f(3, 4)), -f(1, 7)),
        _scale(p(1, 2, f(1, 2)), -f(1, 3)),
        _scale(p(1, 3), f(1, 6)),
        _scale(p(1, 3, f(1, 2)), f(1, 6)),
    )
    bh2 = _add(
        _scale(p(1, 1), f(2, 3)),
        _scale(p(1, 1, f(3, 4)), -f(1, 2)),
        _scale(p(1, 1, f(1, 2)), -f(1, 7)),
        _scale(p(1, 2), f(1, 3)),
        _scale(p(1, 2, f(3, 4)), -f(1, 7)),
        _scale(p(1, 3, f(1, 2)), -f(1, 5)),
    )
    bh3 = _add(
        _scale(p(1, 1), -f(7, 6)),
        _scale(p(1, 1, f(3, 4)), -f(1, 2)),
        _scale(p(1, 1, f(1, 2)), -f(1, 2)),
        _scale(p(1, 2), -f(7, 12)),
        _scale(p(1, 2, f(1, 2)), f(1, 4)),
        _scale(p(1, 3), f(2671, 140)),
        _scale(p(1, 3, f(1, 2)), -f(1, 3)),
    )
    return EmbeddedTableau(
        "erk32zb",
        (f(0), f(1, 2), f(3, 4), f(1)),
        ((), (a10,), (a20, a21), (b0, b1, b2)),
        (b0, b1, b2, ()),
        (bh0, bh1, bh2, bh3),
        3,
        2,
    )


def _erk43zb() -> EmbeddedTableau:
    f = Fraction
    p = _term
    a10 = _scale(p(1, 1, f(1, 6)), f(1, 6))
    a21 = _add(_scale(p(1, 2, f(1, 2)), f(3, 2)), _scale(p(1, 2, f(1, 6)), f(1, 2)))
    a20 = _minus(_scale(p(1, 1, f(1, 2)), f(1, 2)), a21)
    a31 = _add(
        _scale(p(1, 1), f(19, 60)),
        _scale(p(1, 1, f(1, 2)), f(1, 2)),
        _scale(p(1, 1, f(1, 6)), f(1, 2)),
        _scale(p(1, 2, f(1, 2)), 2),
        _scale(p(1, 2, f(1, 6)), f(13, 6)),
        _scale(p(1, 3, f(1, 2)), f(3, 5)),
    )
    a32 = _add(
        _scale(p(1, 1), -f(19, 180)),
        _scale(p(1, 1, f(1, 2)), -f(1, 6)),
        _scale(p(1, 1, f(1, 6)), -f(1, 6)),
        _scale(p(1, 2, f(1, 2)), -f(1, 6)),
        _scale(p(1, 2, f(1, 6)), f(1, 9)),
        _scale(p(1, 3, f(1, 2)), -f(1, 5)),
    )
    a30 = _minus(_scale(p(1, 1, f(1, 2)), f(1, 2)), a31, a32)
    a43 = _add(p(1, 2), p(1, 2, f(1, 2)), _scale(p(1, 3), -6), _scale(p(1, 3, f(1, 2)), -3))
    a41 = _add(
        _scale(p(1, 2), 3),
        _scale(p(1, 2, f(1, 2)), -f(9, 2)),
        _scale(p(1, 2, f(1, 6)), -f(5, 2)),
        _scale(a43, 6),
        a31,
    )
    a42 = _add(_scale(p(1, 3), 6), _scale(p(1, 3, f(1, 2)), 3), _scale(a43, -2), a32)
    a40 = _minus(p(1, 1), a41, a42, a43)
    bh0 = _add(p(1, 1), _scale(p(1, 2), -f(67, 9)), _scale(p(1, 3), f(52, 3)))
    bh1 = _add(_scale(p(1, 2), 8), _scale(p(1, 3), -24))
    bh2 = _add(_scale(p(1, 3), f(26, 3)), _scale(p(1, 2), -f(11, 9)))
    bh3 = _add(_scale(p(1, 2), f(7, 9)), _scale(p(1, 3), -f(10, 3)))
    bh4 = _add(_scale(p(1, 3), f(4, 3)), _scale(p(1, 2), -f(1, 9)))
    return EmbeddedTableau(
        "erk43zb",
        (f(0), f(1, 6), f(1, 2), f(1, 2), f(1)),
        ((), (a10,), (a20, a21), (a30, a31, a32), (a40, a41, a42, a43)),
        (bh0, bh1, bh2, bh3, bh4),
        (a40, a41, a42, a43, ()),
        4,
        3,
    )


ERK32ZB = _erk32zb()
ERK43ZB = _erk43zb()


def pair_step_at(y, stage_times, h, linear_ops, nonlinear_rhs, tableau):
    """Take one embedded ERK step and return (advancing, embedded) states.

    The Fourier layout follows the existing ETD stepper. Each distinct
    phi_k(argument * h * L) is built once per field and trial step.
    """
    from moljax.core.fft_integrators import _fft_pair

    phi_functions = {1: phi1, 2: phi2, 3: phi3}
    fields = {}
    for name, value in y.items():
        op = linear_ops.get(name)
        if op is None:
            forward = inverse = lambda array: array
            z = jnp.asarray(0, dtype=value.dtype)
        else:
            forward, inverse = _fft_pair(op, value)
            z = h * op.eigenvalues
        expressions = [expression for row in tableau.a for expression in row]
        expressions.extend(tableau.b)
        expressions.extend(tableau.bhat)
        keys = {(k, arg) for expression in expressions for _, k, arg in expression}
        phis = {(k, arg): phi_functions[k](float(arg) * z) for k, arg in keys}

        def weight(expression, phis=phis):
            return sum(float(coef) * phis[k, arg] for coef, k, arg in expression)

        fields[name] = (forward, inverse, z, forward(value), weight)

    n_hat = []
    for i, c in enumerate(tableau.c):
        if i == 0:
            stage = y
        else:
            stage = {}
            for name, (_, inverse, z, y_hat, weight) in fields.items():
                hat = jnp.exp(float(c) * z) * y_hat + h * sum(
                    weight(tableau.a[i][j]) * n_hat[j][name] for j in range(i)
                )
                stage[name] = inverse(hat)
        n = nonlinear_rhs(stage, stage_times[i])
        n_hat.append({name: fields[name][0](n[name]) for name in fields})

    results = []
    for weights in (tableau.b, tableau.bhat):
        result = {}
        for name, (_, inverse, z, y_hat, weight) in fields.items():
            hat = jnp.exp(z) * y_hat + h * sum(
                weight(weights[i]) * n_hat[i][name] for i in range(len(weights))
            )
            result[name] = inverse(hat)
        results.append(result)
    return results[0], results[1]
