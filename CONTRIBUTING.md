# Contributing to moljax

Contributions are welcome. This document covers licensing, the branch and
review model, and what CI expects.

## Licensing of contributions

moljax is MIT licensed. **By opening a pull request you confirm that you are
able to license your contribution under the MIT license**, either because you
own the copyright or because you have permission from whoever does.

This matters more than it usually would. Contributions to this project often
come out of funded research, and employers, universities and grant agencies
frequently retain rights to work produced under their funding. If your
contribution was written as part of funded or employed research, please check
with the body that funded it *before* opening the pull request. Discovering an
ownership problem after code has shipped in a tagged release is far worse for
you than a short delay now.

If your parent project is under a copyleft license such as GPL, note that the
direction matters: MIT code can be used inside a GPL project, but GPL-derived
code cannot be relicensed into this MIT one without permission from the rights
holder.

If you are unsure, say so in the pull request and we will work it out before
merging.

## Branches and review

- `main` is protected. It requires a pull request, a review, and passing CI.
  Force pushes and branch deletion are disabled.
- Release tags (`v*`) are protected and cannot be moved or deleted. `v1.0.0`
  in particular is the archival commit cited by the published paper and must
  never change.
- Long-lived topic branches are fine, and preferred over a pull request per
  increment. Stack your work on a branch and open one pull request when the
  increment is coherent.

## What CI runs

Every pull request runs the test suite on Python 3.10 and 3.12, CPU-only,
via `JAX_PLATFORMS=cpu`. This is the required check.

```bash
pip install -e ".[dev,viz]"
pytest
```

Linting runs as an **advisory** job. The repository carries a backlog of style
findings that predates CI, so a failing lint job does not block a merge today.
Please do not add new findings in code you touch:

```bash
ruff check .
```

`E402` is disabled project-wide: JAX requires
`jax.config.update("jax_enable_x64", True)` to execute before `jax.numpy` is
imported, so module-level imports legitimately follow executable statements.

## Tests

- New behavior needs a test. Numerical claims need a test against an
  independent reference (an analytic solution, a dense computation, or a
  well-established library), not just a regression snapshot.
- Mark anything that takes more than about a minute with `@pytest.mark.slow`.
  Slow tests are deselected by default so the standard run stays fast enough
  that people actually run it. Use `pytest -m slow` for those, or `pytest -m ""`
  for everything.
- Benchmarks live in `benchmarks/` and are not part of the test suite.

## Benchmarks and comparisons

If you add a benchmark that compares moljax against another library, the
filename, the docstring and any committed results must make the comparison
unambiguous, including what is held fixed and what is not. A comparison
between methods of different order, or between fixed-step and adaptive
stepping, is easy to misread out of context, and committed result files
outlive the discussion that produced them.

Keep committed result data proportionate. Large generated artifacts are better
regenerated from a script than stored in git history.

## Reporting problems

Open an issue with a minimal reproduction, the output of

```bash
python -c "import jax, moljax; print(jax.__version__, moljax.__version__, jax.devices())"
```

and what you expected instead.
