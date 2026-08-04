# Blackbox Optimizer

Derivative-free, box-constrained black-box minimization for PHP. Given any `array<float> -> float`
objective function and per-dimension bounds, finds the vector that minimizes it — no gradient, no
assumption about what the function computes, just repeated evaluation. Ships three algorithms behind one
standardized interface: **CMA-ES** (adapts a full covariance matrix as it searches — generally the
strongest choice, at some extra complexity), a classic **Rechenberg/Schwefel Evolution Strategy** (isotropic
mutation with Rechenberg's own 1/5 success rule for step-size control — the historical predecessor CMA-ES
grew out of, no covariance matrix), and **Differential Evolution** (mutation/crossover/selection only —
simplest, "the thing to beat").

Framework-agnostic and dependency-free: `require php: >=8.3` only. Originally built inside
[spryker-community/search-ranking-optimizer](https://github.com/andrebarthelmeshellmuth/spryker-search-ranking-optimizer)
to search a Spryker shop's ranking-formula weights against a real relevance-judgment set, then extracted
here once it became clear nothing about the optimization core actually depended on that — see
[Relationship to search-ranking-optimizer](#relationship-to-search-ranking-optimizer).

## Contents

- [Quick example](#quick-example)
- [The API](#the-api)
  - [Parameter and ProblemInterface](#parameter-and-probleminterface)
  - [OptimizerAlgorithmInterface](#optimizeralgorithminterface)
  - [CallableProblem](#callableproblem)
- [Choosing an algorithm](#choosing-an-algorithm)
- [Early termination](#early-termination)
- [Expressing constraints beyond a box](#expressing-constraints-beyond-a-box)
- [Relationship to search-ranking-optimizer](#relationship-to-search-ranking-optimizer)
- [Installation](#installation)
- [Limitations](#limitations)
- [Testing and CI](#testing-and-ci)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Quick example

```php
use BlackboxOptimizer\Algorithm\CmaEsAlgorithm;
use BlackboxOptimizer\Problem\CallableProblem;

// Minimize f(x, y) = (x - 3)^2 + (y + 1)^2 -- known minimum: 0 at (3, -1).
$objectiveFunction = static fn (array $vector): float => ($vector[0] - 3) ** 2 + ($vector[1] + 1) ** 2;

$problem = new CallableProblem($objectiveFunction, lowerBounds: [-10.0, -10.0], upperBounds: [10.0, 10.0]);

$algorithm = new CmaEsAlgorithm();
$algorithm->setStepWidth(1.0)->setPopulationSize(12)->setMaxIterations(100);

$result = $algorithm->optimize($problem);

$result->getBestVector();   // ~[3.0, -1.0]
$result->getBestValue();    // ~0.0
$result->getEvaluationCount();
$result->getBestValueHistory(); // one entry per generation, oldest first
```

## The API

### Parameter and ProblemInterface

A `Parameter` is one dimension of the search space: a name (for logging/debugging only, never used by an
algorithm to make a decision), a lower/upper bound, and a type (`Continuous` today; `Integer` is declared
but not yet enforced by any shipped algorithm). A `ProblemInterface` bundles the parameter list together
with a way to score any point in it:

```php
interface ProblemInterface
{
    /** @return array<int, Parameter> */
    public function getParameters(): array;

    /** array<int, float> $vector -> float, LOWER is better (minimization). */
    public function evaluate(array $vector): float;
}
```

A Problem with 4 dimensions is 4 `Parameter` objects, not 4 entries in 3 parallel arrays a caller has to
keep in sync by position.

### OptimizerAlgorithmInterface

```php
interface OptimizerAlgorithmInterface
{
    public function getName(): string;
    public function getDescription(): string;
    public function optimize(ProblemInterface $problem): OptimizationResult;
    public function setStepWidth(float $stepWidth): static;
    public function setPopulationSize(int $populationSize): static;
    public function setMaxIterations(int $maxIterations): static;
}
```

**`getName()`** / **`getDescription()`** — a short label ("CMA-ES") and a factual, one- or two-sentence
description of what the algorithm mechanically does, e.g. "adapts a full covariance matrix as it
searches." Deliberately not opinionated about how the algorithms compare (no "generally the stronger
choice") — that stays in [Choosing an algorithm](#choosing-an-algorithm) below and is scope for a consumer
to add on top, not for this package. Meant for a consumer building a UI on top of this package (e.g. an
algorithm picker) to read instead of hand-copying this README's prose.

The three setters are the knobs every population-based algorithm here already has *some* version of, given
one shared name instead of a bespoke method per algorithm:

- **`setStepWidth()`** — CMA-ES's initial global step size (`σ0`); Differential Evolution's mutation factor
  (`F`). Both mean "how big are the first exploratory steps."
- **`setPopulationSize()`** — CMA-ES's `λ`; DE's population size.
- **`setMaxIterations()`** — the upper bound every algorithm here runs generations up to. All three may
  stop before reaching it once their own convergence criteria fire (see
  [Early termination](#early-termination)), or run the full count if those never trigger.

All three are optional — skipping every setter falls back to that algorithm's own sensible default.
Anything narrower stays a concrete, algorithm-specific method (`CmaEsAlgorithm::setInitialMean()`,
`DifferentialEvolutionAlgorithm::setCrossoverProbability()`) rather than being crammed into the shared
interface, where it would either lose meaning (a bag of mixed-purpose floats) or force every implementer
to support a parameter only some other algorithm needs.

### CallableProblem

The common case — a benchmark function, or any objective with no named parameters or cross-dimension
constraints of its own to express — doesn't need a dedicated class implementing `ProblemInterface` just
to wrap one closure and two bound arrays:

```php
$problem = new CallableProblem($objectiveFunction, $lowerBounds, $upperBounds);
```

A consumer with real per-parameter names, an `Integer` dimension, or its own reparametrization (see
[Expressing constraints beyond a box](#expressing-constraints-beyond-a-box)) should implement
`ProblemInterface` directly instead — `CallableProblem` is the convenient default, not the only option.

## Choosing an algorithm

- **`CmaEsAlgorithm`** — a faithful port of the standard (μ/μ_w, λ)-CMA-ES algorithm, adapting a full
  covariance matrix from generation to generation so it learns the search space's actual shape (correlated
  dimensions, differing sensitivities) rather than searching each one independently. Generally the
  stronger choice; some extra internal complexity (an eigendecomposition every generation) as the cost.
- **`RechenbergSchwefelEsAlgorithm`** — a (μ+λ)-ES: Rechenberg's isotropic Gaussian mutation plus-selection
  scheme, generalized to multiple parents/offspring the way Schwefel did, with a single scalar step size
  adapted by Rechenberg's own "1/5 success rule" instead of CMA-ES's learned covariance. The historical
  predecessor CMA-ES itself grew out of — meaningfully simpler (no covariance matrix, no cumulation paths),
  at the cost of not learning correlations between dimensions the way CMA-ES does.
- **`DifferentialEvolutionAlgorithm`** — DE/rand/1/bin: mutation + binomial crossover + greedy selection,
  nothing more. Deliberately the simplest population-based optimizer here — included as a baseline "the
  thing to beat" rather than because it's expected to win, and as proof `OptimizerAlgorithmInterface`
  genuinely generalizes beyond CMA-ES's own shape.

All three are validated in this package's own test suite against standard benchmark functions with known
optima (the sphere function, the Rosenbrock "banana" function) before ever being pointed at a real, much
more expensive objective — see [Testing and CI](#testing-and-ci).

## Early termination

All three algorithms now stop before `maxIterations` when they've genuinely converged, diverged, or
plateaued — checked every generation, always on, no toggle to disable it:

| Algorithm | What's checked |
|---|---|
| `CmaEsAlgorithm` | Hansen's standard TolX/TolXUp/ConditionCov/TolFun set (`Internal\TerminationCriteria`) — step size collapsed, step size blew up, covariance ill-conditioned, or best-of-generation fitness plateaued over a real trailing window. |
| `RechenbergSchwefelEsAlgorithm` | The same `TerminationCriteria`, reused as-is — it has a single scalar sigma and no covariance matrix, so TolX/TolXUp collapse to a direct check against sigma itself, and ConditionCov correctly never fires. |
| `DifferentialEvolutionAlgorithm` | Its own criteria — no sigma or covariance to check, so its TolX-equivalent is **population convergence** (every dimension's spread across the current population has collapsed relative to that dimension's own bound range) instead, plus the same fitness-plateau idea. |

**`trustTerminationCriteria()`** (algorithm-specific, call before `optimize()`, all three implement it): raises
the effective iteration ceiling from `DEFAULT_MAX_ITERATIONS`/whatever `setMaxIterations()` was given to a
much larger internal safety ceiling (10,000 generations), so a run is governed by its own convergence
criteria rather than an arbitrary generation-count guess cutting it off first. Any `setMaxIterations()` call
is ignored once this is on.

This is deliberately **not** a literal unbounded loop. These criteria are the standard heuristics from
Hansen's own tutorial and reference implementations (and this package's own equivalents for DE and the ES)
— not a formal proof of termination for an arbitrary black-box objective. A pathological function could in
principle satisfy none of them for a very long time. The safety ceiling stays in place as the last-resort
circuit breaker even in this mode.

Also deliberately not restart machinery (IPOP/BIPOP-CMA-ES) on top of any of this: a restart resets and
re-spends a caller's own evaluation-count budget, a real design tradeoff rather than a strict improvement,
so that stays a separate, still-open decision.

## Expressing constraints beyond a box

`Parameter` only expresses an independent bound per dimension — it has no way to say "these 3 parameters
must sum to 1," or any other constraint spanning several dimensions at once. That's not a gap to patch with
a richer `ParameterType`; a linear-equality (or any multi-dimensional) constraint is a different *kind* of
thing than a per-dimension bound, and teaching this package about one specific constraint shape would drag
a caller's own domain vocabulary into a package that's supposed to stay generic.

The fix is a reparametrization on the **caller's** side: express the constrained space (e.g. a simplex) as
an unconstrained one instead (e.g. via a softmax transform), implement `ProblemInterface::evaluate()` to
convert the unconstrained vector this package hands it back into the real, constrained values before
scoring them, and only ever declare `Parameter`s for the unconstrained dimensions. This package never needs
to know the constraint existed. `spryker-community/search-ranking-optimizer`'s own
`SimplexSoftmaxReparametrization` and `ParameterVectorMapper` are a worked example of exactly this pattern.

## Relationship to search-ranking-optimizer

This package's `CmaEsAlgorithm`/`DifferentialEvolutionAlgorithm` (plus their `SymmetricEigenDecomposition`/
`VectorMath` internals) originally lived inside `spryker-community/search-ranking-optimizer`'s own
`Shared\SearchRankingOptimizer\Optimization` namespace, built and validated there first. Nothing in that
original code ever referenced Spryker, search ranking, or rank_eval — the whole namespace was already
written as a generic `(objectiveFunction, bounds) -> result` black box specifically so it could be
benchmark-tested in isolation, which made it a clean, low-risk extraction rather than a refactor: this
package **is** that code, given a standardized `Parameter`/`ProblemInterface` vocabulary instead of raw
positional bound arrays, and a project-agnostic name.

`search-ranking-optimizer` depends on this package the normal way (a real Composer `require`) and supplies
its own `SimplexSoftmaxReparametrization`/`ParameterVectorMapper` as the domain-specific glue described
above — the actual weight-tuning logic, rank_eval objective, and Spryker integration all stay there; only
the generic optimization core moved.

`RechenbergSchwefelEsAlgorithm` is different — it was written directly in this package, not extracted from
`search-ranking-optimizer`. It implements `OptimizerAlgorithmInterface` the same way the other two do, so
any consumer (including `search-ranking-optimizer`) can pick it up as a third selectable algorithm without
this package's own API changing.

## Installation

```bash
composer require andrebarthelmeshellmuth/blackbox-optimizer
```

No further setup — no config, no service registration, no framework of any kind to wire into. Instantiate
an algorithm, build a `Problem`, call `optimize()`.

## Limitations

- **Global bounds only, no cross-dimension constraints** — see
  [Expressing constraints beyond a box](#expressing-constraints-beyond-a-box) above; anything beyond an
  independent box bound per dimension needs a reparametrization on the caller's side.
- **None of the three support automatic restarts** (e.g. CMA-ES's own IPOP-CMA-ES variant) on top of their
  own early termination — deliberately, in favor of simple, reviewable reference code; see
  [Early termination](#early-termination). Early termination itself is a set of standard heuristics, not a
  formal termination guarantee for an arbitrary objective — `trustTerminationCriteria()` still keeps a
  real, generous safety ceiling rather than looping forever.
- **`Integer` parameters are declared, not enforced.** `ParameterType::Integer` exists so a `Problem` can
  honestly describe an integer dimension, but no shipped algorithm currently rounds a candidate to the
  nearest integer for it — all three operate on plain continuous floats throughout.
- **CMA-ES's eigendecomposition runs every generation**, with no "only every few generations" optimization
  real production CMA-ES implementations use. Fine at the dimensionality this package targets (a handful to
  a few dozen parameters); a problem with hundreds of dimensions would pay a real, avoidable cost here.
- **No parallelism.** Every candidate in a generation is evaluated one at a time in the same process, even
  though they're independent of each other and could in principle run concurrently. A run's total
  wall-clock time is roughly `population size × iterations × one evaluate() call's own cost`.

## Testing and CI

```bash
composer install
composer test     # PHPUnit
composer phpcs     # coding standard
composer phpmd     # complexity / size limits
composer phpstan    # static analysis, level 8
```

All three algorithms are validated against the sphere function (single global minimum at the origin — a
basic convexity sanity check) and the 2D Rosenbrock "banana" function (a narrow, curved, non-convex valley
— a meaningfully stronger check, and specifically the kind of shape CMA-ES's covariance adaptation exists
to navigate better than DE or the ES can, so its own test tolerances are looser). Each test asserts
convergence close to the known optimum, a monotonically non-worsening best-value history, and the exact
expected evaluation count for a given population/iteration budget.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

`CmaEsAlgorithm` is a PHP port of **Nikolaus Hansen**'s own simplified reference implementation of CMA-ES
("purecma") — the algorithm and its careful, from-scratch-avoiding implementation approach are entirely his
life's work; any bugs introduced in adapting it to PHP are mine alone.

`RechenbergSchwefelEsAlgorithm` implements the Evolution Strategy pair **Ingo Rechenberg** and
**Hans-Paul Schwefel** originated in the 1960s/70s — Rechenberg's (1+1)-ES and its 1/5 success rule for
step-size adaptation, generalized to multiple parents/offspring the way Schwefel did. CMA-ES is itself a
much later descendant of this same lineage.
