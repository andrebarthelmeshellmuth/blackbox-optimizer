<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm;

use BlackboxOptimizer\Algorithm\Internal\TerminationCriteria;
use BlackboxOptimizer\Problem\ProblemInterface;
use InvalidArgumentException;
use Random\Randomizer;

/**
 * A (μ+λ)-Evolution Strategy: Rechenberg's isotropic-Gaussian-mutation-plus-selection scheme, generalized
 * from his original (1+1)-ES to multiple parents/offspring the way Schwefel did, with step size adapted
 * by Rechenberg's own "1/5 success rule" rather than CMA-ES's covariance/cumulation machinery. The
 * historical predecessor {@see CmaEsAlgorithm} itself grew out of — no covariance matrix, no cumulation
 * paths, a single scalar step size shared across every dimension. Deliberately the third distinct
 * algorithmic shape in this namespace (isotropic mutation + success-rate step control, vs. CMA-ES's learned
 * covariance and {@see DifferentialEvolutionAlgorithm}'s vector-differential mutation), not a simplified
 * CMA-ES.
 *
 * Reuses {@see TerminationCriteria} exactly as-is for early termination (see {@see trustTerminationCriteria()}),
 * passing a single-element `[1.0]` in place of CmaEsAlgorithm's per-dimension sqrt-eigenvalues array. This
 * is not a hack: it honestly encodes "no covariance, every direction scaled identically by sigma alone" --
 * TolX/TolXUp collapse to a direct comparison against sigma itself (correct, since sigma IS the only step
 * length here), and ConditionCov's ratio is always exactly 1.0, so it correctly never fires (there is no
 * covariance matrix to become ill-conditioned).
 *
 * Deliberately omits what later ES research added on top of this classic pair's own work: no recombination
 * (an offspring is a mutated copy of exactly one parent, never a blend of several — Schwefel's own later
 * (μ/ρ,λ)-ES adds this), no per-dimension or per-individual self-adaptive step sizes (Schwefel's other,
 * separate self-adaptation proposal), no comma-selection option (plus-selection only — parents compete with
 * their own offspring, so the best-found value can only stay the same or improve). Matches this namespace's
 * "reviewable reference code over sophistication" bias — see {@see DifferentialEvolutionAlgorithm}'s own
 * docblock for the same stance on CMA-ES's sophistication.
 *
 * {@see OptimizerAlgorithmInterface::setStepWidth()} maps to Rechenberg's σ (the initial, then
 * self-adapting, isotropic mutation standard deviation) — the same role it plays for CmaEsAlgorithm's σ0.
 * {@see OptimizerAlgorithmInterface::setPopulationSize()} maps to λ, the offspring count per generation —
 * μ (the parent/survivor count) is derived from λ via {@see setParentCount()}'s own default when not set
 * explicitly.
 */
class RechenbergSchwefelEsAlgorithm extends AbstractOptimizerAlgorithm
{
    /**
     * @var int
     */
    protected const DEFAULT_POPULATION_SIZE = 20;

    /**
     * @var float
     */
    protected const DEFAULT_STEP_WIDTH = 0.3;

    /**
     * @var int
     */
    protected const DEFAULT_MAX_ITERATIONS = 200;

    /**
     * Schwefel's own classic guideline for the (μ,λ)/(μ+λ)-ES selection pressure -- μ/λ ≈ 1/7 -- used to
     * derive a default parent count from the offspring count (λ) when {@see setParentCount()} was never
     * called.
     *
     * @var int
     */
    protected const DEFAULT_PARENT_TO_OFFSPRING_RATIO_DIVISOR = 7;

    /**
     * Rechenberg's own target: roughly one in five mutations should succeed. Above it, steps are too
     * timid and should grow; below it, steps are too reckless and should shrink.
     *
     * @var float
     */
    protected const TARGET_SUCCESS_RATE = 0.2;

    /**
     * Schwefel's own recommended adaptation constant for the 1/5 rule -- shrink by this factor on too few
     * successes, grow by its reciprocal on too many. See Beyer & Schwefel, "Evolution Strategies -- A
     * Comprehensive Introduction" (2002), for the same constant.
     *
     * @var float
     */
    protected const STEP_SIZE_ADAPTATION_FACTOR = 0.85;

    /**
     * A practical floor, not part of the classic algorithm -- prevents sigma from ever reaching exactly
     * 0.0 (which would freeze every future mutation to its parent's exact value) after enough consecutive
     * unsuccessful generations.
     *
     * @var float
     */
    protected const MIN_STEP_WIDTH = 1.0E-10;

    /**
     * See {@see CmaEsAlgorithm::SAFETY_ITERATION_CEILING} -- same role, same value, same reasoning.
     *
     * @var int
     */
    protected const SAFETY_ITERATION_CEILING = 10000;

    /**
     * @var int|null
     */
    protected ?int $parentCount = null;

    /**
     * @var bool
     */
    protected bool $trustTerminationCriteria = false;

    /**
     * @var \BlackboxOptimizer\Algorithm\Internal\TerminationCriteria
     */
    protected TerminationCriteria $terminationCriteria;

    /**
     * @param \Random\Randomizer|null $randomizer
     * @param \BlackboxOptimizer\Algorithm\Internal\TerminationCriteria|null $terminationCriteria
     */
    public function __construct(?Randomizer $randomizer = null, ?TerminationCriteria $terminationCriteria = null)
    {
        parent::__construct($randomizer);
        $this->terminationCriteria = $terminationCriteria ?? new TerminationCriteria();
    }

    /**
     * @return string
     */
    public function getName(): string
    {
        return 'Rechenberg/Schwefel ES';
    }

    /**
     * @return string
     */
    public function getDescription(): string
    {
        return 'A (μ+λ)-ES: isotropic Gaussian mutation plus-selection, with a single scalar step size '
            . 'adapted by Rechenberg\'s "1/5 success rule" instead of a learned covariance matrix. Does '
            . 'not learn correlations between dimensions.';
    }

    /**
     * Algorithm-specific setup, deliberately NOT part of {@see OptimizerAlgorithmInterface} — call before
     * optimize() to override the default; skipping this call entirely is fine too.
     *
     * @param int $parentCount "μ" — the number of parents kept as survivors each generation. Must be at
     *   least 1. Validated against λ (the resolved population size) at optimize() time, since λ itself
     *   may still be at its own unresolved default when this setter is called.
     *
     * @throws \InvalidArgumentException
     *
     * @return $this
     */
    public function setParentCount(int $parentCount): static
    {
        if ($parentCount < 1) {
            throw new InvalidArgumentException('parentCount must be at least 1.');
        }

        $this->parentCount = $parentCount;

        return $this;
    }

    /**
     * Algorithm-specific setup, deliberately NOT part of {@see OptimizerAlgorithmInterface} -- call before
     * optimize() to opt in. Same role as {@see CmaEsAlgorithm::trustTerminationCriteria()}: switches the
     * effective iteration ceiling from {@see DEFAULT_MAX_ITERATIONS}/whatever {@see setMaxIterations()}
     * was given to {@see SAFETY_ITERATION_CEILING}, so a run is governed by {@see TerminationCriteria}
     * deciding sigma has converged, diverged, or fitness has plateaued -- not by an arbitrary generation
     * count. Any {@see setMaxIterations()} call is ignored once this is on. Not a literal unbounded loop
     * for the same reason CmaEsAlgorithm's own version isn't -- see that method's docblock.
     *
     * @return $this
     */
    public function trustTerminationCriteria(): static
    {
        $this->trustTerminationCriteria = true;

        return $this;
    }

    /**
     * {@inheritDoc}
     *
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     *
     * @throws \InvalidArgumentException
     *
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    public function optimize(ProblemInterface $problem): OptimizationResult
    {
        $this->resetTracking();
        [$lowerBounds, $upperBounds] = $this->extractBounds($problem);
        $dimensionCount = count($lowerBounds);

        $offspringCount = $this->populationSize ?? static::DEFAULT_POPULATION_SIZE;
        $parentCount = $this->resolveParentCount($offspringCount);
        $sigma = $this->stepWidth ?? static::DEFAULT_STEP_WIDTH;
        $initialSigma = $sigma;
        $maxIterations = $this->trustTerminationCriteria
            ? static::SAFETY_ITERATION_CEILING
            : ($this->maxIterations ?? static::DEFAULT_MAX_ITERATIONS);
        $fitnessHistoryLength = $this->terminationCriteria->resolveFitnessHistoryLength($dimensionCount, $offspringCount);
        $recentGenerationBestValues = [];

        $parents = [];
        $parentValues = [];

        for ($i = 0; $i < $parentCount; $i++) {
            $vector = $this->randomVectorWithinBounds($lowerBounds, $upperBounds);
            $parents[$i] = $vector;
            $parentValues[$i] = $this->evaluate($problem, $vector);
        }

        $this->recordGenerationHistory();

        for ($generation = 0; $generation < $maxIterations; $generation++) {
            [$parents, $parentValues, $sigma] = $this->runOneGeneration(
                $problem,
                $parents,
                $parentValues,
                $offspringCount,
                $sigma,
                $lowerBounds,
                $upperBounds,
            );

            $this->recordGenerationHistory();

            $recentGenerationBestValues[] = $parentValues[0];

            if (count($recentGenerationBestValues) > $fitnessHistoryLength) {
                array_shift($recentGenerationBestValues);
            }

            // A single-element [1.0] in place of CmaEsAlgorithm's per-dimension sqrt-eigenvalues array --
            // see this class's own docblock for why that's a correct, not approximate, degenerate case.
            if ($this->terminationCriteria->shouldTerminateEarly($sigma, $initialSigma, [1.0], $recentGenerationBestValues, $fitnessHistoryLength)) {
                break;
            }
        }

        return $this->buildResult();
    }

    /**
     * @param int $offspringCount
     *
     * @throws \InvalidArgumentException
     *
     * @return int
     */
    protected function resolveParentCount(int $offspringCount): int
    {
        if ($this->parentCount === null) {
            return max(1, (int)round($offspringCount / static::DEFAULT_PARENT_TO_OFFSPRING_RATIO_DIVISOR));
        }

        if ($this->parentCount > $offspringCount) {
            throw new InvalidArgumentException(
                'parentCount (mu) must not exceed the offspring count (lambda, i.e. populationSize).',
            );
        }

        return $this->parentCount;
    }

    /**
     * One generation: mutate λ offspring from randomly chosen parents, adapt sigma from this generation's
     * own success rate (Rechenberg's 1/5 rule), then keep the best μ of the combined parent+offspring pool
     * as the next generation's parents (plus-selection).
     *
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     * @param array<int, array<int, float>> $parents
     * @param array<int, float> $parentValues
     * @param int $offspringCount
     * @param float $sigma
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @return array{0: array<int, array<int, float>>, 1: array<int, float>, 2: float}
     */
    protected function runOneGeneration(
        ProblemInterface $problem,
        array $parents,
        array $parentValues,
        int $offspringCount,
        float $sigma,
        array $lowerBounds,
        array $upperBounds,
    ): array {
        $parentCount = count($parents);
        $dimensionCount = count($lowerBounds);
        $worstParentValue = $this->worstValue($parentValues);

        $offspring = [];
        $offspringValues = [];
        $successCount = 0;

        for ($i = 0; $i < $offspringCount; $i++) {
            $parentIndex = $this->randomizer->getInt(0, $parentCount - 1);
            $offspringVector = $this->mutate($parents[$parentIndex], $sigma, $dimensionCount, $lowerBounds, $upperBounds);
            $offspringValue = $this->evaluate($problem, $offspringVector);

            $offspring[$i] = $offspringVector;
            $offspringValues[$i] = $offspringValue;

            if ($offspringValue < $worstParentValue) {
                $successCount++;
            }
        }

        $sigma = $this->adaptStepSize($sigma, $successCount / $offspringCount);
        [$nextParents, $nextParentValues] = $this->selectSurvivors($parents, $parentValues, $offspring, $offspringValues, $parentCount);

        return [$nextParents, $nextParentValues, $sigma];
    }

    /**
     * Plain loop rather than max($parentValues) -- $parentValues is only ever empty if $parentCount could
     * be 0, which {@see setParentCount()} and {@see resolveParentCount()} both already forbid, but nothing
     * makes that invariant visible to static analysis across the call boundary.
     *
     * @param array<int, float> $values
     *
     * @return float
     */
    protected function worstValue(array $values): float
    {
        $worst = -INF;

        foreach ($values as $value) {
            if ($value > $worst) {
                $worst = $value;
            }
        }

        return $worst;
    }

    /**
     * @param array<int, float> $parent
     * @param float $sigma
     * @param int $dimensionCount
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @return array<int, float>
     */
    protected function mutate(array $parent, float $sigma, int $dimensionCount, array $lowerBounds, array $upperBounds): array
    {
        $offspring = [];

        for ($dimension = 0; $dimension < $dimensionCount; $dimension++) {
            $offspring[$dimension] = $parent[$dimension] + $sigma * $this->standardNormal();
        }

        return $this->clamp($offspring, $lowerBounds, $upperBounds);
    }

    /**
     * Rechenberg's 1/5 success rule, applied once per generation (a common simplification of Rechenberg's
     * own original interval -- re-evaluating success every n generations over ~10n trials -- that trades a
     * slightly noisier signal for a much simpler, more reviewable implementation).
     *
     * @param float $sigma
     * @param float $successRate
     *
     * @return float
     */
    protected function adaptStepSize(float $sigma, float $successRate): float
    {
        if ($successRate > static::TARGET_SUCCESS_RATE) {
            $sigma /= static::STEP_SIZE_ADAPTATION_FACTOR;
        } elseif ($successRate < static::TARGET_SUCCESS_RATE) {
            $sigma *= static::STEP_SIZE_ADAPTATION_FACTOR;
        }

        return max($sigma, static::MIN_STEP_WIDTH);
    }

    /**
     * Plus-selection: the μ+λ combined pool of parents and offspring is ranked, and the best μ survive
     * into the next generation regardless of whether they were a parent or an offspring -- the mechanism
     * that guarantees the best-found value never gets worse from one generation to the next.
     *
     * @param array<int, array<int, float>> $parents
     * @param array<int, float> $parentValues
     * @param array<int, array<int, float>> $offspring
     * @param array<int, float> $offspringValues
     * @param int $parentCount
     *
     * @return array{0: array<int, array<int, float>>, 1: array<int, float>}
     */
    protected function selectSurvivors(array $parents, array $parentValues, array $offspring, array $offspringValues, int $parentCount): array
    {
        $pool = array_merge($parents, $offspring);
        $poolValues = array_merge($parentValues, $offspringValues);

        asort($poolValues);
        $survivorKeys = array_slice(array_keys($poolValues), 0, $parentCount);

        $nextParents = [];
        $nextParentValues = [];

        foreach ($survivorKeys as $newIndex => $poolKey) {
            $nextParents[$newIndex] = $pool[$poolKey];
            $nextParentValues[$newIndex] = $poolValues[$poolKey];
        }

        return [$nextParents, $nextParentValues];
    }
}
