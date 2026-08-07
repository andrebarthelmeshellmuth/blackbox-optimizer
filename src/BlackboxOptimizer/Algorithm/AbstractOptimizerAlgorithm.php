<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm;

use BlackboxOptimizer\Problem\ProblemInterface;
use InvalidArgumentException;
use Random\Randomizer;

/**
 * Common, algorithm-agnostic bookkeeping every population-based black-box optimizer needs: the shared
 * setStepWidth()/setPopulationSize()/setMaxIterations() knobs from {@see OptimizerAlgorithmInterface}
 * (each nullable here so a concrete algorithm can fall back to its OWN default -- including a non-scalar
 * default like CMA-ES's dimension-dependent population-size formula -- when a caller never calls the
 * setter at all), bounds extraction/clamping, a random-vector-within-bounds generator (for an initial
 * population/mean), evaluation-count + best-so-far tracking so every concrete algorithm reports an
 * {@see OptimizationResult} the same way instead of reimplementing this per algorithm, and the two
 * cross-algorithm opt-ins ({@see trustTerminationCriteria()}, {@see setWarmStart()}) whose STORAGE and
 * validation is identical everywhere even though what a concrete algorithm DOES with them differs -- see
 * {@see CmaEsAlgorithm::resolveInitialMean()} vs. {@see seedInitialPopulation()} for the two shapes.
 */
abstract class AbstractOptimizerAlgorithm implements OptimizerAlgorithmInterface
{
    protected Randomizer $randomizer;

    protected ?float $stepWidth = null;

    protected ?int $populationSize = null;

    protected ?int $maxIterations = null;

    protected bool $trustTerminationCriteria = false;

    /**
     * @var array<int, float>|null
     */
    protected ?array $warmStartVector = null;

    protected float $warmStartFraction = 0.0;

    protected int $evaluationCount = 0;

    protected ?float $bestValueSoFar = null;

    /**
     * @var array<int, float>
     */
    protected array $bestVectorSoFar = [];

    /**
     * @var array<int, float>
     */
    protected array $bestValueHistory = [];

    /**
     * @param \Random\Randomizer|null $randomizer Injectable for deterministic tests; defaults to a real
     *   random engine in production use.
     */
    public function __construct(?Randomizer $randomizer = null)
    {
        $this->randomizer = $randomizer ?? new Randomizer();
    }

    /**
     * @param float $stepWidth
     *
     * @throws \InvalidArgumentException
     *
     * @return static
     */
    public function setStepWidth(float $stepWidth): static
    {
        if ($stepWidth <= 0.0) {
            throw new InvalidArgumentException('stepWidth must be greater than 0.');
        }

        $this->stepWidth = $stepWidth;

        return $this;
    }

    /**
     * @param int $populationSize
     *
     * @throws \InvalidArgumentException
     *
     * @return static
     */
    public function setPopulationSize(int $populationSize): static
    {
        if ($populationSize < 4) {
            throw new InvalidArgumentException('populationSize must be at least 4.');
        }

        $this->populationSize = $populationSize;

        return $this;
    }

    /**
     * @param int $maxIterations
     *
     * @throws \InvalidArgumentException
     *
     * @return static
     */
    public function setMaxIterations(int $maxIterations): static
    {
        if ($maxIterations < 1) {
            throw new InvalidArgumentException('maxIterations must be at least 1.');
        }

        $this->maxIterations = $maxIterations;

        return $this;
    }

    /**
     * {@inheritDoc}
     *
     * @return static
     */
    public function trustTerminationCriteria(): static
    {
        $this->trustTerminationCriteria = true;

        return $this;
    }

    /**
     * {@inheritDoc}
     *
     * @param array<int, float> $vector
     * @param float $fraction
     *
     * @throws \InvalidArgumentException
     *
     * @return static
     */
    public function setWarmStart(array $vector, float $fraction): static
    {
        if ($fraction < 0.0 || $fraction > 1.0) {
            throw new InvalidArgumentException('fraction must be between 0.0 and 1.0.');
        }

        $this->warmStartVector = $vector;
        $this->warmStartFraction = $fraction;

        return $this;
    }

    /**
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     *
     * @throws \InvalidArgumentException
     *
     * @return array{0: array<int, float>, 1: array<int, float>} [lowerBounds, upperBounds], one entry per
     *   parameter, in declared order.
     */
    protected function extractBounds(ProblemInterface $problem): array
    {
        $parameters = $problem->getParameters();

        if ($parameters === []) {
            throw new InvalidArgumentException('A problem must declare at least one parameter.');
        }

        $lowerBounds = [];
        $upperBounds = [];

        foreach ($parameters as $index => $parameter) {
            $lowerBounds[$index] = $parameter->getLowerBound();
            $upperBounds[$index] = $parameter->getUpperBound();
        }

        return [$lowerBounds, $upperBounds];
    }

    /**
     * @param array<int, float> $vector
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @return array<int, float>
     */
    protected function clamp(array $vector, array $lowerBounds, array $upperBounds): array
    {
        $clamped = [];

        foreach ($vector as $index => $value) {
            $clamped[$index] = min(max($value, $lowerBounds[$index]), $upperBounds[$index]);
        }

        return $clamped;
    }

    /**
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @throws \InvalidArgumentException
     *
     * @return array<int, float>
     */
    protected function randomVectorWithinBounds(array $lowerBounds, array $upperBounds): array
    {
        $vector = [];

        foreach ($lowerBounds as $index => $lowerBound) {
            $upperBound = $upperBounds[$index];

            if (is_infinite($lowerBound) || is_infinite($upperBound)) {
                throw new InvalidArgumentException(
                    'Every dimension must be finitely bounded to sample a random vector within bounds -- an '
                    . 'infinite bound has no finite range to sample from. An algorithm that needs to start from '
                    . 'an unbounded dimension must be given an explicit starting point instead (e.g. '
                    . 'CmaEsAlgorithm::setInitialMean()).',
                );
            }

            $vector[$index] = $lowerBound + $this->randomizer->getFloat(0.0, 1.0) * ($upperBound - $lowerBound);
        }

        return $vector;
    }

    /**
     * Shared by every population algorithm's initial-population construction ({@see DifferentialEvolutionAlgorithm},
     * {@see RechenbergSchwefelEsAlgorithm}): the first {@see round}($count * $warmStartFraction) members are
     * seeded near {@see setWarmStart()}'s vector (jittered, never the literal identical point -- see
     * {@see jitterVector()} for why more than one identical member would be actively harmful, not just
     * redundant), the rest are this package's original {@see randomVectorWithinBounds()} draw. Falls back to
     * fully random when {@see setWarmStart()} was never called, regardless of $jitterScale.
     *
     * @param int $count
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     * @param float $jitterScale This algorithm's own resolved step width (sigma/F) -- the same unit scale
     *   it already uses for a mutation step, so a seeded member's jitter is comparable in size to a normal
     *   generation's own movement rather than an arbitrarily different scale.
     *
     * @return array<int, array<int, float>>
     */
    protected function seedInitialPopulation(int $count, array $lowerBounds, array $upperBounds, float $jitterScale): array
    {
        $warmStartCount = $this->warmStartVector !== null
            ? (int)round($count * $this->warmStartFraction)
            : 0;

        $population = [];

        for ($i = 0; $i < $count; $i++) {
            $population[$i] = $i < $warmStartCount
                ? $this->jitterVector($this->warmStartVector, $jitterScale, $lowerBounds, $upperBounds)
                : $this->randomVectorWithinBounds($lowerBounds, $upperBounds);
        }

        return $population;
    }

    /**
     * $vector plus independent N(0, $jitterScale) noise per dimension, clamped back into bounds. Never
     * returns $vector unperturbed -- a population algorithm seeding more than one member at the exact same
     * point would degenerate its own mechanism (e.g. Differential Evolution's a + F*(b-c) collapses to a
     * zero step whenever two of its three picks are identical), so every seeded member needs to be a
     * distinct point near $vector, not $vector itself.
     *
     * @param array<int, float> $vector
     * @param float $jitterScale
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @return array<int, float>
     */
    protected function jitterVector(array $vector, float $jitterScale, array $lowerBounds, array $upperBounds): array
    {
        $jittered = [];

        foreach ($vector as $index => $value) {
            $jittered[$index] = $value + $this->standardNormal() * $jitterScale;
        }

        return $this->clamp($jittered, $lowerBounds, $upperBounds);
    }

    /**
     * Every candidate evaluation MUST go through here, never call $problem->evaluate() directly — this is
     * the single place evaluation count and the best-found-so-far vector/value are tracked.
     *
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     * @param array<int, float> $vector
     *
     * @return float
     */
    protected function evaluate(ProblemInterface $problem, array $vector): float
    {
        $value = $problem->evaluate($vector);
        $this->evaluationCount++;

        if ($this->bestValueSoFar === null || $value < $this->bestValueSoFar) {
            $this->bestValueSoFar = $value;
            $this->bestVectorSoFar = $vector;
        }

        return $value;
    }

    /**
     * Call once per generation/iteration, after evaluating that generation's candidates, to append the
     * running best to the reported history.
     *
     * @return void
     */
    protected function recordGenerationHistory(): void
    {
        $this->bestValueHistory[] = $this->bestValueSoFar ?? INF;
    }

    /**
     * Resets all tracking state — every optimize() call must start with this, so a reused algorithm
     * instance never leaks a previous run's best-found vector into a new one.
     *
     * @return void
     */
    protected function resetTracking(): void
    {
        $this->evaluationCount = 0;
        $this->bestValueSoFar = null;
        $this->bestVectorSoFar = [];
        $this->bestValueHistory = [];
    }

    /**
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    protected function buildResult(): OptimizationResult
    {
        return new OptimizationResult(
            $this->bestVectorSoFar,
            $this->bestValueSoFar ?? INF,
            $this->evaluationCount,
            $this->bestValueHistory,
        );
    }

    /**
     * Box-Muller transform — a standard, simple way to draw from N(0,1) without depending on any PHP
     * extension beyond what {@see \Random\Randomizer} already provides. Shared by every algorithm here
     * that needs isotropic Gaussian mutation ({@see \BlackboxOptimizer\Algorithm\CmaEsAlgorithm},
     * {@see \BlackboxOptimizer\Algorithm\RechenbergSchwefelEsAlgorithm}) — promoted up here once a second
     * algorithm needed it, the same justification that already moved matrix/vector arithmetic out into
     * {@see \BlackboxOptimizer\Algorithm\Internal\VectorMath}.
     *
     * @return float
     */
    protected function standardNormal(): float
    {
        $u1 = max($this->randomizer->getFloat(0.0, 1.0), PHP_FLOAT_EPSILON);
        $u2 = $this->randomizer->getFloat(0.0, 1.0);

        return sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
    }
}
