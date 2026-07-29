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
 * population/mean), and evaluation-count + best-so-far tracking so every concrete algorithm reports an
 * {@see OptimizationResult} the same way instead of reimplementing this per algorithm.
 */
abstract class AbstractOptimizerAlgorithm implements OptimizerAlgorithmInterface
{
    protected Randomizer $randomizer;

    protected ?float $stepWidth = null;

    protected ?int $populationSize = null;

    protected ?int $maxIterations = null;

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
     * @return array<int, float>
     */
    protected function randomVectorWithinBounds(array $lowerBounds, array $upperBounds): array
    {
        $vector = [];

        foreach ($lowerBounds as $index => $lowerBound) {
            $vector[$index] = $lowerBound + $this->randomizer->getFloat(0.0, 1.0) * ($upperBounds[$index] - $lowerBound);
        }

        return $vector;
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
}
