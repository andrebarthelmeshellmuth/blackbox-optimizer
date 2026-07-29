<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm;

use BlackboxOptimizer\Problem\ProblemInterface;
use InvalidArgumentException;

/**
 * Classic DE/rand/1/bin Differential Evolution — mutation (a + F * (b - c)) + binomial crossover +
 * greedy selection, nothing more. Deliberately the simplest population-based black-box optimizer in this
 * namespace: no covariance adaptation, no step-size control, just a handful of arithmetic operations per
 * candidate — see this package's README for why this exists alongside CmaEsAlgorithm rather than instead
 * of it (it's "the thing to beat," and it doubles as proof {@see OptimizerAlgorithmInterface} genuinely
 * generalizes beyond CMA-ES's own shape, not just a second implementation of the same idea).
 *
 * {@see OptimizerAlgorithmInterface::setStepWidth()} maps to "F" here (the mutation factor scaling the
 * differential b - c) — the closest DE equivalent to "how big are the first exploratory steps."
 */
class DifferentialEvolutionAlgorithm extends AbstractOptimizerAlgorithm
{
    /**
     * @var int
     */
    protected const DEFAULT_POPULATION_SIZE = 20;

    /**
     * @var float
     */
    protected const DEFAULT_STEP_WIDTH = 0.8;

    /**
     * @var float
     */
    protected const DEFAULT_CROSSOVER_PROBABILITY = 0.9;

    /**
     * @var int
     */
    protected const DEFAULT_MAX_ITERATIONS = 100;

    /**
     * @var float
     */
    protected float $crossoverProbability = self::DEFAULT_CROSSOVER_PROBABILITY;

    /**
     * Algorithm-specific setup, deliberately NOT part of {@see OptimizerAlgorithmInterface} — call before
     * optimize() to override the default; skipping this call entirely is fine too.
     *
     * @param float $crossoverProbability "CR" — per-dimension probability a trial vector's value comes
     *   from the mutant rather than the original target. Must be in [0; 1].
     *
     * @throws \InvalidArgumentException
     *
     * @return $this
     */
    public function setCrossoverProbability(float $crossoverProbability): static
    {
        if ($crossoverProbability < 0.0 || $crossoverProbability > 1.0) {
            throw new InvalidArgumentException('crossoverProbability must be between 0 and 1.');
        }

        $this->crossoverProbability = $crossoverProbability;

        return $this;
    }

    /**
     * {@inheritDoc}
     *
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     *
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    public function optimize(ProblemInterface $problem): OptimizationResult
    {
        $this->resetTracking();
        [$lowerBounds, $upperBounds] = $this->extractBounds($problem);

        $populationSize = $this->populationSize ?? static::DEFAULT_POPULATION_SIZE;
        $mutationFactor = $this->stepWidth ?? static::DEFAULT_STEP_WIDTH;
        $maxIterations = $this->maxIterations ?? static::DEFAULT_MAX_ITERATIONS;

        $population = [];
        $populationValues = [];

        for ($i = 0; $i < $populationSize; $i++) {
            $vector = $this->randomVectorWithinBounds($lowerBounds, $upperBounds);
            $population[$i] = $vector;
            $populationValues[$i] = $this->evaluate($problem, $vector);
        }

        $this->recordGenerationHistory();

        for ($generation = 0; $generation < $maxIterations; $generation++) {
            [$population, $populationValues] = $this->runOneGeneration(
                $problem,
                $population,
                $populationValues,
                $lowerBounds,
                $upperBounds,
                $mutationFactor,
            );

            $this->recordGenerationHistory();
        }

        return $this->buildResult();
    }

    /**
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     * @param array<int, array<int, float>> $population
     * @param array<int, float> $populationValues
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     * @param float $mutationFactor
     *
     * @return array{0: array<int, array<int, float>>, 1: array<int, float>}
     */
    protected function runOneGeneration(
        ProblemInterface $problem,
        array $population,
        array $populationValues,
        array $lowerBounds,
        array $upperBounds,
        float $mutationFactor,
    ): array {
        $dimensionCount = count($lowerBounds);
        $nextPopulation = $population;
        $nextPopulationValues = $populationValues;

        foreach ($population as $targetIndex => $targetVector) {
            [$vectorA, $vectorB, $vectorC] = $this->pickThreeDistinctOthers($population, $targetIndex);
            $forcedDimension = $this->randomizer->getInt(0, $dimensionCount - 1);

            $trialVector = [];

            for ($dimension = 0; $dimension < $dimensionCount; $dimension++) {
                $useMutant = $dimension === $forcedDimension || $this->randomizer->getFloat(0.0, 1.0) < $this->crossoverProbability;

                $trialVector[$dimension] = $useMutant
                    ? $vectorA[$dimension] + $mutationFactor * ($vectorB[$dimension] - $vectorC[$dimension])
                    : $targetVector[$dimension];
            }

            $trialVector = $this->clamp($trialVector, $lowerBounds, $upperBounds);
            $trialValue = $this->evaluate($problem, $trialVector);

            if ($trialValue > $populationValues[$targetIndex]) {
                continue;
            }

            $nextPopulation[$targetIndex] = $trialVector;
            $nextPopulationValues[$targetIndex] = $trialValue;
        }

        return [$nextPopulation, $nextPopulationValues];
    }

    /**
     * @param array<int, array<int, float>> $population
     * @param int $excludeIndex
     *
     * @return array{0: array<int, float>, 1: array<int, float>, 2: array<int, float>}
     */
    protected function pickThreeDistinctOthers(array $population, int $excludeIndex): array
    {
        $candidateIndexes = array_keys($population);
        $candidateIndexes = array_filter($candidateIndexes, fn (int $index): bool => $index !== $excludeIndex);
        $candidateIndexes = array_values($candidateIndexes);

        $pickedIndexes = [];

        while (count($pickedIndexes) < 3) {
            $index = $candidateIndexes[$this->randomizer->getInt(0, count($candidateIndexes) - 1)];

            if (in_array($index, $pickedIndexes, true)) {
                continue;
            }

            $pickedIndexes[] = $index;
        }

        return [$population[$pickedIndexes[0]], $population[$pickedIndexes[1]], $population[$pickedIndexes[2]]];
    }
}
