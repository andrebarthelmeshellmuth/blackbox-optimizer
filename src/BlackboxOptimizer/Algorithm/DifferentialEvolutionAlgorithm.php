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
 * Classic DE/rand/1/bin Differential Evolution — mutation (a + F * (b - c)) + binomial crossover +
 * greedy selection, nothing more. Deliberately the simplest population-based black-box optimizer in this
 * namespace: no covariance adaptation, no step-size control, just a handful of arithmetic operations per
 * candidate — see this package's README for why this exists alongside CmaEsAlgorithm rather than instead
 * of it (it's "the thing to beat," and it doubles as proof {@see OptimizerAlgorithmInterface} genuinely
 * generalizes beyond CMA-ES's own shape, not just a second implementation of the same idea).
 *
 * {@see OptimizerAlgorithmInterface::setStepWidth()} maps to "F" here (the mutation factor scaling the
 * differential b - c) — the closest DE equivalent to "how big are the first exploratory steps."
 *
 * Early termination (see {@see trustTerminationCriteria()}) does NOT reuse
 * {@see \BlackboxOptimizer\Algorithm\Internal\TerminationCriteria}'s own TolX/TolXUp/ConditionCov the way
 * {@see RechenbergSchwefelEsAlgorithm} does -- DE has no scalar step size or covariance to check in the
 * first place (the mutation factor F is fixed, never adapted, so there is nothing analogous to "sigma blew
 * up"). What DE has instead is a whole population of candidate points, so its own TolX-equivalent is
 * **population convergence**: every dimension's spread across the current population has collapsed
 * relative to that dimension's own bound range (see {@see hasPopulationCollapsed()}). Only
 * {@see TerminationCriteria::resolveFitnessHistoryLength()}'s window-length formula is reused as-is (a
 * generic formula, not specific to sigma/eigenvalues); the fitness-plateau check itself is reimplemented
 * locally rather than exposed as a public method on that class, since it's a handful of lines not worth an
 * extra dependency edge for.
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
     * DE's own TolX-equivalent: stop once every dimension's spread across the current population (max -
     * min) has shrunk below this fraction of that dimension's own bound range. Relative to the bounds
     * (not an initial step size, unlike CmaEsAlgorithm/RechenbergSchwefelEsAlgorithm) since DE has no
     * step-size concept of its own to measure against -- the search space's own size is the only fixed
     * reference scale available.
     *
     * @var float
     */
    protected const POPULATION_SPREAD_TOLERANCE = 1.0E-8;

    /**
     * Same role and value as {@see TerminationCriteria::TOL_FUN} -- kept as DE's own constant rather than
     * reusing that class's (private-in-effect, protected) one directly, so this class doesn't need to know
     * TerminationCriteria's internals to stay consistent with it.
     *
     * @var float
     */
    protected const TOL_FUN = 1.0E-12;

    /**
     * See {@see CmaEsAlgorithm::SAFETY_ITERATION_CEILING} -- same role, same value, same reasoning.
     *
     * @var int
     */
    protected const SAFETY_ITERATION_CEILING = 10000;

    /**
     * @var float
     */
    protected float $crossoverProbability = self::DEFAULT_CROSSOVER_PROBABILITY;

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
     * Algorithm-specific setup, deliberately NOT part of {@see OptimizerAlgorithmInterface} -- call before
     * optimize() to opt in. Same role as {@see CmaEsAlgorithm::trustTerminationCriteria()}: switches the
     * effective iteration ceiling from {@see DEFAULT_MAX_ITERATIONS}/whatever {@see setMaxIterations()}
     * was given to {@see SAFETY_ITERATION_CEILING}, so a run is governed by the population having
     * genuinely converged or its fitness having plateaued (see this class's own docblock for how DE's
     * criteria differ from CmaEsAlgorithm's) -- not by an arbitrary generation count. Any
     * {@see setMaxIterations()} call is ignored once this is on. Not a literal unbounded loop for the same
     * reason CmaEsAlgorithm's own version isn't -- see that method's docblock.
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
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    public function optimize(ProblemInterface $problem): OptimizationResult
    {
        $this->resetTracking();
        [$lowerBounds, $upperBounds] = $this->extractBounds($problem);
        $dimensionCount = count($lowerBounds);

        $populationSize = $this->populationSize ?? static::DEFAULT_POPULATION_SIZE;
        $mutationFactor = $this->stepWidth ?? static::DEFAULT_STEP_WIDTH;
        $maxIterations = $this->trustTerminationCriteria
            ? static::SAFETY_ITERATION_CEILING
            : ($this->maxIterations ?? static::DEFAULT_MAX_ITERATIONS);
        $fitnessHistoryLength = $this->terminationCriteria->resolveFitnessHistoryLength($dimensionCount, $populationSize);
        $recentGenerationBestValues = [];

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

            $recentGenerationBestValues[] = $this->arrayMin($populationValues);

            if (count($recentGenerationBestValues) > $fitnessHistoryLength) {
                array_shift($recentGenerationBestValues);
            }

            if (
                $this->hasPopulationCollapsed($population, $lowerBounds, $upperBounds)
                || $this->hasFlatFitnessHistory($recentGenerationBestValues, $fitnessHistoryLength)
            ) {
                break;
            }
        }

        return $this->buildResult();
    }

    /**
     * DE's own TolX-equivalent -- see this class's own docblock for why population spread, not a step
     * size, is the right thing to check here.
     *
     * @param array<int, array<int, float>> $population
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @return bool
     */
    protected function hasPopulationCollapsed(array $population, array $lowerBounds, array $upperBounds): bool
    {
        foreach ($lowerBounds as $dimension => $lowerBound) {
            $boundRange = $upperBounds[$dimension] - $lowerBound;
            $min = INF;
            $max = -INF;

            foreach ($population as $vector) {
                $value = $vector[$dimension];

                if ($value < $min) {
                    $min = $value;
                }

                if ($value > $max) {
                    $max = $value;
                }
            }

            if (($max - $min) >= static::POPULATION_SPREAD_TOLERANCE * max($boundRange, PHP_FLOAT_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Same TolFun idea as {@see TerminationCriteria::shouldTerminateEarly()}'s own fitness-plateau check,
     * reimplemented locally rather than shared -- see this class's own docblock for why.
     *
     * @param array<int, float> $recentGenerationBestValues
     * @param int $fitnessHistoryLength
     *
     * @return bool
     */
    protected function hasFlatFitnessHistory(array $recentGenerationBestValues, int $fitnessHistoryLength): bool
    {
        if (count($recentGenerationBestValues) < $fitnessHistoryLength) {
            return false;
        }

        $min = INF;
        $max = -INF;

        foreach ($recentGenerationBestValues as $value) {
            if ($value < $min) {
                $min = $value;
            }

            if ($value > $max) {
                $max = $value;
            }
        }

        return ($max - $min) < static::TOL_FUN;
    }

    /**
     * Plain loop rather than min() directly -- $populationValues is only ever empty if populationSize
     * could be 0, which {@see AbstractOptimizerAlgorithm::setPopulationSize()} already forbids (minimum
     * 4), but nothing makes that invariant visible to static analysis across the call boundary.
     *
     * @param array<int, float> $values
     *
     * @return float
     */
    protected function arrayMin(array $values): float
    {
        $result = INF;

        foreach ($values as $value) {
            if ($value < $result) {
                $result = $value;
            }
        }

        return $result;
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
