<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizerTest\Algorithm;

use BlackboxOptimizer\Algorithm\OptimizationResult;
use BlackboxOptimizer\Algorithm\OptimizerAlgorithmInterface;
use BlackboxOptimizer\Problem\ProblemInterface;

/**
 * A test double returning a scripted, predetermined sequence of {@see OptimizationResult}s -- one per
 * optimize() call, in order -- so {@see RestartingOptimizerDecoratorTest} can pin exactly what
 * {@see \BlackboxOptimizer\Algorithm\RestartingOptimizerDecorator} sees without depending on a real
 * algorithm's stochastic behavior.
 */
final class ScriptedOptimizerAlgorithm implements OptimizerAlgorithmInterface
{
    private int $callIndex = 0;

    /**
     * @var array<int, int>
     */
    public array $populationSizesSeen = [];

    /**
     * @var array<int, int>
     */
    public array $maxIterationsSeen = [];

    /**
     * @var array<int, array{0: array<int, float>, 1: float}>
     */
    public array $warmStartCalls = [];

    /**
     * @param array<int, \BlackboxOptimizer\Algorithm\OptimizationResult> $scriptedResults
     */
    public function __construct(private array $scriptedResults)
    {
    }

    /**
     * @return int
     */
    public function getOptimizeCallCount(): int
    {
        return $this->callIndex;
    }

    /**
     * @return string
     */
    public function getName(): string
    {
        return 'Scripted';
    }

    /**
     * @return string
     */
    public function getDescription(): string
    {
        return 'A test double returning a scripted result sequence.';
    }

    /**
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     *
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    // phpcs:disable SlevomatCodingStandard.Functions.UnusedParameter -- must match OptimizerAlgorithmInterface::optimize().
    public function optimize(ProblemInterface $problem): OptimizationResult
    {
        return $this->scriptedResults[$this->callIndex++];
    }
    // phpcs:enable SlevomatCodingStandard.Functions.UnusedParameter

    /**
     * @return int
     */
    public function estimateEvaluationCount(): int
    {
        return 0;
    }

    /**
     * @return static
     */
    public function trustTerminationCriteria(): static
    {
        return $this;
    }

    /**
     * @param array<int, float> $vector
     * @param float $fraction
     *
     * @return static
     */
    public function setWarmStart(array $vector, float $fraction): static
    {
        $this->warmStartCalls[] = [$vector, $fraction];

        return $this;
    }

    /**
     * @param float $stepWidth
     *
     * @return static
     */
    // phpcs:disable SlevomatCodingStandard.Functions.UnusedParameter -- must match OptimizerAlgorithmInterface::setStepWidth().
    public function setStepWidth(float $stepWidth): static
    {
        return $this;
    }
    // phpcs:enable SlevomatCodingStandard.Functions.UnusedParameter

    /**
     * @param int $populationSize
     *
     * @return static
     */
    public function setPopulationSize(int $populationSize): static
    {
        $this->populationSizesSeen[] = $populationSize;

        return $this;
    }

    /**
     * @param int $maxIterations
     *
     * @return static
     */
    public function setMaxIterations(int $maxIterations): static
    {
        $this->maxIterationsSeen[] = $maxIterations;

        return $this;
    }
}
