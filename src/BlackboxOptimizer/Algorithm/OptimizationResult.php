<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm;

/**
 * Plain value object -- deliberately has no dependency on anything else in this package beyond plain
 * arrays/scalars, so a consumer can type-hint against it without pulling in the rest of the Algorithm
 * namespace.
 */
final class OptimizationResult
{
    /**
     * @param array<int, float> $bestVector
     * @param float $bestValue
     * @param int $evaluationCount
     * @param array<int, float> $bestValueHistory One entry per generation/iteration -- the best value found
     *   so far at that point, oldest first. Useful for convergence checks and for diagnosing a run that
     *   didn't converge; empty if an algorithm doesn't track it.
     */
    public function __construct(
        private array $bestVector,
        private float $bestValue,
        private int $evaluationCount,
        private array $bestValueHistory = [],
    ) {
    }

    /**
     * @return array<int, float>
     */
    public function getBestVector(): array
    {
        return $this->bestVector;
    }

    /**
     * @return float
     */
    public function getBestValue(): float
    {
        return $this->bestValue;
    }

    /**
     * @return int
     */
    public function getEvaluationCount(): int
    {
        return $this->evaluationCount;
    }

    /**
     * @return array<int, float>
     */
    public function getBestValueHistory(): array
    {
        return $this->bestValueHistory;
    }
}
