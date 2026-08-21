<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm;

use BlackboxOptimizer\Algorithm\Internal\TerminationReason;

/**
 * Plain value object -- deliberately has no dependency on anything else in this package beyond plain
 * arrays/scalars (and the {@see TerminationReason} enum, itself dependency-free) -- so a consumer can
 * type-hint against it without pulling in the rest of the Algorithm namespace.
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
     * @param \BlackboxOptimizer\Algorithm\Internal\TerminationReason $terminationReason Which early-termination
     *   check (if any) stopped this run before it used its full iteration budget. {@see TerminationReason::NONE}
     *   for a run that used its full budget, or one that never enabled early termination at all --
     *   {@see getBestValueHistory()} being shorter than the requested iteration count remains how those two
     *   cases are told apart.
     * @param array<int, \BlackboxOptimizer\Algorithm\RestartHistoryEntry> $restartHistory Empty for a plain
     *   `optimize()` call; populated by {@see RestartingOptimizerDecorator} with one entry per restart it
     *   ran, oldest first.
     */
    public function __construct(
        private array $bestVector,
        private float $bestValue,
        private int $evaluationCount,
        private array $bestValueHistory = [],
        private TerminationReason $terminationReason = TerminationReason::NONE,
        private array $restartHistory = [],
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

    /**
     * @return \BlackboxOptimizer\Algorithm\Internal\TerminationReason
     */
    public function getTerminationReason(): TerminationReason
    {
        return $this->terminationReason;
    }

    /**
     * @return array<int, \BlackboxOptimizer\Algorithm\RestartHistoryEntry>
     */
    public function getRestartHistory(): array
    {
        return $this->restartHistory;
    }
}
