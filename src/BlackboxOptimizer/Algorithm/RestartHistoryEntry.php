<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm;

use BlackboxOptimizer\Algorithm\Internal\TerminationReason;

/**
 * One restart's own summary, as recorded by {@see RestartingOptimizerDecorator} -- a plain value object
 * with the same dependency-free design as {@see OptimizationResult} itself. A caller building a report
 * ("Restart 0: converged at 0.71 after 29 generations, doubled to population 24...") reads these off
 * {@see OptimizationResult::getRestartHistory()} directly rather than re-deriving them from raw run state.
 */
final class RestartHistoryEntry
{
    /**
     * @param int $restartIndex 0-based -- the FIRST run (before any restart has happened) is index 0, same
     *   as every restart after it, so a caller can render "Restart N" uniformly without a special case for
     *   the initial run.
     * @param int $populationSize The population size this restart actually ran with (doubled from the
     *   previous restart's, except for index 0).
     * @param int $generationsAllowed The generation cap this restart was given -- may be less than the
     *   caller's own originally configured cap, when the remaining evaluation budget couldn't afford a full
     *   one at this restart's (larger) population size.
     * @param int $generationsUsed How many generations this restart actually ran before stopping -- equal
     *   to $generationsAllowed only when it used its full allowance rather than stopping early.
     * @param \BlackboxOptimizer\Algorithm\Internal\TerminationReason $terminationReason Why this restart
     *   stopped -- {@see TerminationReason::NONE} if it used its full generation allowance rather than
     *   triggering an early-termination check.
     * @param float $bestValueAtStop The best value this restart itself found.
     * @param int $evaluationsConsumed How many {@see \BlackboxOptimizer\Problem\ProblemInterface::evaluate()}
     *   calls this restart made.
     * @param bool $improvedOverallBest Whether this restart's own best value beat every prior restart's --
     *   i.e. whether it's the source of {@see OptimizationResult::getBestValue()}/{@see OptimizationResult::getBestVector()}
     *   on the final, decorator-returned result.
     */
    public function __construct(
        private int $restartIndex,
        private int $populationSize,
        private int $generationsAllowed,
        private int $generationsUsed,
        private TerminationReason $terminationReason,
        private float $bestValueAtStop,
        private int $evaluationsConsumed,
        private bool $improvedOverallBest,
    ) {
    }

    /**
     * @return int
     */
    public function getRestartIndex(): int
    {
        return $this->restartIndex;
    }

    /**
     * @return int
     */
    public function getPopulationSize(): int
    {
        return $this->populationSize;
    }

    /**
     * @return int
     */
    public function getGenerationsAllowed(): int
    {
        return $this->generationsAllowed;
    }

    /**
     * @return int
     */
    public function getGenerationsUsed(): int
    {
        return $this->generationsUsed;
    }

    /**
     * @return \BlackboxOptimizer\Algorithm\Internal\TerminationReason
     */
    public function getTerminationReason(): TerminationReason
    {
        return $this->terminationReason;
    }

    /**
     * @return float
     */
    public function getBestValueAtStop(): float
    {
        return $this->bestValueAtStop;
    }

    /**
     * @return int
     */
    public function getEvaluationsConsumed(): int
    {
        return $this->evaluationsConsumed;
    }

    /**
     * @return bool
     */
    public function improvedOverallBest(): bool
    {
        return $this->improvedOverallBest;
    }
}
