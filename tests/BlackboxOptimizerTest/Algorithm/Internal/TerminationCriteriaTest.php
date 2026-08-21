<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizerTest\Algorithm\Internal;

use BlackboxOptimizer\Algorithm\Internal\TerminationCriteria;
use BlackboxOptimizer\Algorithm\Internal\TerminationReason;
use PHPUnit\Framework\TestCase;

class TerminationCriteriaTest extends TestCase
{
    /**
     * Hansen's own formula: 10 + ceil(30n/lambda).
     *
     * @return void
     */
    public function testResolveFitnessHistoryLengthMatchesHansensFormula(): void
    {
        $terminationCriteria = new TerminationCriteria();

        $this->assertSame(14, $terminationCriteria->resolveFitnessHistoryLength(1, 8), '10 + ceil(30*1/8) = 10 + 4 = 14.');
        $this->assertSame(40, $terminationCriteria->resolveFitnessHistoryLength(12, 12), '10 + ceil(30*12/12) = 10 + 30 = 40.');
    }

    /**
     * @return void
     */
    public function testShouldTerminateEarlyReturnsFalseWhenNothingHasConvergedDivergedOrPlateaued(): void
    {
        $terminationCriteria = new TerminationCriteria();

        $result = $terminationCriteria->shouldTerminateEarly(
            sigma: 1.0,
            initialSigma: 1.0,
            sqrtEigenvalues: [1.0, 1.0],
            recentGenerationBestValues: [10.0, 5.0, 1.0],
            fitnessHistoryLength: 14,
        );

        $this->assertSame(TerminationReason::NONE, $result);
    }

    /**
     * TolX -- step size collapsed to effectively zero relative to where it started.
     *
     * @return void
     */
    public function testShouldTerminateEarlyOnTolX(): void
    {
        $terminationCriteria = new TerminationCriteria();

        $result = $terminationCriteria->shouldTerminateEarly(
            sigma: 1.0E-13,
            initialSigma: 1.0,
            sqrtEigenvalues: [1.0, 1.0],
            recentGenerationBestValues: [],
            fitnessHistoryLength: 14,
        );

        $this->assertSame(TerminationReason::TOL_X, $result);
    }

    /**
     * TolXUp -- step size blew up past the starting value, the standard CMA-ES divergence signal.
     *
     * @return void
     */
    public function testShouldTerminateEarlyOnTolXUp(): void
    {
        $terminationCriteria = new TerminationCriteria();

        $result = $terminationCriteria->shouldTerminateEarly(
            sigma: 1.0E5,
            initialSigma: 1.0,
            sqrtEigenvalues: [1.0, 1.0],
            recentGenerationBestValues: [],
            fitnessHistoryLength: 14,
        );

        $this->assertSame(TerminationReason::TOL_X_UP, $result);
    }

    /**
     * ConditionCov -- the covariance matrix's eigenvalues span too many orders of magnitude to stay
     * numerically trustworthy.
     *
     * @return void
     */
    public function testShouldTerminateEarlyOnConditionCov(): void
    {
        $terminationCriteria = new TerminationCriteria();

        $result = $terminationCriteria->shouldTerminateEarly(
            // A spread of [1.0, 1.0E8] (rather than [1.0E8, 1.0]) would also blow sigma * maxSqrtEigenvalue
            // past TolXUp's own threshold, triggering that check FIRST -- [1.0, 1.0E-8] isolates ConditionCov
            // alone: max stays small enough to keep sigma * maxSqrtEigenvalue inside both TolX/TolXUp
            // bounds, while (max/min)^2 = 1.0E16 still clears CONDITION_NUMBER_LIMIT (1.0E14).
            sigma: 1.0,
            initialSigma: 1.0,
            sqrtEigenvalues: [1.0, 1.0E-8],
            recentGenerationBestValues: [],
            fitnessHistoryLength: 14,
        );

        $this->assertSame(TerminationReason::CONDITION_COV, $result);
    }

    /**
     * TolFun -- the best-of-generation value has stopped meaningfully changing over a real trailing window.
     *
     * @return void
     */
    public function testShouldTerminateEarlyOnTolFunOnceTheHistoryWindowIsFull(): void
    {
        $terminationCriteria = new TerminationCriteria();

        $result = $terminationCriteria->shouldTerminateEarly(
            sigma: 1.0,
            initialSigma: 1.0,
            sqrtEigenvalues: [1.0, 1.0],
            recentGenerationBestValues: array_fill(0, 14, 5.0),
            fitnessHistoryLength: 14,
        );

        $this->assertSame(TerminationReason::TOL_FUN, $result);
    }

    /**
     * The same flat history that triggers {@see testShouldTerminateEarlyOnTolFunOnceTheHistoryWindowIsFull()}
     * must NOT trigger before the window has actually filled -- proves the length check, not just the
     * range check, is load-bearing.
     *
     * @return void
     */
    public function testShouldTerminateEarlyDoesNotTriggerTolFunBeforeTheHistoryWindowFills(): void
    {
        $terminationCriteria = new TerminationCriteria();

        $result = $terminationCriteria->shouldTerminateEarly(
            sigma: 1.0,
            initialSigma: 1.0,
            sqrtEigenvalues: [1.0, 1.0],
            recentGenerationBestValues: array_fill(0, 13, 5.0),
            fitnessHistoryLength: 14,
        );

        $this->assertSame(TerminationReason::NONE, $result);
    }
}
