<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm\Internal;

/**
 * Which of {@see TerminationCriteria}'s four checks (or, for {@see \BlackboxOptimizer\Algorithm\DifferentialEvolutionAlgorithm},
 * its own reimplemented equivalents -- see that class's own docblock) made an `optimize()` call stop before
 * its configured iteration budget. `NONE` covers both "the run used its full budget" and "trustTerminationCriteria()
 * was never enabled" -- {@see \BlackboxOptimizer\Algorithm\OptimizationResult::getBestValueHistory()} being
 * shorter than the requested iteration count remains the way to tell those two apart, exactly as before this
 * enum existed; this only adds the "if it DID stop early, why" half of the picture.
 *
 * Introduced for {@see \BlackboxOptimizer\Algorithm\RestartingOptimizerDecorator}, which needs to tell a
 * genuine fitness plateau (`TOL_FUN` -- the case worth restarting from a fresh point) apart from convergence
 * (`TOL_X`), divergence (`TOL_X_UP`), or numerical degeneracy (`CONDITION_COV`), none of which a restart
 * would help with: a converged run found what it could from this starting point, a diverged or degenerate
 * one has a problem no amount of restarting fixes.
 */
enum TerminationReason
{
    /**
     * The run completed its full configured iteration budget without triggering any early-termination
     * check.
     */
    case NONE;

    /**
     * TolX -- the step size (or, for {@see \BlackboxOptimizer\Algorithm\DifferentialEvolutionAlgorithm},
     * the population's own spread) has collapsed to effectively zero. The search has converged as far as
     * it can from this starting point.
     */
    case TOL_X;

    /**
     * TolXUp -- the step size has blown up past a sane multiple of its starting value. The standard signal
     * of a diverged run, often caused by an objective that is unbounded or badly scaled for this algorithm.
     */
    case TOL_X_UP;

    /**
     * ConditionCov -- the covariance matrix's eigenvalues span too many orders of magnitude to stay
     * numerically trustworthy. Only ever reported by {@see \BlackboxOptimizer\Algorithm\CmaEsAlgorithm}.
     */
    case CONDITION_COV;

    /**
     * TolFun -- the best-found value has stopped meaningfully changing over a real trailing window: a
     * genuine fitness plateau. The one reason {@see \BlackboxOptimizer\Algorithm\RestartingOptimizerDecorator}
     * treats as worth restarting from a fresh point rather than accepting as a final answer.
     */
    case TOL_FUN;
}
