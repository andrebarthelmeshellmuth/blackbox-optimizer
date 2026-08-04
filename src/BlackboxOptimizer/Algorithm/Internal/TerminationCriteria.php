<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm\Internal;

/**
 * The standard CMA-ES early-termination criteria (TolX, TolXUp, ConditionCov, TolFun -- see Hansen's own
 * "The CMA Evolution Strategy: A Tutorial" and the purecma/pycma reference implementations), extracted out
 * of {@see \BlackboxOptimizer\Algorithm\CmaEsAlgorithm} once adding them tipped that class's own complexity
 * over this package's phpmd threshold -- the same reasoning that already justified extracting
 * {@see VectorMath} and {@see SymmetricEigenDecomposition}: these are a self-contained concern, not
 * CMA-ES's own core covariance-adaptation math.
 *
 * @internal Used only by {@see \BlackboxOptimizer\Algorithm\CmaEsAlgorithm}.
 */
class TerminationCriteria
{
    /**
     * TolX -- stop once the largest step CMA-ES could still take, in any direction (`sigma * max(sqrt(eigenvalues))`),
     * has shrunk below this fraction of its own starting value. Converged: further generations would not
     * meaningfully move the search. Same name/role as pycma's own `tolx` (relative to `sigma0` here, since
     * CmaEsAlgorithm's `sigma` is already in the problem's own units, never normalized).
     *
     * @var float
     */
    protected const TOL_X_FACTOR = 1.0E-11;

    /**
     * TolXUp -- the opposite failure: stop if that same largest step has grown past this multiple of its
     * starting value instead. A blown-up step size is the standard CMA-ES signal of a diverged run, often
     * caused by an objective that is unbounded or badly scaled for this algorithm.
     *
     * @var float
     */
    protected const TOL_X_UP_FACTOR = 1.0E4;

    /**
     * ConditionCov -- stop if the covariance matrix's eigenvalues span more than this many orders of
     * magnitude. Beyond this point the eigendecomposition is no longer numerically trustworthy: the
     * matrix is effectively degenerate in its most-compressed direction.
     *
     * @var float
     */
    protected const CONDITION_NUMBER_LIMIT = 1.0E14;

    /**
     * TolFun -- stop once the range (max - min) of each generation's own best-found value, over a real
     * trailing window (see {@see resolveFitnessHistoryLength()}), drops below this. Hansen's own default;
     * assumes a reasonably-scaled objective the same way the reference implementations do -- an objective
     * whose meaningful differences are themselves smaller than 1e-12 would need a tighter value passed via
     * a future setter, not currently exposed.
     *
     * @var float
     */
    protected const TOL_FUN = 1.0E-12;

    /**
     * Hansen's own history length for the TolFun check below -- long enough that short-term noise across a
     * handful of generations can't trigger it, short enough to still catch a genuine plateau well before a
     * typical run's own maxIterations.
     *
     * @param int $n
     * @param int $lambda
     *
     * @return int
     */
    public function resolveFitnessHistoryLength(int $n, int $lambda): int
    {
        return (int)(10 + ceil(30 * $n / $lambda));
    }

    /**
     * Checked once per generation, after that generation's own state has fully settled. A run that never
     * triggers any of these four simply runs its full maxIterations budget, exactly as if this class did
     * not exist. No field anywhere reports which criterion fired (or whether any did) --
     * {@see \BlackboxOptimizer\Algorithm\OptimizationResult::getBestValueHistory()} being shorter than
     * maxIterations is the caller-visible signal a run stopped early, an existing part of the API rather
     * than a new one.
     *
     * @param float $sigma
     * @param float $initialSigma
     * @param array<int, float> $sqrtEigenvalues
     * @param array<int, float> $recentGenerationBestValues
     * @param int $fitnessHistoryLength
     *
     * @return bool
     */
    public function shouldTerminateEarly(
        float $sigma,
        float $initialSigma,
        array $sqrtEigenvalues,
        array $recentGenerationBestValues,
        int $fitnessHistoryLength,
    ): bool {
        $maxSqrtEigenvalue = $this->arrayMax($sqrtEigenvalues);
        $minSqrtEigenvalue = max($this->arrayMin($sqrtEigenvalues), PHP_FLOAT_EPSILON);

        if ($sigma * $maxSqrtEigenvalue < static::TOL_X_FACTOR * $initialSigma) {
            return true;
        }

        if ($sigma * $maxSqrtEigenvalue > static::TOL_X_UP_FACTOR * $initialSigma) {
            return true;
        }

        if (($maxSqrtEigenvalue / $minSqrtEigenvalue) ** 2 > static::CONDITION_NUMBER_LIMIT) {
            return true;
        }

        return $this->hasFlatFitnessHistory($recentGenerationBestValues, $fitnessHistoryLength);
    }

    /**
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

        $range = $this->arrayMax($recentGenerationBestValues) - $this->arrayMin($recentGenerationBestValues);

        return $range < static::TOL_FUN;
    }

    /**
     * Plain loop rather than max() directly -- both callers' arrays are only ever empty if
     * CmaEsAlgorithm's own invariants (at least one dimension; the fitness-history array is only compared
     * once it already holds $fitnessHistoryLength entries) already forbid, but nothing makes that visible
     * to static analysis across the call boundary.
     *
     * @param array<int, float> $values
     *
     * @return float
     */
    protected function arrayMax(array $values): float
    {
        $result = -INF;

        foreach ($values as $value) {
            if ($value > $result) {
                $result = $value;
            }
        }

        return $result;
    }

    /**
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
}
