<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm;

use BlackboxOptimizer\Algorithm\Internal\TerminationReason;
use BlackboxOptimizer\Problem\ProblemInterface;
use InvalidArgumentException;
use LogicException;
use Random\Randomizer;

/**
 * IPOP-style restart wrapper (Auger & Hansen 2005) around any {@see OptimizerAlgorithmInterface} -- composed
 * around an inner algorithm rather than built into one, so {@see CmaEsAlgorithm}, {@see RechenbergSchwefelEsAlgorithm},
 * and {@see DifferentialEvolutionAlgorithm} all get this for free, and none of them need their own bespoke
 * restart machinery (each already documents that omission as a deliberate, separate decision).
 *
 * The trigger is deliberately narrow: a restart only happens when a run stops on {@see TerminationReason::TOL_FUN}
 * -- a genuine fitness plateau, the one case where "try again from a fresh point" is actually likely to help.
 * TOL_X (converged), TOL_X_UP (diverged), and CONDITION_COV (numerically degenerate) all stop this decorator
 * too, without restarting: a converged run already found what this starting point had to offer, and a
 * diverged/degenerate one has a problem restarting from a new point does not fix.
 *
 * Budget discipline: the TOTAL evaluation budget across every restart is fixed up front, at exactly
 * `populationSize * maxIterations` -- the same number a single, non-restarting {@see setPopulationSize()}/
 * {@see setMaxIterations()} call already means today (or, with {@see trustRestartBudget()}, `populationSize
 * * getSafetyIterationCeiling()` instead -- see that method's own docblock). A restart's own population size
 * doubles each time (classic IPOP), which is what actually bounds the restart COUNT: growth is geometric, so
 * even a modest budget only buys a handful of restarts before the next doubled population can no longer be
 * afforded, at which point this decorator stops and returns the best candidate found across every restart --
 * never more evaluations than the budget in effect promised, and never fewer restarts than that budget can
 * actually pay for.
 *
 * Deliberately does NOT support {@see trustTerminationCriteria()} (it throws) -- that mode replaces a
 * caller's own `maxIterations` with each algorithm's internal safety ceiling directly on the INNER
 * algorithm, bypassing this decorator's own per-restart accounting entirely. {@see trustRestartBudget()} is
 * this decorator's own, budget-aware answer to the same underlying need -- see that method's own docblock
 * for why the two are not the same thing wearing different names.
 */
final class RestartingOptimizerDecorator implements OptimizerAlgorithmInterface
{
    private ?int $initialPopulationSize = null;

    private ?int $maxGenerationsPerRestart = null;

    private bool $trustRestartBudget = false;

    private Randomizer $randomizer;

    /**
     * @param \BlackboxOptimizer\Algorithm\OptimizerAlgorithmInterface $innerAlgorithm The algorithm actually
     *   run each restart -- its own setPopulationSize()/setMaxIterations() are driven by this decorator, not
     *   the caller, once optimize() starts (see this class's own docblock).
     * @param \Random\Randomizer|null $randomizer Injectable for deterministic tests; defaults to a real
     *   random engine in production use. Draws the fresh starting point each restart after the first.
     */
    public function __construct(
        private OptimizerAlgorithmInterface $innerAlgorithm,
        ?Randomizer $randomizer = null,
    ) {
        $this->randomizer = $randomizer ?? new Randomizer();
    }

    /**
     * @return string
     */
    public function getName(): string
    {
        return $this->innerAlgorithm->getName() . ' (with plateau restarts)';
    }

    /**
     * @return string
     */
    public function getDescription(): string
    {
        return $this->innerAlgorithm->getDescription() . ' Wrapped in an IPOP-style restart: on a genuine '
            . 'fitness plateau, restarts from a fresh random point with a doubled population, within a '
            . 'fixed total evaluation budget.';
    }

    /**
     * {@inheritDoc}
     *
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     *
     * @throws \InvalidArgumentException When {@see setPopulationSize()} was never called, or when neither
     *   {@see setMaxIterations()} nor {@see trustRestartBudget()} was -- unlike the inner algorithms, this
     *   decorator has no fallback default: it needs to know the total restart budget from ONE of those two
     *   sources before it can do anything.
     *
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    public function optimize(ProblemInterface $problem): OptimizationResult
    {
        if ($this->initialPopulationSize === null) {
            throw new InvalidArgumentException(
                'RestartingOptimizerDecorator requires setPopulationSize() to be called -- with no fallback '
                . 'default the way a single inner algorithm run might have.',
            );
        }

        $totalBudget = $this->resolveTotalBudget($this->initialPopulationSize);
        [$lowerBounds, $upperBounds] = $this->extractBounds($problem);

        $remainingBudget = $totalBudget;
        $currentPopulationSize = $this->initialPopulationSize;
        $restartIndex = 0;

        $overallBestValue = null;
        $overallBestVector = [];
        $combinedHistory = [];
        $restartHistory = [];
        $totalEvaluations = 0;
        $finalTerminationReason = TerminationReason::NONE;

        while (true) {
            if ($restartIndex > 0) {
                $this->innerAlgorithm->setWarmStart($this->drawRandomVector($lowerBounds, $upperBounds), 1.0);
            }

            // Deliberately not clamped against $this->maxGenerationsPerRestart -- doing so would be a no-op
            // in the (default) fixed-budget mode (that budget is ITSELF populationSize * maxGenerationsPerRestart,
            // so this quotient can never exceed it: restart 0 gets exactly maxGenerationsPerRestart, and
            // every later restart strictly less, since population only grows and remaining budget only
            // shrinks), and in trustRestartBudget() mode there is no $maxGenerationsPerRestart to clamp
            // against in the first place -- that's the whole point of that mode.
            $generationsAllowed = intdiv($remainingBudget, $currentPopulationSize);

            if ($generationsAllowed < 1) {
                break;
            }

            $this->innerAlgorithm->setPopulationSize($currentPopulationSize);
            $this->innerAlgorithm->setMaxIterations($generationsAllowed);

            $result = $this->innerAlgorithm->optimize($problem);

            $evaluationsConsumed = $result->getEvaluationCount();
            $remainingBudget -= $evaluationsConsumed;
            $totalEvaluations += $evaluationsConsumed;
            $combinedHistory = array_merge($combinedHistory, $result->getBestValueHistory());
            $finalTerminationReason = $result->getTerminationReason();

            $improvedOverallBest = $overallBestValue === null || $result->getBestValue() < $overallBestValue;

            if ($improvedOverallBest) {
                $overallBestValue = $result->getBestValue();
                $overallBestVector = $result->getBestVector();
            }

            $restartHistory[] = new RestartHistoryEntry(
                $restartIndex,
                $currentPopulationSize,
                $generationsAllowed,
                count($result->getBestValueHistory()),
                $finalTerminationReason,
                $result->getBestValue(),
                $evaluationsConsumed,
                $improvedOverallBest,
            );

            if ($finalTerminationReason !== TerminationReason::TOL_FUN) {
                break;
            }

            $currentPopulationSize *= 2;
            $restartIndex++;
        }

        return new OptimizationResult(
            $overallBestVector,
            $overallBestValue ?? INF,
            $totalEvaluations,
            $combinedHistory,
            $finalTerminationReason,
            $restartHistory,
        );
    }

    /**
     * {@inheritDoc}
     *
     * The total restart budget itself -- see this class's own docblock for why that, not a per-restart
     * estimate, is the honest answer: this decorator guarantees it never exceeds this number, however many
     * restarts actually happen within it. Reflects {@see trustRestartBudget()} when that's in effect, same
     * as {@see optimize()} itself.
     *
     * @throws \InvalidArgumentException When {@see setPopulationSize()} was never called, or when neither
     *   {@see setMaxIterations()} nor {@see trustRestartBudget()} was.
     *
     * @return int
     */
    public function estimateEvaluationCount(): int
    {
        if ($this->initialPopulationSize === null) {
            throw new InvalidArgumentException(
                'RestartingOptimizerDecorator requires setPopulationSize() to be called -- with no fallback '
                . 'default the way a single inner algorithm run might have.',
            );
        }

        return $this->resolveTotalBudget($this->initialPopulationSize);
    }

    /**
     * {@inheritDoc}
     *
     * Deliberately unsupported -- see this class's own docblock for why trusting the INNER algorithm's own
     * safety ceiling directly, bypassing this decorator entirely, would blow through this decorator's own
     * per-restart accounting. {@see trustRestartBudget()} is this decorator's own, budget-aware equivalent.
     *
     * @throws \LogicException Always.
     *
     * @return never
     */
    public function trustTerminationCriteria(): static
    {
        throw new LogicException(
            'RestartingOptimizerDecorator does not support trustTerminationCriteria() -- it would let each '
            . 'restart run out to the inner algorithm\'s own safety ceiling directly, bypassing this '
            . 'decorator\'s own per-restart accounting entirely. Call trustRestartBudget() instead -- it '
            . 'gives every restart that same generous ceiling, but keeps the decorator\'s own bookkeeping in '
            . 'the loop so restarts still stay bounded and reported the normal way.',
        );
    }

    /**
     * Opts into a MUCH larger total restart budget -- `populationSize * getSafetyIterationCeiling()`
     * (the inner algorithm's own generous internal ceiling, the same one {@see trustTerminationCriteria()}
     * uses for a single, non-restarting run) instead of `populationSize * maxIterations` (a caller-chosen,
     * typically much smaller number). Makes {@see setMaxIterations()} optional (and, if still called,
     * ignored) -- mirroring how a plain algorithm's own `trustTerminationCriteria()` already makes ITS
     * `setMaxIterations()` call irrelevant.
     *
     * Why this genuinely differs from just raising a per-restart ceiling on its own (a fix that would do
     * NOTHING under the default budget): every restart's own ceiling is `min(ceiling, remainingBudget /
     * currentPopulationSize)`, and under the default budget (`populationSize * maxIterations`), restart 0's
     * own `remainingBudget / populationSize` is EXACTLY `maxIterations` by construction -- a separate,
     * larger ceiling constant would never be the smaller side of that `min()`, so it would never actually
     * bind. Growing the TOTAL budget itself (this method) is what actually gives restart 0 (and, via the
     * same shrinking-quotient math as every other restart, every restart after it) real room: restart 0 gets
     * up to `getSafetyIterationCeiling()` generations, a doubled-population restart 1 gets roughly half that,
     * restart 2 roughly a quarter, and so on -- the exact same self-limiting shape this decorator already
     * has, just seeded from a far larger starting number.
     *
     * The tradeoff, same one `trustTerminationCriteria()` already makes for a single run: the strong
     * "total evaluations never exceed what you configured" guarantee this decorator otherwise offers is
     * replaced by a much looser one -- bounded by `getSafetyIterationCeiling()`, not by a number the caller
     * chose. A run that never triggers any of TolX/TolXUp/ConditionCov/TolFun at any restart could
     * genuinely use close to that much budget before this decorator gives up.
     *
     * @return static
     */
    public function trustRestartBudget(): static
    {
        $this->trustRestartBudget = true;

        return $this;
    }

    /**
     * {@inheritDoc}
     *
     * Forwarded to the inner algorithm as-is -- this decorator has no ceiling of its own distinct from the
     * one the algorithm it wraps would use.
     *
     * @return int
     */
    public function getSafetyIterationCeiling(): int
    {
        return $this->innerAlgorithm->getSafetyIterationCeiling();
    }

    /**
     * @throws \InvalidArgumentException When {@see setPopulationSize()} was never called, or when neither
     *   {@see setMaxIterations()} nor {@see trustRestartBudget()} was.
     *
     * @return int
     */
    /**
     * @param int $populationSize The caller's already-validated {@see setPopulationSize()} value -- taken
     *   as a parameter (not read from `$this->initialPopulationSize` directly) purely so static analysis
     *   can see it's non-null at every call site, each of which already checked that itself.
     *
     * @throws \InvalidArgumentException When neither {@see setMaxIterations()} nor {@see trustRestartBudget()}
     *   was called.
     *
     * @return int
     */
    private function resolveTotalBudget(int $populationSize): int
    {
        if ($this->trustRestartBudget) {
            return $populationSize * $this->innerAlgorithm->getSafetyIterationCeiling();
        }

        if ($this->maxGenerationsPerRestart === null) {
            throw new InvalidArgumentException(
                'RestartingOptimizerDecorator requires either setMaxIterations() or trustRestartBudget() to '
                . 'be called -- the total restart budget is populationSize * one of those two, with no '
                . 'fallback default the way a single inner algorithm run might have.',
            );
        }

        return $populationSize * $this->maxGenerationsPerRestart;
    }

    /**
     * {@inheritDoc}
     *
     * Forwarded to the inner algorithm as-is, and only actually used for the FIRST run -- every restart
     * after that overrides it with a fresh random point (see this class's own docblock).
     *
     * @param array<int, float> $vector
     * @param float $fraction
     *
     * @return static
     */
    public function setWarmStart(array $vector, float $fraction): static
    {
        $this->innerAlgorithm->setWarmStart($vector, $fraction);

        return $this;
    }

    /**
     * {@inheritDoc}
     *
     * Forwarded to the inner algorithm as-is.
     *
     * @param float $stepWidth
     *
     * @return static
     */
    public function setStepWidth(float $stepWidth): static
    {
        $this->innerAlgorithm->setStepWidth($stepWidth);

        return $this;
    }

    /**
     * {@inheritDoc}
     *
     * The STARTING population size -- restart 0's own, before any doubling. Required (see {@see optimize()}):
     * this decorator has no fallback default, since it defines the total restart budget together with
     * {@see setMaxIterations()}.
     *
     * @param int $populationSize
     *
     * @return static
     */
    public function setPopulationSize(int $populationSize): static
    {
        if ($populationSize < 4) {
            throw new InvalidArgumentException('populationSize must be at least 4.');
        }

        $this->initialPopulationSize = $populationSize;

        return $this;
    }

    /**
     * {@inheritDoc}
     *
     * Together with {@see setPopulationSize()}, defines the total restart budget (`populationSize *
     * maxIterations`) -- unless {@see trustRestartBudget()} is used instead, in which case this value is
     * still stored (calling this first, then trustRestartBudget(), is harmless) but ignored when computing
     * the budget, the same way a plain algorithm's own `trustTerminationCriteria()` already ignores its
     * `setMaxIterations()`. Restart 0 gets exactly this many generations; every restart after that gets
     * strictly less, since the remaining budget shrinks while the (doubled) population it's divided by
     * grows -- see {@see RestartHistoryEntry::getGenerationsAllowed()}.
     *
     * @param int $maxIterations
     *
     * @return static
     */
    public function setMaxIterations(int $maxIterations): static
    {
        if ($maxIterations < 1) {
            throw new InvalidArgumentException('maxIterations must be at least 1.');
        }

        $this->maxGenerationsPerRestart = $maxIterations;

        return $this;
    }

    /**
     * Same shape as {@see AbstractOptimizerAlgorithm::extractBounds()} -- duplicated rather than shared
     * since this decorator composes an algorithm rather than extending that abstract base (it tracks no
     * best-value/evaluation-count state of its own to inherit).
     *
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     *
     * @throws \InvalidArgumentException
     *
     * @return array{0: array<int, float>, 1: array<int, float>}
     */
    private function extractBounds(ProblemInterface $problem): array
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
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @throws \InvalidArgumentException When any dimension is infinitely bounded -- same restriction
     *   {@see AbstractOptimizerAlgorithm::randomVectorWithinBounds()} already has, for the same reason.
     *
     * @return array<int, float>
     */
    private function drawRandomVector(array $lowerBounds, array $upperBounds): array
    {
        $vector = [];

        foreach ($lowerBounds as $index => $lowerBound) {
            $upperBound = $upperBounds[$index];

            if (is_infinite($lowerBound) || is_infinite($upperBound)) {
                throw new InvalidArgumentException(
                    'Every dimension must be finitely bounded for RestartingOptimizerDecorator to draw a '
                    . 'fresh restart point -- an infinite bound has no finite range to sample from.',
                );
            }

            $vector[$index] = $lowerBound + $this->randomizer->getFloat(0.0, 1.0) * ($upperBound - $lowerBound);
        }

        return $vector;
    }
}
