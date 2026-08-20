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
 * {@see setMaxIterations()} call already means today. A restart's own population size doubles each time
 * (classic IPOP), which is what actually bounds the restart COUNT: growth is geometric, so even a modest
 * budget only buys a handful of restarts before the next doubled population can no longer be afforded, at
 * which point this decorator stops and returns the best candidate found across every restart -- never more
 * evaluations than the original budget promised, and never fewer restarts than that budget can actually pay
 * for.
 *
 * Deliberately does NOT support {@see trustTerminationCriteria()} -- that mode replaces a caller's own
 * `maxIterations` with each algorithm's internal safety ceiling (thousands of generations), which would
 * silently blow through this decorator's own budget arithmetic on every restart. This decorator's own
 * restart-on-plateau mechanism already IS "trust the termination criteria, but bounded" — see that method's
 * override here for the details.
 */
final class RestartingOptimizerDecorator implements OptimizerAlgorithmInterface
{
    private ?int $initialPopulationSize = null;

    private ?int $maxGenerationsPerRestart = null;

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
     * @throws \InvalidArgumentException When {@see setPopulationSize()}/{@see setMaxIterations()} were never
     *   called -- unlike the inner algorithms, this decorator has no fallback default: the restart budget IS
     *   `populationSize * maxIterations`, so both are required to know what that budget even is.
     *
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    public function optimize(ProblemInterface $problem): OptimizationResult
    {
        if ($this->initialPopulationSize === null || $this->maxGenerationsPerRestart === null) {
            throw new InvalidArgumentException(
                'RestartingOptimizerDecorator requires both setPopulationSize() and setMaxIterations() to be '
                . 'called -- the total restart budget is populationSize * maxIterations, with no fallback '
                . 'default the way a single inner algorithm run might have.',
            );
        }

        [$lowerBounds, $upperBounds] = $this->extractBounds($problem);

        $totalBudget = $this->initialPopulationSize * $this->maxGenerationsPerRestart;
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

            $generationsAllowed = intdiv($remainingBudget, $currentPopulationSize);

            if ($generationsAllowed < 1) {
                break;
            }

            $this->innerAlgorithm->setPopulationSize($currentPopulationSize);
            $this->innerAlgorithm->setMaxIterations(min($this->maxGenerationsPerRestart, $generationsAllowed));

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
                min($this->maxGenerationsPerRestart, $generationsAllowed),
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
     * restarts actually happen within it.
     *
     * @throws \InvalidArgumentException When {@see setPopulationSize()}/{@see setMaxIterations()} were never
     *   called.
     *
     * @return int
     */
    public function estimateEvaluationCount(): int
    {
        if ($this->initialPopulationSize === null || $this->maxGenerationsPerRestart === null) {
            throw new InvalidArgumentException(
                'RestartingOptimizerDecorator requires both setPopulationSize() and setMaxIterations() to be '
                . 'called before estimateEvaluationCount() -- see optimize()\'s own exception for why.',
            );
        }

        return $this->initialPopulationSize * $this->maxGenerationsPerRestart;
    }

    /**
     * {@inheritDoc}
     *
     * Deliberately unsupported -- see this class's own docblock for why trusting each algorithm's internal
     * safety ceiling instead of a fixed `maxIterations` would blow through this decorator's own budget
     * arithmetic. This decorator's restart-on-plateau mechanism is already this package's answer to the same
     * underlying need ("don't stop the search just because one fixed generation count ran out"), scoped to
     * a budget the caller actually chose.
     *
     * @throws \LogicException Always.
     *
     * @return never
     */
    public function trustTerminationCriteria(): static
    {
        throw new LogicException(
            'RestartingOptimizerDecorator does not support trustTerminationCriteria() -- it would let each '
            . 'restart run out to the inner algorithm\'s own safety ceiling instead of the budget this '
            . 'decorator\'s setPopulationSize()/setMaxIterations() define. Call setMaxIterations() with the '
            . 'generation budget you actually want instead.',
        );
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
     * The per-restart generation cap -- together with {@see setPopulationSize()}, defines the total restart
     * budget (`populationSize * maxIterations`). A later restart may be capped BELOW this, when the
     * remaining budget can't afford a full allowance at its (larger, doubled) population size -- see
     * {@see RestartHistoryEntry::getGenerationsAllowed()}.
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
