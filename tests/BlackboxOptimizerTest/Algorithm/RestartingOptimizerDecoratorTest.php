<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizerTest\Algorithm;

use BlackboxOptimizer\Algorithm\Internal\TerminationReason;
use BlackboxOptimizer\Algorithm\OptimizationResult;
use BlackboxOptimizer\Algorithm\OptimizerAlgorithmInterface;
use BlackboxOptimizer\Algorithm\RestartingOptimizerDecorator;
use BlackboxOptimizer\Problem\CallableProblem;
use BlackboxOptimizer\Problem\ProblemInterface;
use InvalidArgumentException;
use LogicException;
use PHPUnit\Framework\TestCase;

/**
 * Exercises the decorator's own restart/budget/reporting logic in isolation, against a scripted inner
 * {@see OptimizerAlgorithmInterface} rather than a real (stochastic) algorithm -- the decorator's behavior
 * is deterministic GIVEN a sequence of inner results, so pinning that sequence directly is both simpler and
 * more precise than trying to engineer a real objective that plateaus on cue.
 */
class RestartingOptimizerDecoratorTest extends TestCase
{
    /**
     * @return \BlackboxOptimizer\Problem\ProblemInterface
     */
    protected function createProblem(): ProblemInterface
    {
        // phpcs:disable SlevomatCodingStandard.Functions.UnusedParameter -- the closure's signature must
        // match ProblemInterface::evaluate()'s single array parameter; the inner algorithm here is always
        // scripted, so nothing ever actually calls this.
        $constant = static fn (array $vector): float => 0.0;
        // phpcs:enable SlevomatCodingStandard.Functions.UnusedParameter

        return new CallableProblem($constant, [0.0, 0.0], [1.0, 1.0]);
    }

    /**
     * @return void
     */
    public function testImplementsTheGenericOptimizerAlgorithmInterface(): void
    {
        $this->assertInstanceOf(
            OptimizerAlgorithmInterface::class,
            new RestartingOptimizerDecorator(new ScriptedOptimizerAlgorithm([])),
        );
    }

    /**
     * @return void
     */
    public function testOptimizeRequiresPopulationSizeAndMaxIterations(): void
    {
        $decorator = new RestartingOptimizerDecorator(new ScriptedOptimizerAlgorithm([]));

        $this->expectException(InvalidArgumentException::class);

        $decorator->optimize($this->createProblem());
    }

    /**
     * @return void
     */
    public function testEstimateEvaluationCountRequiresPopulationSizeAndMaxIterations(): void
    {
        $decorator = new RestartingOptimizerDecorator(new ScriptedOptimizerAlgorithm([]));

        $this->expectException(InvalidArgumentException::class);

        $decorator->estimateEvaluationCount();
    }

    /**
     * @return void
     */
    public function testEstimateEvaluationCountIsPopulationSizeTimesMaxIterations(): void
    {
        $decorator = (new RestartingOptimizerDecorator(new ScriptedOptimizerAlgorithm([])))
            ->setPopulationSize(10)
            ->setMaxIterations(5);

        $this->assertSame(50, $decorator->estimateEvaluationCount());
    }

    /**
     * @return void
     */
    public function testTrustTerminationCriteriaIsUnsupported(): void
    {
        $decorator = new RestartingOptimizerDecorator(new ScriptedOptimizerAlgorithm([]));

        $this->expectException(LogicException::class);

        $decorator->trustTerminationCriteria();
    }

    /**
     * A run that stops for any reason OTHER than TOL_FUN (here: TOL_X, a converged run) must not restart at
     * all, however much budget remains -- see this class's own docblock for why.
     *
     * @return void
     */
    public function testDoesNotRestartOnNonPlateauTermination(): void
    {
        $inner = new ScriptedOptimizerAlgorithm([
            new OptimizationResult([1.0, 1.0], 5.0, 20, [10.0, 5.0], TerminationReason::TOL_X),
        ]);

        $result = (new RestartingOptimizerDecorator($inner))
            ->setPopulationSize(10)
            ->setMaxIterations(5)
            ->optimize($this->createProblem());

        $this->assertSame(1, $inner->getOptimizeCallCount());
        $this->assertSame(5.0, $result->getBestValue());
        $this->assertSame(20, $result->getEvaluationCount());
        $this->assertCount(1, $result->getRestartHistory());
        $this->assertSame(TerminationReason::TOL_X, $result->getTerminationReason());
    }

    /**
     * A plain run that never even fills the fitness-history window (TerminationReason::NONE, budget fully
     * used) must not restart either -- it isn't a plateau, it's "the run finished".
     *
     * @return void
     */
    public function testDoesNotRestartWhenTheBudgetSimplyRunsOut(): void
    {
        $inner = new ScriptedOptimizerAlgorithm([
            new OptimizationResult([1.0, 1.0], 5.0, 50, array_fill(0, 5, 5.0), TerminationReason::NONE),
        ]);

        $result = (new RestartingOptimizerDecorator($inner))
            ->setPopulationSize(10)
            ->setMaxIterations(5)
            ->optimize($this->createProblem());

        $this->assertSame(1, $inner->getOptimizeCallCount());
        $this->assertCount(1, $result->getRestartHistory());
    }

    /**
     * The core IPOP behavior: a TOL_FUN stop restarts with a DOUBLED population, and stops for good once a
     * restart itself doesn't report TOL_FUN. Also proves the budget-driven generation cap: restart 1's
     * remaining budget (30) only affords 1 generation at its doubled population (20), even though the
     * caller's own maxIterations is 5.
     *
     * @return void
     */
    public function testRestartsOnPlateauWithDoublingPopulationAndBudgetCappedGenerations(): void
    {
        $inner = new ScriptedOptimizerAlgorithm([
            new OptimizationResult([1.0, 1.0], 5.0, 20, array_fill(0, 5, 5.0), TerminationReason::TOL_FUN),
            new OptimizationResult([2.0, 2.0], 3.0, 20, [4.0], TerminationReason::NONE),
        ]);

        $result = (new RestartingOptimizerDecorator($inner))
            ->setPopulationSize(10)
            ->setMaxIterations(5)
            ->optimize($this->createProblem());

        // Restart 0: populationSize=10, full 5-generation allowance (budget=50, 50/10=5).
        // Restart 1: populationSize=20 (doubled), but remaining budget is 50-20=30, and 30/20=1 generation
        // -- capped below the caller's own maxIterations=5, exactly as the budget discipline requires.
        $this->assertSame([10, 20], $inner->populationSizesSeen);
        $this->assertSame([5, 1], $inner->maxIterationsSeen);

        $this->assertSame(2, $inner->getOptimizeCallCount());
        $this->assertSame(40, $result->getEvaluationCount());
        $this->assertSame(3.0, $result->getBestValue());
        $this->assertSame([2.0, 2.0], $result->getBestVector());
        $this->assertSame(TerminationReason::NONE, $result->getTerminationReason());

        // bestValueHistory is the concatenation across every restart, in order -- 5 entries from restart 0
        // plus 1 from restart 1.
        $this->assertSame([5.0, 5.0, 5.0, 5.0, 5.0, 4.0], $result->getBestValueHistory());

        $restartHistory = $result->getRestartHistory();
        $this->assertCount(2, $restartHistory);

        $this->assertSame(0, $restartHistory[0]->getRestartIndex());
        $this->assertSame(10, $restartHistory[0]->getPopulationSize());
        $this->assertSame(5, $restartHistory[0]->getGenerationsAllowed());
        $this->assertSame(5, $restartHistory[0]->getGenerationsUsed());
        $this->assertSame(TerminationReason::TOL_FUN, $restartHistory[0]->getTerminationReason());
        $this->assertTrue($restartHistory[0]->improvedOverallBest());

        $this->assertSame(1, $restartHistory[1]->getRestartIndex());
        $this->assertSame(20, $restartHistory[1]->getPopulationSize());
        $this->assertSame(1, $restartHistory[1]->getGenerationsAllowed());
        $this->assertSame(TerminationReason::NONE, $restartHistory[1]->getTerminationReason());
        $this->assertTrue($restartHistory[1]->improvedOverallBest());
    }

    /**
     * A later restart's own best value can be WORSE than an earlier one's -- the final result must still
     * report the best EVER found, not just the last restart's, and {@see RestartHistoryEntry::improvedOverallBest()}
     * must say so.
     *
     * @return void
     */
    public function testKeepsTheBestVectorAcrossRestartsEvenIfALaterRestartIsWorse(): void
    {
        $inner = new ScriptedOptimizerAlgorithm([
            new OptimizationResult([1.0, 1.0], 5.0, 20, [5.0], TerminationReason::TOL_FUN),
            new OptimizationResult([9.0, 9.0], 7.0, 20, [7.0], TerminationReason::NONE),
        ]);

        $result = (new RestartingOptimizerDecorator($inner))
            ->setPopulationSize(10)
            ->setMaxIterations(5)
            ->optimize($this->createProblem());

        $this->assertSame(5.0, $result->getBestValue());
        $this->assertSame([1.0, 1.0], $result->getBestVector());
        $this->assertFalse($result->getRestartHistory()[1]->improvedOverallBest());
    }

    /**
     * Budget discipline's own hard edge: a restart that WOULD plateau-restart again must stop instead once
     * the next (doubled) population can no longer be afforded at all, even at just 1 generation.
     *
     * @return void
     */
    public function testStopsRestartingWhenTheNextDoubledPopulationCannotBeAffordedAtAll(): void
    {
        $inner = new ScriptedOptimizerAlgorithm([
            // Consumes 45 of the 50-evaluation total budget; only 5 remain, not enough for populationSize=20.
            new OptimizationResult([1.0, 1.0], 5.0, 45, [5.0], TerminationReason::TOL_FUN),
        ]);

        $result = (new RestartingOptimizerDecorator($inner))
            ->setPopulationSize(10)
            ->setMaxIterations(5)
            ->optimize($this->createProblem());

        $this->assertSame(1, $inner->getOptimizeCallCount());
        $this->assertCount(1, $result->getRestartHistory());
        $this->assertSame(45, $result->getEvaluationCount());
    }

    /**
     * @return void
     */
    public function testDrawsAFreshWarmStartBeforeEveryRestartAfterTheFirstButNotBeforeTheFirst(): void
    {
        $inner = new ScriptedOptimizerAlgorithm([
            new OptimizationResult([1.0, 1.0], 5.0, 10, [5.0], TerminationReason::TOL_FUN),
            new OptimizationResult([1.0, 1.0], 5.0, 10, [5.0], TerminationReason::NONE),
        ]);

        (new RestartingOptimizerDecorator($inner))
            ->setPopulationSize(10)
            ->setMaxIterations(5)
            ->optimize($this->createProblem());

        $this->assertCount(1, $inner->warmStartCalls, 'Only restart 1 should draw a fresh warm start, not restart 0.');
        [$vector, $fraction] = $inner->warmStartCalls[0];
        $this->assertSame(1.0, $fraction);
        $this->assertCount(2, $vector);

        foreach ($vector as $value) {
            $this->assertGreaterThanOrEqual(0.0, $value);
            $this->assertLessThanOrEqual(1.0, $value);
        }
    }

    /**
     * @return void
     */
    public function testGetSafetyIterationCeilingForwardsToTheInnerAlgorithm(): void
    {
        $decorator = new RestartingOptimizerDecorator(new ScriptedOptimizerAlgorithm([]));

        $this->assertSame(10000, $decorator->getSafetyIterationCeiling());
    }

    /**
     * @return void
     */
    public function testOptimizeAcceptsTrustRestartBudgetInPlaceOfMaxIterations(): void
    {
        $inner = new ScriptedOptimizerAlgorithm([
            new OptimizationResult([1.0, 1.0], 5.0, 20, [5.0], TerminationReason::TOL_X),
        ]);

        // No setMaxIterations() call at all -- trustRestartBudget() alone must be enough.
        $result = (new RestartingOptimizerDecorator($inner))
            ->setPopulationSize(10)
            ->trustRestartBudget()
            ->optimize($this->createProblem());

        $this->assertSame(5.0, $result->getBestValue());
    }

    /**
     * @return void
     */
    public function testEstimateEvaluationCountUsesTheSafetyIterationCeilingUnderTrustRestartBudget(): void
    {
        $decorator = (new RestartingOptimizerDecorator(new ScriptedOptimizerAlgorithm([])))
            ->setPopulationSize(10)
            ->trustRestartBudget();

        // ScriptedOptimizerAlgorithm::getSafetyIterationCeiling() returns 10000 -- see its own definition.
        $this->assertSame(100000, $decorator->estimateEvaluationCount());
    }

    /**
     * The central claim {@see RestartingOptimizerDecorator::trustRestartBudget()}'s own docblock makes:
     * restart 0's own ceiling is the FULL safety-iteration-ceiling number, not a caller-chosen (typically
     * much smaller) maxIterations -- a plain, separate "raise the per-restart ceiling" constant could never
     * achieve this under the default budget, since restart 0's own remainingBudget/populationSize is
     * EXACTLY maxIterations by construction there (see that method's own docblock for the full argument).
     *
     * @return void
     */
    public function testTrustRestartBudgetGivesTheFirstRestartTheFullSafetyCeilingAsItsOwnMaxIterations(): void
    {
        $inner = new ScriptedOptimizerAlgorithm([
            new OptimizationResult([1.0, 1.0], 5.0, 20, [5.0], TerminationReason::TOL_X),
        ]);

        (new RestartingOptimizerDecorator($inner))
            ->setPopulationSize(10)
            ->trustRestartBudget()
            ->optimize($this->createProblem());

        $this->assertSame([10000], $inner->maxIterationsSeen);
    }

    /**
     * A `setMaxIterations()` call before `trustRestartBudget()` must not leak into the budget calculation --
     * mirroring how a plain algorithm's own `trustTerminationCriteria()` already ignores its own
     * `setMaxIterations()`.
     *
     * @return void
     */
    public function testTrustRestartBudgetIgnoresAPreviouslySetMaxIterations(): void
    {
        $decorator = (new RestartingOptimizerDecorator(new ScriptedOptimizerAlgorithm([])))
            ->setPopulationSize(10)
            ->setMaxIterations(5)
            ->trustRestartBudget();

        $this->assertSame(100000, $decorator->estimateEvaluationCount());
    }

    /**
     * @return void
     */
    public function testOptimizeThrowsWhenNeitherMaxIterationsNorTrustRestartBudgetWasSet(): void
    {
        $decorator = (new RestartingOptimizerDecorator(new ScriptedOptimizerAlgorithm([])))
            ->setPopulationSize(10);

        $this->expectException(InvalidArgumentException::class);

        $decorator->optimize($this->createProblem());
    }
}
