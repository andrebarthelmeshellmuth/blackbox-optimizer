<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizerTest\Algorithm;

use BlackboxOptimizer\Algorithm\OptimizerAlgorithmInterface;
use BlackboxOptimizer\Algorithm\RechenbergSchwefelEsAlgorithm;
use BlackboxOptimizer\Problem\CallableProblem;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use ReflectionMethod;

/**
 * Validates against known toy benchmark functions (sphere, Rosenbrock) with known optima, per this
 * package's own decision to prove any optimizer implementation correct BEFORE ever pointing it at a real
 * (and much more expensive to debug) objective function.
 */
class RechenbergSchwefelEsAlgorithmTest extends TestCase
{
    /**
     * @return void
     */
    public function testImplementsTheGenericOptimizerAlgorithmInterface(): void
    {
        $this->assertInstanceOf(OptimizerAlgorithmInterface::class, new RechenbergSchwefelEsAlgorithm());
    }

    /**
     * @return void
     */
    public function testExposesFactualNameAndDescriptionMetadata(): void
    {
        $algorithm = new RechenbergSchwefelEsAlgorithm();

        $this->assertSame('Rechenberg/Schwefel ES', $algorithm->getName());
        $this->assertNotSame('', $algorithm->getDescription());
    }

    /**
     * Reuses {@see \BlackboxOptimizer\Algorithm\RechenbergSchwefelEsAlgorithm::resolveParentCount()}
     * directly, so this is really a test that estimateEvaluationCount() and optimize() are computing the
     * initial batch size the exact same way, not two independent copies of the mu/lambda ~ 1/7 formula.
     *
     * @return void
     */
    public function testEstimateEvaluationCountMatchesParentCountPlusOffspringCountTimesIterations(): void
    {
        // Arrange
        $algorithm = (new RechenbergSchwefelEsAlgorithm())->setPopulationSize(20)->setMaxIterations(200);

        // Act
        $estimate = $algorithm->estimateEvaluationCount();

        // Assert -- parentCount = round(20 / 7) = 3.
        $this->assertSame(3 + (20 * 200), $estimate);
    }

    /**
     * The n-dimensional sphere function f(x) = sum(x_i^2) has a single global minimum of 0 at the origin
     * -- the simplest possible convex benchmark, good for a basic sanity check.
     *
     * @return void
     */
    public function testOptimizeFindsTheKnownMinimumOfTheSphereFunction(): void
    {
        // Arrange
        $sphere = static function (array $vector): float {
            $sum = 0.0;

            foreach ($vector as $component) {
                $sum += $component ** 2;
            }

            return $sum;
        };

        $problem = new CallableProblem($sphere, [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);

        $algorithm = new RechenbergSchwefelEsAlgorithm();
        $algorithm->setPopulationSize(20)->setMaxIterations(200);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThan(1e-3, $result->getBestValue(), 'The ES should get very close to the sphere function\'s known minimum of 0.');

        foreach ($result->getBestVector() as $component) {
            $this->assertEqualsWithDelta(0.0, $component, 0.2, 'Each dimension should converge close to the known optimum at the origin.');
        }
    }

    /**
     * estimateEvaluationCount() is an upper bound -- this algorithm's own early termination (see
     * {@see optimize()}) can make a real run stop before spending its full maxIterations budget. This is
     * the guard against the two ever silently drifting apart in the other direction (a real run spending
     * MORE than predicted).
     *
     * @return void
     */
    public function testEstimateEvaluationCountIsNeverLessThanARealRunsActualEvaluationCount(): void
    {
        // Arrange
        $sphere = static function (array $vector): float {
            $sum = 0.0;

            foreach ($vector as $component) {
                $sum += $component ** 2;
            }

            return $sum;
        };

        $problem = new CallableProblem($sphere, [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);

        $algorithm = new RechenbergSchwefelEsAlgorithm();
        $algorithm->setPopulationSize(20)->setMaxIterations(200);

        $estimate = $algorithm->estimateEvaluationCount();

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThanOrEqual($estimate, $result->getEvaluationCount());
    }

    /**
     * The 2D Rosenbrock "banana" function f(x,y) = (a-x)^2 + b(y-x^2)^2 (a=1, b=100) has a known global
     * minimum of 0 at (1, 1) -- a classic non-convex benchmark with a narrow, curved valley. A much harder
     * test for an isotropic-mutation algorithm with no covariance adaptation than for CmaEsAlgorithm, since
     * a single scalar sigma can't learn the valley's own shape -- loose tolerances here are deliberate.
     *
     * @return void
     */
    public function testOptimizeFindsTheKnownMinimumOfTheRosenbrockFunction(): void
    {
        // Arrange
        $rosenbrock = static function (array $vector): float {
            [$x, $y] = $vector;

            return (1 - $x) ** 2 + 100 * ($y - $x ** 2) ** 2;
        };

        $problem = new CallableProblem($rosenbrock, [-3.0, -3.0], [3.0, 3.0]);

        $algorithm = new RechenbergSchwefelEsAlgorithm();
        $algorithm->setPopulationSize(30)->setMaxIterations(800);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThan(0.5, $result->getBestValue(), 'The ES should get reasonably close to the Rosenbrock function\'s known minimum of 0.');
        $this->assertEqualsWithDelta(1.0, $result->getBestVector()[0], 0.5, 'x should converge reasonably close to the known optimum at (1, 1).');
        $this->assertEqualsWithDelta(1.0, $result->getBestVector()[1], 0.5, 'y should converge reasonably close to the known optimum at (1, 1).');
    }

    /**
     * @return void
     */
    public function testOptimizeReportsAnEvaluationCountAndAMonotonicallyImprovingHistory(): void
    {
        // Arrange
        $sphere = static fn (array $vector): float => array_sum(array_map(fn (float $value): float => $value ** 2, $vector));
        $problem = new CallableProblem($sphere, [-5.0], [5.0]);

        $algorithm = new RechenbergSchwefelEsAlgorithm();
        $algorithm->setPopulationSize(10)->setParentCount(2)->setMaxIterations(5);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert -- 2 initial parent evaluations + 5 generations * 10 offspring each
        $this->assertSame(52, $result->getEvaluationCount());
        $this->assertCount(6, $result->getBestValueHistory(), 'One history entry for the initial parents plus one per generation.');

        $history = $result->getBestValueHistory();
        $historyCount = count($history);

        for ($i = 1; $i < $historyCount; $i++) {
            $this->assertLessThanOrEqual($history[$i - 1], $history[$i], 'Plus-selection guarantees the best-found value never gets worse from one generation to the next.');
        }
    }

    /**
     * Explicit parentCount below the default 1/7-of-lambda ratio, to prove {@see RechenbergSchwefelEsAlgorithm::setParentCount()}
     * actually takes effect rather than the default silently winning.
     *
     * @return void
     */
    public function testSetParentCountIsHonoredOverTheDefaultRatio(): void
    {
        // Arrange
        $sphere = static fn (array $vector): float => array_sum(array_map(fn (float $value): float => $value ** 2, $vector));
        $problem = new CallableProblem($sphere, [-5.0], [5.0]);

        $algorithm = new RechenbergSchwefelEsAlgorithm();
        $algorithm->setPopulationSize(10)->setParentCount(1)->setMaxIterations(3);

        // Act -- 1 initial parent + 3 generations * 10 offspring
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertSame(31, $result->getEvaluationCount());
    }

    /**
     * @return void
     */
    public function testSetParentCountRejectsLessThanOne(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new RechenbergSchwefelEsAlgorithm())->setParentCount(0);
    }

    /**
     * @return void
     */
    public function testOptimizeThrowsWhenParentCountExceedsThePopulationSize(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $problem = new CallableProblem(static fn (array $vector): float => $vector[0] ** 2, [-5.0], [5.0]);
        (new RechenbergSchwefelEsAlgorithm())->setPopulationSize(4)->setParentCount(5)->optimize($problem);
    }

    /**
     * @return void
     */
    public function testSetPopulationSizeRejectsATooSmallPopulation(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new RechenbergSchwefelEsAlgorithm())->setPopulationSize(3);
    }

    /**
     * Like {@see \BlackboxOptimizer\Algorithm\DifferentialEvolutionAlgorithm}, the initial parent
     * population is drawn uniformly from the bounds themselves, undefined for an infinite bound.
     *
     * @return void
     */
    public function testOptimizeThrowsWhenABoundIsInfinite(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $problem = new CallableProblem(static fn (array $vector): float => $vector[0] ** 2, [-INF], [INF]);
        (new RechenbergSchwefelEsAlgorithm())->optimize($problem);
    }

    /**
     * Mirrors {@see \BlackboxOptimizerTest\Algorithm\CmaEsAlgorithmTest::testTrustTerminationCriteriaOverridesATooSmallSetMaxIterations()} --
     * a deliberately tiny setMaxIterations() is ignored once trustTerminationCriteria() is on.
     *
     * @return void
     */
    public function testTrustTerminationCriteriaOverridesATooSmallSetMaxIterations(): void
    {
        // Arrange
        $sphere = static fn (array $vector): float => $vector[0] ** 2 + $vector[1] ** 2;
        $problem = new CallableProblem($sphere, [-5.0, -5.0], [5.0, 5.0]);

        $algorithm = new RechenbergSchwefelEsAlgorithm();
        $algorithm->setPopulationSize(20)->setMaxIterations(3)->trustTerminationCriteria();

        // Act
        $result = $algorithm->optimize($problem);

        // Assert -- history includes the initial-parents entry, so this is generations-run + 1.
        $generationsRun = count($result->getBestValueHistory());
        $this->assertGreaterThan(4, $generationsRun, 'The 3-generation cap from setMaxIterations() must be ignored once trustTerminationCriteria() is on.');
        $this->assertLessThan(10000, $generationsRun, 'Should stop via the sigma-collapse criterion well before the safety ceiling, not by exhausting it.');
        $this->assertLessThan(1e-4, $result->getBestValue(), 'Given the room to actually converge, the known minimum should be reached closely.');
    }

    /**
     * This class reuses {@see \BlackboxOptimizer\Algorithm\Internal\TerminationCriteria} exactly as-is,
     * the same shared TolX/TolXUp/ConditionCov/TolFun class {@see \BlackboxOptimizer\Algorithm\CmaEsAlgorithm}
     * uses -- but the test above only ever exercises the TolX branch (sigma collapsing as the sphere
     * converges). TolFun has never fired through THIS class's own wiring, only proven to work in the
     * abstract via {@see \BlackboxOptimizerTest\Algorithm\Internal\TerminationCriteriaTest} and concretely
     * via CmaEsAlgorithmTest's own equivalent test. A perfectly constant objective isolates it here too:
     * with plus-selection, an offspring never scores strictly better than a tied parent, so the reported
     * best value is 0.0 from the very first generation regardless of how sigma/the population itself moves
     * -- hitting the flat-fitness window as soon as it fills, well before sigma could plausibly collapse
     * 11 orders of magnitude in the same handful of generations.
     *
     * @return void
     */
    public function testOptimizeStopsEarlyViaTolFunOnAConstantObjective(): void
    {
        // Arrange
        // phpcs:disable SlevomatCodingStandard.Functions.UnusedParameter -- the closure's signature must
        // match ProblemInterface::evaluate()'s single array parameter; a constant objective never reads it.
        $constant = static fn (array $vector): float => 0.0;
        // phpcs:enable SlevomatCodingStandard.Functions.UnusedParameter
        $problem = new CallableProblem($constant, [-5.0], [5.0]);

        $algorithm = new RechenbergSchwefelEsAlgorithm();
        $algorithm->setPopulationSize(8)->setMaxIterations(500);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert -- fitness history length for n=1, lambda=8 is 10 + ceil(30*1/8) = 14 (same formula
        // TerminationCriteria::resolveFitnessHistoryLength() uses), plus the initial-parents entry
        // recorded before the generation loop even starts (see
        // testTrustTerminationCriteriaOverridesATooSmallSetMaxIterations() above for the same +1).
        $this->assertLessThanOrEqual(15, count($result->getBestValueHistory()), 'A perfectly flat objective should trigger TolFun as soon as the fitness-history window fills.');
    }

    /**
     * @return void
     */
    public function testSetWarmStartRejectsAFractionBelowZero(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new RechenbergSchwefelEsAlgorithm())->setWarmStart([0.0], -0.1);
    }

    /**
     * @return void
     */
    public function testSetWarmStartRejectsAFractionAboveOne(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new RechenbergSchwefelEsAlgorithm())->setWarmStart([0.0], 1.1);
    }

    /**
     * Mirrors {@see \BlackboxOptimizerTest\Algorithm\DifferentialEvolutionAlgorithmTest::testOptimizeStillThrowsWhenABoundIsInfiniteAndWarmStartFractionIsZero()} --
     * fraction=0.0 seeds zero parents near the vector, so every initial parent still goes through the same
     * uniform-within-bounds draw {@see testOptimizeThrowsWhenABoundIsInfinite()} already proves is undefined
     * for an infinite bound.
     *
     * @return void
     */
    public function testOptimizeStillThrowsWhenABoundIsInfiniteAndWarmStartFractionIsZero(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $problem = new CallableProblem(static fn (array $vector): float => $vector[0] ** 2, [-INF], [INF]);
        (new RechenbergSchwefelEsAlgorithm())->setWarmStart([3.0], 0.0)->optimize($problem);
    }

    /**
     * Mirrors {@see \BlackboxOptimizerTest\Algorithm\DifferentialEvolutionAlgorithmTest::testOptimizeFullyWarmStartsTheInitialPopulationWhenFractionIsOne()} --
     * fraction=1.0 seeds every initial parent near the warm-start vector (jittered by a tight step width /
     * sigma). getBestValueHistory()'s first entry is recorded right after the initial parents, before any
     * generation runs, isolating what the initial parent population alone found.
     *
     * @return void
     */
    public function testOptimizeFullyWarmStartsTheInitialParentsWhenFractionIsOne(): void
    {
        // Arrange -- a tight jitter (stepWidth/sigma) keeps every seeded parent very close to (10, 10).
        $shiftedSphere = static function (array $vector): float {
            return ($vector[0] - 10) ** 2 + ($vector[1] - 10) ** 2;
        };

        $problem = new CallableProblem($shiftedSphere, [-100.0, -100.0], [100.0, 100.0]);

        $algorithm = new RechenbergSchwefelEsAlgorithm();
        $algorithm->setPopulationSize(20)->setStepWidth(0.05)->setWarmStart([10.0, 10.0], 1.0)->setMaxIterations(1);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $history = $result->getBestValueHistory();
        $this->assertLessThan(0.5, $history[0], 'A fully warm-started, tightly jittered initial parent population should already be close to the known optimum, before any generation runs.');
    }

    /**
     * Mirrors {@see \BlackboxOptimizerTest\Algorithm\DifferentialEvolutionAlgorithmTest::testSeedInitialPopulationSplitsExactlyByWarmStartFractionAtAPartialFraction()}
     * -- see that test's own docblock for why an outcome-based optimize() run can't isolate the split
     * arithmetic cleanly and a direct call to the shared, inherited
     * {@see \BlackboxOptimizer\Algorithm\AbstractOptimizerAlgorithm::seedInitialPopulation()} is needed
     * instead. Written once per concrete class (matching this suite's own existing convention for the
     * validation tests above, which cover identical shared behavior too) rather than once globally, even
     * though the method body itself is inherited unchanged.
     *
     * @return void
     */
    public function testSeedInitialPopulationSplitsExactlyByWarmStartFractionAtAPartialFraction(): void
    {
        // Arrange -- bounds are huge relative to the jitter scale, so a member seeded via the random path
        // has a negligible chance of accidentally landing within the warm-detection radius below.
        $algorithm = new RechenbergSchwefelEsAlgorithm();
        $algorithm->setWarmStart([10.0, 10.0], 0.3);

        $seedInitialPopulation = new ReflectionMethod($algorithm, 'seedInitialPopulation');
        $seedInitialPopulation->setAccessible(true);

        // Act -- round(20 * 0.3) = 6 members expected near the vector, jitterScale=0.01 keeps them tight.
        $population = $seedInitialPopulation->invoke($algorithm, 20, [-1000.0, -1000.0], [1000.0, 1000.0], 0.01);

        // Assert
        $warmCount = 0;

        foreach ($population as $vector) {
            $distanceFromWarmVector = sqrt(($vector[0] - 10.0) ** 2 + ($vector[1] - 10.0) ** 2);

            if ($distanceFromWarmVector < 1.0) {
                $warmCount++;
            }
        }

        $this->assertSame(6, $warmCount, 'fraction=0.3 of a population of 20 should seed exactly round(20 * 0.3) = 6 members near the warm-start vector, not merely "some".');
    }
}
