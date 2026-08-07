<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizerTest\Algorithm;

use BlackboxOptimizer\Algorithm\DifferentialEvolutionAlgorithm;
use BlackboxOptimizer\Algorithm\OptimizerAlgorithmInterface;
use BlackboxOptimizer\Problem\CallableProblem;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;

/**
 * Validates against known toy benchmark functions (sphere, Rosenbrock) with known optima, per this
 * package's own decision to prove any optimizer implementation correct BEFORE ever pointing it at a real
 * (and much more expensive to debug) objective function.
 */
class DifferentialEvolutionAlgorithmTest extends TestCase
{
    /**
     * @return void
     */
    public function testImplementsTheGenericOptimizerAlgorithmInterface(): void
    {
        $this->assertInstanceOf(OptimizerAlgorithmInterface::class, new DifferentialEvolutionAlgorithm());
    }

    /**
     * @return void
     */
    public function testExposesFactualNameAndDescriptionMetadata(): void
    {
        $algorithm = new DifferentialEvolutionAlgorithm();

        $this->assertSame('Differential Evolution', $algorithm->getName());
        $this->assertNotSame('', $algorithm->getDescription());
    }

    /**
     * Unlike CMA-ES, DE has a fixed default population size, so estimateEvaluationCount() works even
     * without an explicit setPopulationSize() call.
     *
     * @return void
     */
    public function testEstimateEvaluationCountFallsBackToTheDefaultPopulationSize(): void
    {
        // Arrange
        $algorithm = (new DifferentialEvolutionAlgorithm())->setMaxIterations(10);

        // Act
        $estimate = $algorithm->estimateEvaluationCount();

        // Assert -- DE evaluates one extra initial-population batch before its generation loop starts.
        $this->assertSame(20 * (10 + 1), $estimate);
    }

    /**
     * @return void
     */
    public function testEstimateEvaluationCountMatchesPopulationSizeTimesIterationsPlusOne(): void
    {
        // Arrange
        $algorithm = (new DifferentialEvolutionAlgorithm())->setPopulationSize(30)->setMaxIterations(150);

        // Act
        $estimate = $algorithm->estimateEvaluationCount();

        // Assert
        $this->assertSame(30 * 151, $estimate);
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

        $algorithm = new DifferentialEvolutionAlgorithm();
        $algorithm->setPopulationSize(30)->setMaxIterations(150);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThan(1e-4, $result->getBestValue(), 'DE should get very close to the sphere function\'s known minimum of 0.');

        foreach ($result->getBestVector() as $component) {
            $this->assertEqualsWithDelta(0.0, $component, 0.1, 'Each dimension should converge close to the known optimum at the origin.');
        }
    }

    /**
     * estimateEvaluationCount() is an upper bound -- DE's own early termination (population collapse or a
     * flat fitness history, see {@see optimize()}) can make a real run stop before spending its full
     * maxIterations budget. This is the guard against the two ever silently drifting apart in the other
     * direction (a real run spending MORE than predicted).
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

        $algorithm = new DifferentialEvolutionAlgorithm();
        $algorithm->setPopulationSize(30)->setMaxIterations(150);

        $estimate = $algorithm->estimateEvaluationCount();

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThanOrEqual($estimate, $result->getEvaluationCount());
    }

    /**
     * The 2D Rosenbrock "banana" function f(x,y) = (a-x)^2 + b(y-x^2)^2 (a=1, b=100) has a known global
     * minimum of 0 at (1, 1) -- a classic non-convex benchmark with a narrow, curved valley that's much
     * harder to navigate than the sphere function, a meaningfully stronger correctness check.
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

        $algorithm = new DifferentialEvolutionAlgorithm();
        $algorithm->setPopulationSize(40)->setMaxIterations(500);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThan(0.05, $result->getBestValue(), 'DE should get close to the Rosenbrock function\'s known minimum of 0.');
        $this->assertEqualsWithDelta(1.0, $result->getBestVector()[0], 0.2, 'x should converge close to the known optimum at (1, 1).');
        $this->assertEqualsWithDelta(1.0, $result->getBestVector()[1], 0.2, 'y should converge close to the known optimum at (1, 1).');
    }

    /**
     * @return void
     */
    public function testOptimizeReportsAnEvaluationCountAndAMonotonicallyImprovingHistory(): void
    {
        // Arrange
        $sphere = static fn (array $vector): float => array_sum(array_map(fn (float $value): float => $value ** 2, $vector));
        $problem = new CallableProblem($sphere, [-5.0], [5.0]);

        $algorithm = new DifferentialEvolutionAlgorithm();
        $algorithm->setPopulationSize(10)->setMaxIterations(5);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert -- 10 initial evaluations + 5 generations * 10 candidates each
        $this->assertSame(60, $result->getEvaluationCount());
        $this->assertCount(6, $result->getBestValueHistory(), 'One history entry for the initial population plus one per generation.');

        $history = $result->getBestValueHistory();
        $historyCount = count($history);

        for ($i = 1; $i < $historyCount; $i++) {
            $this->assertLessThanOrEqual($history[$i - 1], $history[$i], 'The best-found value must never get worse from one generation to the next.');
        }
    }

    /**
     * @return void
     */
    public function testSetPopulationSizeRejectsATooSmallPopulation(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new DifferentialEvolutionAlgorithm())->setPopulationSize(3);
    }

    /**
     * @return void
     */
    public function testSetCrossoverProbabilityRejectsAnOutOfRangeValue(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new DifferentialEvolutionAlgorithm())->setCrossoverProbability(1.5);
    }

    /**
     * Unlike {@see \BlackboxOptimizer\Algorithm\CmaEsAlgorithm}, DE has no initialMean-style escape hatch --
     * its initial population is always drawn uniformly from the bounds themselves, which is undefined for
     * an infinite bound. Must fail loudly rather than silently produce INF/NAN candidates that never
     * improve (a real bug this test guards against: the initial population generator used to build such a
     * vector with no validation at all).
     *
     * @return void
     */
    public function testOptimizeThrowsWhenABoundIsInfinite(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $problem = new CallableProblem(static fn (array $vector): float => $vector[0] ** 2, [-INF], [INF]);
        (new DifferentialEvolutionAlgorithm())->optimize($problem);
    }

    /**
     * Mirrors {@see \BlackboxOptimizerTest\Algorithm\CmaEsAlgorithmTest::testTrustTerminationCriteriaOverridesATooSmallSetMaxIterations()} --
     * a deliberately tiny setMaxIterations() is ignored once trustTerminationCriteria() is on. Uses DE's
     * own population-collapse criterion (no sigma/eigenvalues to check, unlike the other two algorithms --
     * see this class's own docblock).
     *
     * @return void
     */
    public function testTrustTerminationCriteriaOverridesATooSmallSetMaxIterations(): void
    {
        // Arrange
        $sphere = static fn (array $vector): float => $vector[0] ** 2 + $vector[1] ** 2;
        $problem = new CallableProblem($sphere, [-5.0, -5.0], [5.0, 5.0]);

        $algorithm = new DifferentialEvolutionAlgorithm();
        $algorithm->setPopulationSize(20)->setMaxIterations(3)->trustTerminationCriteria();

        // Act
        $result = $algorithm->optimize($problem);

        // Assert -- history includes the initial-population entry, so this is generations-run + 1.
        $generationsRun = count($result->getBestValueHistory());
        $this->assertGreaterThan(4, $generationsRun, 'The 3-generation cap from setMaxIterations() must be ignored once trustTerminationCriteria() is on.');
        $this->assertLessThan(10000, $generationsRun, 'Should stop via population collapse well before the safety ceiling, not by exhausting it.');
        $this->assertLessThan(1e-4, $result->getBestValue(), 'Given the room to actually converge, the known minimum should be reached closely.');
    }

    /**
     * @return void
     */
    public function testSetWarmStartRejectsAFractionBelowZero(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new DifferentialEvolutionAlgorithm())->setWarmStart([0.0], -0.1);
    }

    /**
     * @return void
     */
    public function testSetWarmStartRejectsAFractionAboveOne(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new DifferentialEvolutionAlgorithm())->setWarmStart([0.0], 1.1);
    }

    /**
     * fraction=0.0 must be bit-identical to never calling setWarmStart() at all -- zero population members
     * get seeded near the vector, so every one of them still goes through the same uniform-within-bounds
     * draw {@see testOptimizeThrowsWhenABoundIsInfinite()} already proves is undefined for an infinite
     * bound. If even one member had been warm-seeded instead, this would not throw.
     *
     * @return void
     */
    public function testOptimizeStillThrowsWhenABoundIsInfiniteAndWarmStartFractionIsZero(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $problem = new CallableProblem(static fn (array $vector): float => $vector[0] ** 2, [-INF], [INF]);
        (new DifferentialEvolutionAlgorithm())->setWarmStart([3.0], 0.0)->optimize($problem);
    }

    /**
     * fraction=1.0 seeds the entire initial population near the warm-start vector (jittered by a tight
     * step width). {@see \BlackboxOptimizer\Algorithm\OptimizationResult::getBestValueHistory()}'s first
     * entry is recorded right after the initial population, before any generation runs -- so this isolates
     * what the initial population alone found, independent of maxIterations. Placing the known optimum
     * exactly at the warm-start vector, far from where uniform-random sampling across the full bounds would
     * typically land, is the only way this could get so close with only the initial population evaluated.
     *
     * @return void
     */
    public function testOptimizeFullyWarmStartsTheInitialPopulationWhenFractionIsOne(): void
    {
        // Arrange -- a tight jitter (stepWidth) keeps every seeded member very close to (10, 10).
        $shiftedSphere = static function (array $vector): float {
            return ($vector[0] - 10) ** 2 + ($vector[1] - 10) ** 2;
        };

        $problem = new CallableProblem($shiftedSphere, [-100.0, -100.0], [100.0, 100.0]);

        $algorithm = new DifferentialEvolutionAlgorithm();
        $algorithm->setPopulationSize(20)->setStepWidth(0.05)->setWarmStart([10.0, 10.0], 1.0)->setMaxIterations(1);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $history = $result->getBestValueHistory();
        $this->assertLessThan(0.5, $history[0], 'A fully warm-started, tightly jittered initial population should already be very close to the known optimum, before any generation runs.');
    }
}
