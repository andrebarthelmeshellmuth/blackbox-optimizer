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
}
