<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizerTest\Algorithm;

use BlackboxOptimizer\Algorithm\CmaEsAlgorithm;
use BlackboxOptimizer\Algorithm\OptimizerAlgorithmInterface;
use BlackboxOptimizer\Problem\CallableProblem;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;

/**
 * Benchmark-function discipline: prove correctness against known optima BEFORE ever pointing this at a
 * real (and much more expensive to debug) objective function.
 */
class CmaEsAlgorithmTest extends TestCase
{
    /**
     * @return void
     */
    public function testImplementsTheGenericOptimizerAlgorithmInterface(): void
    {
        $this->assertInstanceOf(OptimizerAlgorithmInterface::class, new CmaEsAlgorithm());
    }

    /**
     * @return void
     */
    public function testExposesFactualNameAndDescriptionMetadata(): void
    {
        $algorithm = new CmaEsAlgorithm();

        $this->assertSame('CMA-ES', $algorithm->getName());
        $this->assertNotSame('', $algorithm->getDescription());
    }

    /**
     * @return void
     */
    public function testEstimateEvaluationCountRequiresAnExplicitPopulationSize(): void
    {
        // Arrange
        $algorithm = new CmaEsAlgorithm();

        // Assert
        $this->expectException(InvalidArgumentException::class);

        // Act
        $algorithm->estimateEvaluationCount();
    }

    /**
     * @return void
     */
    public function testEstimateEvaluationCountMatchesPopulationSizeTimesMaxIterations(): void
    {
        // Arrange
        $algorithm = (new CmaEsAlgorithm())->setPopulationSize(10)->setMaxIterations(50);

        // Act
        $estimate = $algorithm->estimateEvaluationCount();

        // Assert
        $this->assertSame(500, $estimate);
    }

    /**
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

        $algorithm = new CmaEsAlgorithm();
        $algorithm->setPopulationSize(12)->setStepWidth(1.0)->setMaxIterations(100);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThan(1e-4, $result->getBestValue(), 'CMA-ES should get very close to the sphere function\'s known minimum of 0.');

        foreach ($result->getBestVector() as $component) {
            $this->assertEqualsWithDelta(0.0, $component, 0.1, 'Each dimension should converge close to the known optimum at the origin.');
        }
    }

    /**
     * estimateEvaluationCount() is an upper bound, not an exact prediction -- CMA-ES's always-on
     * TolX/TolXUp/ConditionCov/TolFun early termination (see {@see \BlackboxOptimizer\Algorithm\Internal\TerminationCriteria})
     * can make a real run stop before spending its full maxIterations budget. This is the guard against the
     * two ever silently drifting apart in the other direction (a real run spending MORE than predicted).
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

        $algorithm = new CmaEsAlgorithm();
        $algorithm->setPopulationSize(12)->setStepWidth(1.0)->setMaxIterations(100);

        $estimate = $algorithm->estimateEvaluationCount();

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThanOrEqual($estimate, $result->getEvaluationCount());
    }

    /**
     * CMA-ES's whole reason for existing over simpler methods is navigating exactly this kind of narrow,
     * curved valley via its adapted covariance matrix -- a meaningfully stronger correctness check than
     * the sphere function alone.
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

        $algorithm = new CmaEsAlgorithm();
        $algorithm->setPopulationSize(16)->setStepWidth(0.5)->setMaxIterations(200);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThan(0.05, $result->getBestValue(), 'CMA-ES should get close to the Rosenbrock function\'s known minimum of 0.');
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

        $algorithm = new CmaEsAlgorithm();
        $algorithm->setPopulationSize(8)->setStepWidth(1.0)->setMaxIterations(5);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert -- 5 generations * 8 candidates each, no separate initial-population evaluation batch
        // (unlike DE, CMA-ES's first generation IS the initial sample).
        $this->assertSame(40, $result->getEvaluationCount());
        $this->assertCount(5, $result->getBestValueHistory(), 'One history entry per generation.');

        $history = $result->getBestValueHistory();
        $historyCount = count($history);

        for ($i = 1; $i < $historyCount; $i++) {
            $this->assertLessThanOrEqual($history[$i - 1], $history[$i], 'The best-found value must never get worse from one generation to the next.');
        }
    }

    /**
     * TolX: a well-behaved, easily-converged problem given a generous generation budget should stop well
     * before spending it all, once sigma has shrunk close to nothing -- proves the early-termination
     * criteria actually fire, not just that they compile.
     *
     * @return void
     */
    public function testOptimizeStopsEarlyViaTolXOnATriviallyConvergedSphereFunction(): void
    {
        // Arrange
        $sphere = static fn (array $vector): float => $vector[0] ** 2 + $vector[1] ** 2;
        $problem = new CallableProblem($sphere, [-5.0, -5.0], [5.0, 5.0]);

        $algorithm = new CmaEsAlgorithm();
        $algorithm->setPopulationSize(8)->setStepWidth(1.0)->setMaxIterations(500);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThan(500, count($result->getBestValueHistory()), 'A trivially convergent sphere function should trigger TolX long before the 500-generation budget is spent.');
        $this->assertLessThan(1e-6, $result->getBestValue(), 'Still converged close to the known minimum despite stopping early.');
    }

    /**
     * TolFun: an objective whose value never changes at all should stop as soon as the fitness-history
     * window fills, since its range is always exactly 0 -- the most direct possible trigger for this
     * specific criterion, isolated from TolX/TolXUp/ConditionCov (sigma has no reason to move unusually
     * here, so none of the other three would fire first).
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

        $algorithm = new CmaEsAlgorithm();
        $algorithm->setPopulationSize(8)->setStepWidth(1.0)->setMaxIterations(500);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert -- fitness history length for n=1, lambda=8 is 10 + ceil(30*1/8) = 14.
        $this->assertLessThanOrEqual(14, count($result->getBestValueHistory()), 'A perfectly flat objective should trigger TolFun as soon as the fitness-history window fills.');
    }

    /**
     * @return void
     */
    public function testOptimizeUsesTheMidpointOfFiniteBoundsAsTheDefaultInitialMean(): void
    {
        // Arrange -- a sphere function shifted so its minimum is exactly at the bounds' midpoint (10, 10),
        // with a tiny population/generation budget: only a starting point already close to the optimum
        // could possibly get this close in so few evaluations, proving the midpoint default is honored.
        $shiftedSphere = static function (array $vector): float {
            return ($vector[0] - 10) ** 2 + ($vector[1] - 10) ** 2;
        };

        $problem = new CallableProblem($shiftedSphere, [5.0, 5.0], [15.0, 15.0]);

        $algorithm = new CmaEsAlgorithm();
        $algorithm->setPopulationSize(4)->setStepWidth(0.1)->setMaxIterations(1);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThan(1.0, $result->getBestValue(), 'Starting from the bounds\' midpoint (10, 10), which is exactly the optimum, even one generation should land very close.');
    }

    /**
     * @return void
     */
    public function testOptimizeThrowsWhenABoundIsInfiniteAndNoInitialMeanWasGiven(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $problem = new CallableProblem(static fn (array $vector): float => $vector[0] ** 2, [-INF], [INF]);
        (new CmaEsAlgorithm())->optimize($problem);
    }

    /**
     * @return void
     */
    public function testOptimizeAcceptsAnExplicitInitialMeanForAnUnboundedDimension(): void
    {
        // Arrange
        $sphere = static fn (array $vector): float => $vector[0] ** 2;
        $problem = new CallableProblem($sphere, [-INF], [INF]);

        $algorithm = new CmaEsAlgorithm();
        $algorithm->setPopulationSize(8)->setStepWidth(1.0)->setInitialMean([3.0])->setMaxIterations(60);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertLessThan(1e-3, $result->getBestValue());
    }

    /**
     * @return void
     */
    public function testSetPopulationSizeRejectsATooSmallPopulation(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new CmaEsAlgorithm())->setPopulationSize(3);
    }

    /**
     * @return void
     */
    public function testSetStepWidthRejectsANonPositiveValue(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new CmaEsAlgorithm())->setStepWidth(0.0);
    }

    /**
     * @return void
     */
    public function testSetMaxIterationsRejectsATooSmallValue(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new CmaEsAlgorithm())->setMaxIterations(0);
    }

    /**
     * A deliberately tiny setMaxIterations() (far too small to converge a 2D sphere function) is ignored
     * once trustTerminationCriteria() is on -- the run keeps going well past it, converges close to the
     * known minimum via TolX, and still stops well short of SAFETY_ITERATION_CEILING (proving it stopped
     * via the criteria, not by exhausting the safety ceiling itself).
     *
     * @return void
     */
    public function testTrustTerminationCriteriaOverridesATooSmallSetMaxIterations(): void
    {
        // Arrange
        $sphere = static fn (array $vector): float => $vector[0] ** 2 + $vector[1] ** 2;
        $problem = new CallableProblem($sphere, [-5.0, -5.0], [5.0, 5.0]);

        $algorithm = new CmaEsAlgorithm();
        $algorithm->setPopulationSize(8)->setStepWidth(1.0)->setMaxIterations(3)->trustTerminationCriteria();

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $generationsRun = count($result->getBestValueHistory());
        $this->assertGreaterThan(3, $generationsRun, 'The 3-generation cap from setMaxIterations() must be ignored once trustTerminationCriteria() is on.');
        $this->assertLessThan(10000, $generationsRun, 'Should stop via TolX well before the safety ceiling, not by exhausting it.');
        $this->assertLessThan(1e-6, $result->getBestValue(), 'Given the room to actually converge, the known minimum should be reached closely.');
    }
}
