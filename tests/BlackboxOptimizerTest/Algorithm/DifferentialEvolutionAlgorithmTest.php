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
use Random\Engine\PcgOneseq128XslRr64;
use Random\Randomizer;
use ReflectionMethod;

/**
 * Validates against known toy benchmark functions (sphere, Rosenbrock) with known optima, per this
 * package's own decision to prove any optimizer implementation correct BEFORE ever pointing it at a real
 * (and much more expensive to debug) objective function.
 */
class DifferentialEvolutionAlgorithmTest extends TestCase
{
    /**
     * Every algorithm here is stochastic, so an unseeded run turns each convergence assertion below into a
     * probabilistic claim rather than a fact: the same commit passes or fails depending on the draw, which
     * is exactly how this suite used to go red on one PHP version and green on another. Seeding pins the
     * draw, so a failure always means a real behavioral regression.
     *
     * @var int
     */
    protected const RNG_SEED = 20260818;

    /**
     * A draw that reproduced the premature-stop defect described in
     * {@see testAStalledBestValueDoesNotStopARunWhileThePopulationIsStillSpread()}. Pinned separately from
     * {@see RNG_SEED} so that test keeps witnessing the bug even if the suite-wide seed is ever changed.
     *
     * @var int
     */
    protected const PREMATURE_STOP_WITNESS_SEED = 4;

    /**
     * A seeded engine rather than the production default: {@see Randomizer} with no engine uses
     * {@see \Random\Engine\Secure}, which cannot be seeded by design. PcgOneseq128XslRr64 is
     * deterministic for a given seed and stable across PHP versions and platforms, which is what makes
     * the assertions reproducible.
     *
     * @return \BlackboxOptimizer\Algorithm\DifferentialEvolutionAlgorithm
     */
    protected function createAlgorithm(): DifferentialEvolutionAlgorithm
    {
        return new DifferentialEvolutionAlgorithm(new Randomizer(new PcgOneseq128XslRr64(static::RNG_SEED)));
    }

    /**
     * @return void
     */
    public function testImplementsTheGenericOptimizerAlgorithmInterface(): void
    {
        $this->assertInstanceOf(OptimizerAlgorithmInterface::class, $this->createAlgorithm());
    }

    /**
     * @return void
     */
    public function testExposesFactualNameAndDescriptionMetadata(): void
    {
        $algorithm = $this->createAlgorithm();

        $this->assertSame('Differential Evolution', $algorithm->getName());
        $this->assertNotSame('', $algorithm->getDescription());
    }

    /**
     * @return void
     */
    public function testExposesTheSameSafetyIterationCeilingItsOwnTrustTerminationCriteriaModeUses(): void
    {
        $this->assertSame(10000, $this->createAlgorithm()->getSafetyIterationCeiling());
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
        $algorithm = $this->createAlgorithm()->setMaxIterations(10);

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
        $algorithm = $this->createAlgorithm()->setPopulationSize(30)->setMaxIterations(150);

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

        $algorithm = $this->createAlgorithm();
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

        $algorithm = $this->createAlgorithm();
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

        $algorithm = $this->createAlgorithm();
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

        $algorithm = $this->createAlgorithm();
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

        $this->createAlgorithm()->setPopulationSize(3);
    }

    /**
     * @return void
     */
    public function testSetCrossoverProbabilityRejectsAnOutOfRangeValue(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->createAlgorithm()->setCrossoverProbability(1.5);
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
        $this->createAlgorithm()->optimize($problem);
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

        $algorithm = $this->createAlgorithm();
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
     * DE's own TolFun-equivalent ({@see \BlackboxOptimizer\Algorithm\DifferentialEvolutionAlgorithm::hasFlatFitnessHistory()})
     * is a local reimplementation, not shared with {@see \BlackboxOptimizer\Algorithm\Internal\TerminationCriteria}
     * (see this class's own docblock for why) -- the population-collapse test above never exercises it,
     * since a converging sphere's population collapses in SPACE well before its fitness history could ever
     * flatten first. A perfectly constant objective isolates the opposite: every candidate scores identically
     * regardless of position, so a trial vector is never strictly better than its target and the population
     * itself never changes generation to generation -- spread stays exactly whatever the initial random draw
     * produced (not collapsed), while the best-value history is 0.0 from the very first generation, hitting
     * the flat-fitness window as soon as it fills.
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

        $algorithm = $this->createAlgorithm();
        $algorithm->setPopulationSize(8)->setMaxIterations(500);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert -- fitness history length for n=1, lambda=8 is 10 + ceil(30*1/8) = 14 (same formula
        // TerminationCriteria::resolveFitnessHistoryLength() uses, reused as-is by this class), plus the
        // initial-population entry recorded before the generation loop even starts (see
        // testTrustTerminationCriteriaOverridesATooSmallSetMaxIterations() above for the same +1).
        $this->assertLessThanOrEqual(15, count($result->getBestValueHistory()), 'A perfectly flat objective should trigger the fitness-plateau check as soon as the history window fills.');
    }

    /**
     * The other half of the constant-objective test above, and a regression guard for a real defect: the
     * fitness-plateau check used to stop a run on its own, which is wrong for DE. One lucky early
     * individual can stay the population's best for longer than the plateau window while the rest of the
     * population is still spread out and still converging, so the run was cut off at a bad point and no
     * amount of extra maxIterations budget could recover it -- a 3x larger budget produced a byte-identical
     * result. Measured over 200 seeds this hit roughly one run in five, which is what made this suite go
     * red on one PHP version and green on another.
     *
     * Uses its own seed rather than {@see RNG_SEED} precisely because the default seed is one of the draws
     * that got LUCKY: the bug only reproduces on particular draws, so pinning a witness seed is the only
     * way to keep this honest. The generation-count assertion is the one that actually catches a
     * regression -- under the defect this run stopped after 20 of 151 generations (570 of 4530
     * evaluations) at a best value of 6.4e-2, versus the 4.9e-14 it reaches when allowed to finish.
     *
     * @return void
     */
    public function testAStalledBestValueDoesNotStopARunWhileThePopulationIsStillSpread(): void
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

        $algorithm = new DifferentialEvolutionAlgorithm(new Randomizer(new PcgOneseq128XslRr64(static::PREMATURE_STOP_WITNESS_SEED)));
        $algorithm->setPopulationSize(30)->setMaxIterations(150);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $this->assertGreaterThan(100, count($result->getBestValueHistory()), 'A plateaued best value must not stop the run while the population is still spread out.');
        $this->assertLessThan(1e-4, $result->getBestValue(), 'Allowed to run to convergence, DE reaches the sphere function\'s known minimum of 0.');
    }

    /**
     * @return void
     */
    public function testSetWarmStartRejectsAFractionBelowZero(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->createAlgorithm()->setWarmStart([0.0], -0.1);
    }

    /**
     * @return void
     */
    public function testSetWarmStartRejectsAFractionAboveOne(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->createAlgorithm()->setWarmStart([0.0], 1.1);
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
        $this->createAlgorithm()->setWarmStart([3.0], 0.0)->optimize($problem);
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

        $algorithm = $this->createAlgorithm();
        $algorithm->setPopulationSize(20)->setStepWidth(0.05)->setWarmStart([10.0, 10.0], 1.0)->setMaxIterations(1);

        // Act
        $result = $algorithm->optimize($problem);

        // Assert
        $history = $result->getBestValueHistory();
        $this->assertLessThan(0.5, $history[0], 'A fully warm-started, tightly jittered initial population should already be very close to the known optimum, before any generation runs.');
    }

    /**
     * The extremes (fraction=0.0, fraction=1.0) are already proven above, but the actual split arithmetic
     * -- {@see \BlackboxOptimizer\Algorithm\AbstractOptimizerAlgorithm::seedInitialPopulation()}'s
     * `(int)round($count * $warmStartFraction)` -- is shared with
     * {@see \BlackboxOptimizerTest\Algorithm\RechenbergSchwefelEsAlgorithmTest::testSeedInitialPopulationSplitsExactlyByWarmStartFractionAtAPartialFraction()}
     * and never exercised at a real fraction in between, where a rounding/off-by-one bug would actually be
     * visible. An outcome-based test (an optimize() run's best value) can't isolate this cleanly: even a
     * handful of tightly-jittered warm members already pulls the population-wide best value close to the
     * warm-start vector regardless of whether the true split is 20% or 80%, so a wrong count could still
     * pass. Calling the shared, inherited seeding method directly and counting how many of its own returned
     * vectors actually land near the vector is the only way to pin down the exact count, not just "some
     * did."
     *
     * @return void
     */
    public function testSeedInitialPopulationSplitsExactlyByWarmStartFractionAtAPartialFraction(): void
    {
        // Arrange -- bounds are huge relative to the jitter scale, so a member seeded via the random path
        // has a negligible chance of accidentally landing within the warm-detection radius below.
        $algorithm = $this->createAlgorithm();
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
