<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm;

use BlackboxOptimizer\Problem\ProblemInterface;

/**
 * Deliberately generic -- nothing in this namespace knows about any particular problem domain. An
 * implementer optimizes an arbitrary black-box, derivative-free, box-constrained MINIMIZATION problem:
 * given a {@see ProblemInterface} (its parameter list plus a way to score any point in it), find the
 * vector that minimizes it.
 *
 * The three setters below are the knobs every population-based algorithm in this package already has SOME
 * version of, given one shared name instead of a bespoke method per algorithm:
 * - setStepWidth(): CMA-ES's initial global step size (sigma_0); Differential Evolution's mutation
 *   factor (F) -- both mean "how big are the first exploratory steps."
 * - setPopulationSize(): CMA-ES's lambda; DE's population size.
 * - setMaxIterations(): a fixed generation/iteration count, this package's only stopping criterion.
 *
 * Anything narrower than that stays a concrete, algorithm-specific method on the concrete class (e.g.
 * CmaEsAlgorithm::setInitialMean(), DifferentialEvolutionAlgorithm::setCrossoverProbability()) rather than
 * cramming every algorithm's own knobs into this interface, which would either lose meaning (a bag of
 * mixed-purpose floats) or force every implementer to support parameters only some other algorithm needs.
 * Every setter is optional to call at all -- skipping all of them falls back to that algorithm's own
 * sensible defaults.
 */
interface OptimizerAlgorithmInterface
{
    /**
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     *
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    public function optimize(ProblemInterface $problem): OptimizationResult;

    /**
     * @param float $stepWidth Must be greater than 0.
     *
     * @throws \InvalidArgumentException
     *
     * @return static
     */
    public function setStepWidth(float $stepWidth): static;

    /**
     * @param int $populationSize Must be at least 4 (every algorithm in this package needs at least that
     *   many candidates per generation to do anything meaningful).
     *
     * @throws \InvalidArgumentException
     *
     * @return static
     */
    public function setPopulationSize(int $populationSize): static;

    /**
     * @param int $maxIterations Must be at least 1.
     *
     * @throws \InvalidArgumentException
     *
     * @return static
     */
    public function setMaxIterations(int $maxIterations): static;
}
