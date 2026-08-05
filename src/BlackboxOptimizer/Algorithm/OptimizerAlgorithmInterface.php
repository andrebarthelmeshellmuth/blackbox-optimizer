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
     * A short, human-readable name for this algorithm (e.g. "CMA-ES"). Factual identification only --
     * this package stays deliberately unopinionated about which algorithm a caller "should" pick, so
     * this is not the place for a sales pitch; see {@see getDescription()} for the same constraint.
     *
     * @return string
     */
    public function getName(): string;

    /**
     * A one- or two-sentence, factual description of what this algorithm mechanically does -- e.g. "adapts
     * a full covariance matrix as it searches" -- not an opinion about how it compares to the others (no
     * "generally the stronger choice" or similar framing). This package's own docblocks and README already
     * describe each algorithm this way; this is that same factual prose made runtime-readable so a
     * consumer (e.g. a UI built on top of this package) doesn't have to hand-copy it. Any recommendation or
     * comparison between algorithms is scope for that consumer to add on top, not for this package, which
     * stays deliberately generic about problem domains and consumers alike.
     *
     * @return string
     */
    public function getDescription(): string;

    /**
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     *
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    public function optimize(ProblemInterface $problem): OptimizationResult;

    /**
     * Predicts the total number of {@see \BlackboxOptimizer\Problem\ProblemInterface::evaluate()} calls an
     * {@see optimize()} run will make, without actually running it -- e.g. so a caller can show a progress
     * bar's total before the run starts. Mirrors this algorithm's own real internal evaluation-count
     * bookkeeping (initial-population batches, an early-stopping mode's iteration ceiling, etc.) exactly,
     * so it never drifts out of sync with what {@see optimize()} actually does. Every setter here is
     * optional to call before {@see optimize()} because each algorithm falls back to its own sensible
     * default; the same applies here, EXCEPT for an algorithm whose own default population size depends on
     * the problem's dimension count (which this method, given no {@see \BlackboxOptimizer\Problem\ProblemInterface},
     * cannot know) -- such an algorithm requires an explicit {@see setPopulationSize()} call first and
     * throws otherwise.
     *
     * @throws \InvalidArgumentException When this algorithm cannot resolve a default population size
     *   without a problem instance and {@see setPopulationSize()} was never called.
     *
     * @return int
     */
    public function estimateEvaluationCount(): int;

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
