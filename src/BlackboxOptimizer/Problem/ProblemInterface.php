<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Problem;

/**
 * The thing being optimized: a fixed list of {@see Parameter}s (the search space) plus a way to score any
 * point in it. An algorithm never inspects, caches, or assumes anything about what evaluate() computes --
 * this is what lets the SAME algorithm be validated against toy benchmark functions in this package's own
 * tests and separately be handed a real, much more expensive problem by a consuming project.
 *
 * Implementations are expected to be cheap to construct and stateless across evaluate() calls -- an
 * algorithm may call evaluate() many thousands of times over one optimize() run.
 */
interface ProblemInterface
{
    /**
     * @return array<int, \BlackboxOptimizer\Problem\Parameter>
     */
    public function getParameters(): array;

    /**
     * @param array<int, float> $vector Same order and length as {@see getParameters()}.
     *
     * @return float LOWER is better (minimization). A maximization problem negates its own score before
     *   returning it here.
     */
    public function evaluate(array $vector): float;
}
