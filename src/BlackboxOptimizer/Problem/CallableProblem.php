<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Problem;

/**
 * A {@see ProblemInterface} built from a plain callable plus one bound pair per dimension -- the common
 * case (a benchmark function in this package's own tests, or a consumer with no named parameters or
 * cross-dimension constraints of its own to express) doesn't need a dedicated class implementing
 * ProblemInterface just to wrap one closure and two arrays.
 *
 * A consumer with real per-parameter names, an Integer dimension, or its own reparametrization (e.g. a
 * simplex-constrained set of weights, which needs to convert to/from an unconstrained space before and
 * after this package ever sees it) should implement ProblemInterface directly instead -- this class is
 * the convenient default, not the only option.
 */
final class CallableProblem implements ProblemInterface
{
    /**
     * @var callable
     */
    private $objectiveFunction;

    /**
     * @var array<int, \BlackboxOptimizer\Problem\Parameter>
     */
    private array $parameters;

    /**
     * @param callable $objectiveFunction `array<int, float> $vector -> float`, LOWER is better.
     * @param array<int, float> $lowerBounds One bound per dimension, same length as $upperBounds.
     * @param array<int, float> $upperBounds One bound per dimension, same length as $lowerBounds.
     */
    public function __construct(callable $objectiveFunction, array $lowerBounds, array $upperBounds)
    {
        $this->objectiveFunction = $objectiveFunction;
        $this->parameters = $this->buildParameters($lowerBounds, $upperBounds);
    }

    /**
     * @return array<int, \BlackboxOptimizer\Problem\Parameter>
     */
    public function getParameters(): array
    {
        return $this->parameters;
    }

    /**
     * @param array<int, float> $vector
     *
     * @return float
     */
    public function evaluate(array $vector): float
    {
        return ($this->objectiveFunction)($vector);
    }

    /**
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @return array<int, \BlackboxOptimizer\Problem\Parameter>
     */
    private function buildParameters(array $lowerBounds, array $upperBounds): array
    {
        $parameters = [];

        foreach (array_values($lowerBounds) as $index => $lowerBound) {
            $parameters[] = new Parameter((string)$index, $lowerBound, array_values($upperBounds)[$index]);
        }

        return $parameters;
    }
}
