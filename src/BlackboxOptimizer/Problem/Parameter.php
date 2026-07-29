<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Problem;

use InvalidArgumentException;

/**
 * One dimension of a {@see ProblemInterface}'s search space: a name (for logging/debugging only, never
 * used by an algorithm to make a decision), a box bound, and a type. A Problem with 4 dimensions is 4 of
 * these, not 4 entries in 3 parallel arrays a caller has to keep in sync by position.
 *
 * Restrictions beyond an independent per-dimension box bound (e.g. "these 3 parameters must sum to 1")
 * are deliberately NOT expressible here -- a linear-equality constraint over several dimensions is a
 * different kind of thing than a per-dimension bound, and belongs in a reparametrization on the CALLER's
 * side (an unconstrained-space transform feeding this package only ever box-bounded dimensions), not in
 * this package's own vocabulary.
 */
final class Parameter
{
    /**
     * @param string $name
     * @param float $lowerBound
     * @param float $upperBound
     * @param \BlackboxOptimizer\Problem\ParameterType $type
     *
     * @throws \InvalidArgumentException
     */
    public function __construct(
        private string $name,
        private float $lowerBound,
        private float $upperBound,
        private ParameterType $type = ParameterType::Continuous,
    ) {
        if ($this->lowerBound > $this->upperBound) {
            throw new InvalidArgumentException(sprintf(
                'Parameter "%s": lower bound (%f) exceeds upper bound (%f).',
                $this->name,
                $this->lowerBound,
                $this->upperBound,
            ));
        }
    }

    /**
     * @return string
     */
    public function getName(): string
    {
        return $this->name;
    }

    /**
     * @return float
     */
    public function getLowerBound(): float
    {
        return $this->lowerBound;
    }

    /**
     * @return float
     */
    public function getUpperBound(): float
    {
        return $this->upperBound;
    }

    /**
     * @return \BlackboxOptimizer\Problem\ParameterType
     */
    public function getType(): ParameterType
    {
        return $this->type;
    }
}
