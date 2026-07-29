<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizerTest\Problem;

use BlackboxOptimizer\Problem\Parameter;
use BlackboxOptimizer\Problem\ParameterType;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;

class ParameterTest extends TestCase
{
    /**
     * @return void
     */
    public function testExposesItsOwnNameBoundsAndType(): void
    {
        $parameter = new Parameter('relevanceWeight', 0.1, 0.9, ParameterType::Continuous);

        $this->assertSame('relevanceWeight', $parameter->getName());
        $this->assertSame(0.1, $parameter->getLowerBound());
        $this->assertSame(0.9, $parameter->getUpperBound());
        $this->assertSame(ParameterType::Continuous, $parameter->getType());
    }

    /**
     * @return void
     */
    public function testDefaultsToContinuousWhenNoTypeIsGiven(): void
    {
        $parameter = new Parameter('x', -1.0, 1.0);

        $this->assertSame(ParameterType::Continuous, $parameter->getType());
    }

    /**
     * @return void
     */
    public function testThrowsWhenLowerBoundExceedsUpperBound(): void
    {
        $this->expectException(InvalidArgumentException::class);

        new Parameter('x', 5.0, 1.0);
    }
}
