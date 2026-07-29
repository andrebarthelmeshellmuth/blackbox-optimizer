<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizerTest\Problem;

use BlackboxOptimizer\Problem\CallableProblem;
use PHPUnit\Framework\TestCase;

class CallableProblemTest extends TestCase
{
    /**
     * @return void
     */
    public function testGetParametersReturnsOneParameterPerBoundPairInOrder(): void
    {
        $problem = new CallableProblem(static fn (): float => 0.0, [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);

        $parameters = $problem->getParameters();

        $this->assertCount(3, $parameters);
        $this->assertSame(-2.0, $parameters[1]->getLowerBound());
        $this->assertSame(2.0, $parameters[1]->getUpperBound());
    }

    /**
     * @return void
     */
    public function testEvaluateDelegatesToTheGivenCallable(): void
    {
        $problem = new CallableProblem(static fn (array $vector): float => array_sum($vector), [-1.0], [1.0]);

        $this->assertSame(6.0, $problem->evaluate([1.0, 2.0, 3.0]));
    }
}
