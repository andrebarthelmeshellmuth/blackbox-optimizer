<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Problem;

/**
 * Continuous is the only type any algorithm in this package actually treats specially today (Integer is
 * accepted by {@see Parameter} and reported back on request, but no shipped algorithm rounds a candidate
 * to the nearest integer for it yet) -- present now so a Problem can already DECLARE an integer dimension
 * honestly, ahead of an algorithm that enforces it.
 */
enum ParameterType
{
    case Continuous;
    case Integer;
}
