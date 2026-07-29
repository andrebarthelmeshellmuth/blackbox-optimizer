<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm\Internal;

/**
 * Plain vector/matrix arithmetic (plain PHP arrays, no external dependency) — extracted out of
 * {@see \BlackboxOptimizer\Algorithm\CmaEsAlgorithm} once that class's own complexity grew large enough
 * to trip this package's phpmd threshold, the same reasoning that already justified extracting
 * {@see SymmetricEigenDecomposition} as its own class: these are generic linear-algebra primitives, not
 * CMA-ES-specific algorithmic logic.
 *
 * @internal Used only by {@see \BlackboxOptimizer\Algorithm\CmaEsAlgorithm}.
 */
class VectorMath
{
    /**
     * @param array<int, array<int, float>> $matrix
     * @param array<int, float> $vector
     *
     * @return array<int, float>
     */
    public function matrixVectorMultiply(array $matrix, array $vector): array
    {
        $result = [];

        foreach ($matrix as $i => $row) {
            $sum = 0.0;

            foreach ($row as $j => $value) {
                $sum += $value * $vector[$j];
            }

            $result[$i] = $sum;
        }

        return $result;
    }

    /**
     * matrix^T * vector -- since {@see SymmetricEigenDecomposition} returns eigenvectors as columns of
     * $matrix, this is what's needed to project a vector INTO eigenspace (matrixVectorMultiply projects
     * back OUT of it).
     *
     * @param array<int, array<int, float>> $matrix
     * @param array<int, float> $vector
     *
     * @return array<int, float>
     */
    public function transformByTranspose(array $matrix, array $vector): array
    {
        $n = count($vector);
        $result = array_fill(0, $n, 0.0);

        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $result[$j] += $matrix[$i][$j] * $vector[$i];
            }
        }

        return $result;
    }

    /**
     * @param array<int, float> $diagonal
     * @param array<int, float> $vector
     *
     * @return array<int, float>
     */
    public function applyDiagonal(array $diagonal, array $vector): array
    {
        $result = [];

        foreach ($vector as $index => $value) {
            $result[$index] = $diagonal[$index] * $value;
        }

        return $result;
    }

    /**
     * @param array<int, float> $vector
     * @param float $scalar
     *
     * @return array<int, float>
     */
    public function scaleVector(array $vector, float $scalar): array
    {
        return array_map(static fn (float $value): float => $value * $scalar, $vector);
    }

    /**
     * @param array<int, float> $a
     * @param array<int, float> $b
     *
     * @return array<int, float>
     */
    public function addVectors(array $a, array $b): array
    {
        $result = [];

        foreach ($a as $index => $value) {
            $result[$index] = $value + $b[$index];
        }

        return $result;
    }

    /**
     * @param array<int, float> $a
     * @param array<int, float> $b
     *
     * @return array<int, float>
     */
    public function subtractVectors(array $a, array $b): array
    {
        $result = [];

        foreach ($a as $index => $value) {
            $result[$index] = $value - $b[$index];
        }

        return $result;
    }

    /**
     * @param array<int, float> $vector
     *
     * @return float
     */
    public function vectorNorm(array $vector): float
    {
        $sumOfSquares = 0.0;

        foreach ($vector as $value) {
            $sumOfSquares += $value ** 2;
        }

        return sqrt($sumOfSquares);
    }
}
