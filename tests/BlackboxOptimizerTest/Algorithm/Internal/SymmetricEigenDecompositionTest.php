<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizerTest\Algorithm\Internal;

use BlackboxOptimizer\Algorithm\Internal\SymmetricEigenDecomposition;
use PHPUnit\Framework\TestCase;

class SymmetricEigenDecompositionTest extends TestCase
{
    /**
     * @return void
     */
    public function testDecomposeReturnsTheDiagonalItselfAsEigenvaluesForAnAlreadyDiagonalMatrix(): void
    {
        // Arrange
        $matrix = [
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ];

        // Act
        $result = (new SymmetricEigenDecomposition())->decompose($matrix);

        // Assert
        $this->assertEqualsWithDelta([3.0, 1.0, 2.0], $result['eigenvalues'], 1e-9);
    }

    /**
     * [[2,1],[1,2]] has the textbook-known eigenvalues 3 (eigenvector (1,1)/sqrt(2)) and 1 (eigenvector
     * (1,-1)/sqrt(2)) -- a hand-verifiable case, not just a reconstruction check.
     *
     * @return void
     */
    public function testDecomposeFindsTheKnownEigenvaluesOfASimpleTwoByTwoMatrix(): void
    {
        // Arrange
        $matrix = [
            [2.0, 1.0],
            [1.0, 2.0],
        ];

        // Act
        $result = (new SymmetricEigenDecomposition())->decompose($matrix);

        // Assert
        $sortedEigenvalues = $result['eigenvalues'];
        sort($sortedEigenvalues);
        $this->assertEqualsWithDelta([1.0, 3.0], $sortedEigenvalues, 1e-9);
    }

    /**
     * The general, hand-picked-matrix-independent correctness check: for ANY real symmetric matrix,
     * reconstructing V * diag(eigenvalues) * V^T must give back the original matrix.
     *
     * @return void
     */
    public function testDecomposeReconstructsTheOriginalMatrixFromEigenvaluesAndEigenvectors(): void
    {
        // Arrange -- an arbitrary, non-trivial 4x4 symmetric matrix.
        $matrix = [
            [4.0, 1.0, 2.0, 0.5],
            [1.0, 3.0, 0.0, 1.5],
            [2.0, 0.0, 5.0, 0.0],
            [0.5, 1.5, 0.0, 2.0],
        ];

        // Act
        $result = (new SymmetricEigenDecomposition())->decompose($matrix);
        $reconstructed = $this->reconstruct($result['eigenvectors'], $result['eigenvalues']);

        // Assert
        foreach ($matrix as $i => $row) {
            foreach ($row as $j => $expectedValue) {
                $this->assertEqualsWithDelta($expectedValue, $reconstructed[$i][$j], 1e-6, sprintf('Mismatch at [%d][%d]', $i, $j));
            }
        }
    }

    /**
     * Eigenvectors of a real symmetric matrix must be orthonormal: V^T * V = identity.
     *
     * @return void
     */
    public function testDecomposeReturnsOrthonormalEigenvectors(): void
    {
        // Arrange
        $matrix = [
            [4.0, 1.0, 2.0, 0.5],
            [1.0, 3.0, 0.0, 1.5],
            [2.0, 0.0, 5.0, 0.0],
            [0.5, 1.5, 0.0, 2.0],
        ];

        // Act
        $result = (new SymmetricEigenDecomposition())->decompose($matrix);
        $v = $result['eigenvectors'];
        $n = count($v);

        // Assert
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $dotProduct = 0.0;

                for ($k = 0; $k < $n; $k++) {
                    $dotProduct += $v[$k][$i] * $v[$k][$j];
                }

                $this->assertEqualsWithDelta($i === $j ? 1.0 : 0.0, $dotProduct, 1e-6, sprintf('Column %d . Column %d', $i, $j));
            }
        }
    }

    /**
     * @param array<int, array<int, float>> $eigenvectors
     * @param array<int, float> $eigenvalues
     *
     * @return array<int, array<int, float>>
     */
    protected function reconstruct(array $eigenvectors, array $eigenvalues): array
    {
        $n = count($eigenvalues);
        $reconstructed = [];

        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $sum = 0.0;

                for ($k = 0; $k < $n; $k++) {
                    $sum += $eigenvectors[$i][$k] * $eigenvalues[$k] * $eigenvectors[$j][$k];
                }

                $reconstructed[$i][$j] = $sum;
            }
        }

        return $reconstructed;
    }
}
