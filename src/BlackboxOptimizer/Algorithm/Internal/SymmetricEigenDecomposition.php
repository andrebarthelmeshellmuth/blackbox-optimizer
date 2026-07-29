<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm\Internal;

/**
 * The classic cyclic Jacobi eigenvalue algorithm for real symmetric matrices — repeatedly applies Givens
 * rotations that zero one off-diagonal pair at a time until the whole matrix is (numerically) diagonal.
 * Chosen deliberately over a more elaborate QR-algorithm-based solver: CMA-ES only ever needs this for its
 * own covariance matrix, which stays small for the black-box, derivative-free problems this package
 * targets (typically a handful to a few dozen dimensions), a regime where Jacobi's O(n^3) per-sweep cost
 * is trivial and its simplicity/reviewability matters far more than the faster asymptotic behavior a
 * QR-based solver would offer at hundreds+ dimensions. NOT intended as a general-purpose linear-algebra
 * utility.
 *
 * @internal Used only by {@see \BlackboxOptimizer\Algorithm\CmaEsAlgorithm}.
 */
class SymmetricEigenDecomposition
{
    /**
     * @var int
     */
    protected const MAX_SWEEPS = 100;

    /**
     * @var float
     */
    protected const CONVERGENCE_THRESHOLD = 1e-12;

    /**
     * Decomposes a real symmetric matrix A into A = V * diag(eigenvalues) * V^T.
     *
     * @param array<int, array<int, float>> $matrix Symmetric n x n matrix.
     *
     * @return array{eigenvalues: array<int, float>, eigenvectors: array<int, array<int, float>>} eigenvectors[i][j]
     *   is the i-th component of the j-th eigenvector (i.e. eigenvectors are COLUMNS), paired with
     *   eigenvalues[j] at the same index j.
     */
    public function decompose(array $matrix): array
    {
        $n = count($matrix);
        $a = $matrix;
        $v = $this->buildIdentity($n);

        for ($sweep = 0; $sweep < static::MAX_SWEEPS; $sweep++) {
            $offDiagonalSum = $this->sumOfSquaredOffDiagonals($a, $n);

            if ($offDiagonalSum < static::CONVERGENCE_THRESHOLD) {
                break;
            }

            for ($p = 0; $p < $n - 1; $p++) {
                for ($q = $p + 1; $q < $n; $q++) {
                    if (abs($a[$p][$q]) < static::CONVERGENCE_THRESHOLD) {
                        continue;
                    }

                    [$cos, $sin] = $this->computeRotation($a, $p, $q);
                    $this->applyRotation($a, $v, $p, $q, $cos, $sin, $n);
                }
            }
        }

        $eigenvalues = [];

        for ($i = 0; $i < $n; $i++) {
            $eigenvalues[$i] = $a[$i][$i];
        }

        return ['eigenvalues' => $eigenvalues, 'eigenvectors' => $v];
    }

    /**
     * @param int $n
     *
     * @return array<int, array<int, float>>
     */
    protected function buildIdentity(int $n): array
    {
        $identity = [];

        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $identity[$i][$j] = $i === $j ? 1.0 : 0.0;
            }
        }

        return $identity;
    }

    /**
     * @param array<int, array<int, float>> $a
     * @param int $n
     *
     * @return float
     */
    protected function sumOfSquaredOffDiagonals(array $a, int $n): float
    {
        $sum = 0.0;

        for ($i = 0; $i < $n - 1; $i++) {
            for ($j = $i + 1; $j < $n; $j++) {
                $sum += $a[$i][$j] ** 2;
            }
        }

        return $sum;
    }

    /**
     * @param array<int, array<int, float>> $a
     * @param int $p
     * @param int $q
     *
     * @return array{0: float, 1: float} [cos(theta), sin(theta)]
     */
    protected function computeRotation(array $a, int $p, int $q): array
    {
        if ($a[$p][$p] === $a[$q][$q]) {
            $theta = M_PI / 4.0;

            return [cos($theta), sin($theta) * ($a[$p][$q] >= 0 ? 1.0 : -1.0)];
        }

        $tau = ($a[$q][$q] - $a[$p][$p]) / (2.0 * $a[$p][$q]);
        $tauSign = $tau >= 0 ? 1.0 : -1.0;
        $t = $tauSign / (abs($tau) + sqrt(1.0 + $tau ** 2));
        $cos = 1.0 / sqrt(1.0 + $t ** 2);
        $sin = $t * $cos;

        return [$cos, $sin];
    }

    /**
     * Applies the Givens rotation zeroing A[p][q] to both A (in place) and the accumulated eigenvector
     * matrix V (in place).
     *
     * @param array<int, array<int, float>> $a
     * @param array<int, array<int, float>> $v
     * @param int $p
     * @param int $q
     * @param float $cos
     * @param float $sin
     * @param int $n
     *
     * @return void
     */
    protected function applyRotation(array &$a, array &$v, int $p, int $q, float $cos, float $sin, int $n): void
    {
        $app = $a[$p][$p];
        $aqq = $a[$q][$q];
        $apq = $a[$p][$q];

        $a[$p][$p] = $cos ** 2 * $app - 2 * $sin * $cos * $apq + $sin ** 2 * $aqq;
        $a[$q][$q] = $sin ** 2 * $app + 2 * $sin * $cos * $apq + $cos ** 2 * $aqq;
        $a[$p][$q] = 0.0;
        $a[$q][$p] = 0.0;

        for ($i = 0; $i < $n; $i++) {
            if ($i === $p || $i === $q) {
                continue;
            }

            $aip = $a[$i][$p];
            $aiq = $a[$i][$q];

            $a[$i][$p] = $cos * $aip - $sin * $aiq;
            $a[$p][$i] = $a[$i][$p];
            $a[$i][$q] = $sin * $aip + $cos * $aiq;
            $a[$q][$i] = $a[$i][$q];
        }

        for ($i = 0; $i < $n; $i++) {
            $vip = $v[$i][$p];
            $viq = $v[$i][$q];

            $v[$i][$p] = $cos * $vip - $sin * $viq;
            $v[$i][$q] = $sin * $vip + $cos * $viq;
        }
    }
}
