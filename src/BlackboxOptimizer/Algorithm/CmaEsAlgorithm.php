<?php

/**
 * This file is part of the andrebarthelmeshellmuth/blackbox-optimizer package.
 * For full license information, please view the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace BlackboxOptimizer\Algorithm;

use BlackboxOptimizer\Algorithm\Internal\SymmetricEigenDecomposition;
use BlackboxOptimizer\Algorithm\Internal\VectorMath;
use BlackboxOptimizer\Problem\ProblemInterface;
use InvalidArgumentException;
use Random\Randomizer;

/**
 * (μ/μ_w, λ)-CMA-ES — a faithful port of the standard algorithm as described in Nikolaus Hansen's own
 * simplified "purecma" reference implementation (deliberately published as a compact, portable reference
 * distinct from the full-featured `pycma` production library), not reimplemented from the original paper's
 * equations directly. Ported rather than invented from scratch specifically to minimize the chance of a
 * subtle sign/normalization bug in the step-size and covariance adaptation machinery — see this package's
 * README for the full reasoning, including why the eigendecomposition needed
 * ({@see SymmetricEigenDecomposition}) is a hand-rolled Jacobi solver rather than an external dependency.
 *
 * Deliberately simple relative to a production CMA-ES: eigendecomposition happens every generation
 * (skipping the usual "only every few generations" performance optimization, unnecessary at the
 * dimensionality this package's own tests and its origin project actually use — a handful to a few dozen
 * parameters) and there is no automatic restart/IPOP machinery — a fixed generation count is the only
 * stopping criterion, matching this project's "reviewable reference code over sophistication" bias and
 * this namespace's simple generation-count stopping rule elsewhere (DifferentialEvolutionAlgorithm).
 */
class CmaEsAlgorithm extends AbstractOptimizerAlgorithm
{
    /**
     * @var float
     */
    protected const DEFAULT_STEP_WIDTH = 0.3;

    /**
     * @var int
     */
    protected const DEFAULT_MAX_ITERATIONS = 200;

    /**
     * @var array<int, float>|null
     */
    protected ?array $initialMean = null;

    /**
     * @var \BlackboxOptimizer\Algorithm\Internal\SymmetricEigenDecomposition
     */
    protected SymmetricEigenDecomposition $eigenDecomposition;

    /**
     * @var \BlackboxOptimizer\Algorithm\Internal\VectorMath
     */
    protected VectorMath $vectorMath;

    /**
     * @param \BlackboxOptimizer\Algorithm\Internal\SymmetricEigenDecomposition|null $eigenDecomposition
     * @param \Random\Randomizer|null $randomizer
     * @param \BlackboxOptimizer\Algorithm\Internal\VectorMath|null $vectorMath
     */
    public function __construct(
        ?SymmetricEigenDecomposition $eigenDecomposition = null,
        ?Randomizer $randomizer = null,
        ?VectorMath $vectorMath = null,
    ) {
        parent::__construct($randomizer);
        $this->eigenDecomposition = $eigenDecomposition ?? new SymmetricEigenDecomposition();
        $this->vectorMath = $vectorMath ?? new VectorMath();
    }

    /**
     * Algorithm-specific setup, deliberately NOT part of {@see OptimizerAlgorithmInterface} — call before
     * optimize() to override the default; skipping this call entirely is fine too (the midpoint of the
     * problem's bounds is used instead, which requires every dimension to be finitely bounded).
     *
     * @param array<int, float>|null $initialMean Starting point "m0", same length/order as the problem's
     *   parameters. Null (the default) uses the midpoint of the problem's own bounds at optimize() time.
     *
     * @return $this
     */
    public function setInitialMean(?array $initialMean): static
    {
        $this->initialMean = $initialMean;

        return $this;
    }

    /**
     * {@inheritDoc}
     *
     * "λ" (population size) falls back to Hansen's classic default (4 + floor(3 * ln(n))) when
     * {@see setPopulationSize()} was never called — computed from the actual dimension count, which is
     * why it can't simply be a fixed default constant the way step width and max iterations are.
     *
     * @param \BlackboxOptimizer\Problem\ProblemInterface $problem
     *
     * @return \BlackboxOptimizer\Algorithm\OptimizationResult
     */
    public function optimize(ProblemInterface $problem): OptimizationResult
    {
        $this->resetTracking();
        [$lowerBounds, $upperBounds] = $this->extractBounds($problem);
        $n = count($lowerBounds);

        $strategy = $this->buildStrategyParameters($n);
        $mean = $this->initialMean ?? $this->midpoint($lowerBounds, $upperBounds);
        $sigma = $this->stepWidth ?? static::DEFAULT_STEP_WIDTH;
        $maxIterations = $this->maxIterations ?? static::DEFAULT_MAX_ITERATIONS;

        $covariance = $this->buildIdentity($n);
        $pathSigma = array_fill(0, $n, 0.0);
        $pathC = array_fill(0, $n, 0.0);
        [$eigenvectors, $eigenvalues] = $this->eigenSqrt($covariance);

        for ($generation = 1; $generation <= $maxIterations; $generation++) {
            $samples = $this->sampleOffspring($strategy['lambda'], $n, $mean, $sigma, $eigenvectors, $eigenvalues, $lowerBounds, $upperBounds);
            $values = [];

            foreach ($samples as $index => $sample) {
                $values[$index] = $this->evaluate($problem, $sample['x']);
            }

            asort($values);
            $rankedIndexes = array_keys($values);

            $newMean = $this->weightedRecombination($samples, $rankedIndexes, $strategy['weights'], $n);
            $samplingSigma = $sigma;
            $yMean = $this->vectorMath->scaleVector($this->vectorMath->subtractVectors($newMean, $mean), 1.0 / $samplingSigma);

            $pathSigma = $this->updatePathSigma($pathSigma, $yMean, $eigenvectors, $eigenvalues, $strategy);
            $sigma = $this->updateStepSize($sigma, $pathSigma, $strategy);

            $hSigma = $this->computeHeaviside($pathSigma, $strategy, $n, $generation);
            $pathC = $this->updatePathC($pathC, $yMean, $hSigma, $strategy);

            // Uses $samplingSigma (the step size actually used to draw THIS generation's samples), not the
            // just-updated $sigma above -- the rank-mu term's y_i = (x_i - m_old) / sigma_old must match
            // what was used to produce those same x_i in sampleOffspring().
            $covariance = $this->updateCovariance($covariance, $pathC, $samples, $rankedIndexes, $strategy, $hSigma, $samplingSigma, $mean, $n);

            $mean = $newMean;
            [$eigenvectors, $eigenvalues] = $this->eigenSqrt($covariance);

            $this->recordGenerationHistory();
        }

        return $this->buildResult();
    }

    /**
     * All the (μ/μ_w, λ)-CMA-ES strategy constants, computed once per optimize() call from the dimension
     * count and (if given) the configured population size — standard formulas, see Hansen's "The CMA
     * Evolution Strategy: A Tutorial" for the derivations; not reproduced here since this is a port, not a
     * re-derivation.
     *
     * @param int $n
     *
     * @return array<string, mixed>
     */
    protected function buildStrategyParameters(int $n): array
    {
        $lambda = $this->populationSize ?? (int)(4 + floor(3 * log($n)));
        $mu = (int)floor($lambda / 2);

        $rawWeights = [];

        for ($i = 1; $i <= $mu; $i++) {
            $rawWeights[$i - 1] = log($mu + 0.5) - log($i);
        }

        $weightSum = array_sum($rawWeights);
        $weights = array_map(static fn (float $weight): float => $weight / $weightSum, $rawWeights);

        $muEff = 1.0 / array_sum(array_map(static fn (float $weight): float => $weight ** 2, $weights));

        $cSigma = ($muEff + 2) / ($n + $muEff + 5);
        $dSigma = 1 + 2 * max(0.0, sqrt(($muEff - 1) / ($n + 1)) - 1) + $cSigma;
        $cc = (4 + $muEff / $n) / ($n + 4 + 2 * $muEff / $n);
        $c1 = 2 / (($n + 1.3) ** 2 + $muEff);
        $cMu = min(1 - $c1, 2 * ($muEff - 2 + 1 / $muEff) / (($n + 2) ** 2 + $muEff));
        $chiN = sqrt((float)$n) * (1 - 1 / (4 * $n) + 1 / (21 * $n ** 2));

        return [
            'lambda' => $lambda,
            'mu' => $mu,
            'weights' => $weights,
            'muEff' => $muEff,
            'cSigma' => $cSigma,
            'dSigma' => $dSigma,
            'cc' => $cc,
            'c1' => $c1,
            'cMu' => $cMu,
            'chiN' => $chiN,
        ];
    }

    /**
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @throws \InvalidArgumentException
     *
     * @return array<int, float>
     */
    protected function midpoint(array $lowerBounds, array $upperBounds): array
    {
        $mean = [];

        foreach ($lowerBounds as $index => $lowerBound) {
            if (is_infinite($lowerBound) || is_infinite($upperBounds[$index])) {
                throw new InvalidArgumentException('An initialMean must be given via setInitialMean() when any bound is infinite -- the midpoint of an unbounded dimension is undefined.');
            }

            $mean[$index] = ($lowerBound + $upperBounds[$index]) / 2.0;
        }

        return $mean;
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
     * @param array<int, array<int, float>> $covariance
     *
     * @return array{0: array<int, array<int, float>>, 1: array<int, float>} [eigenvectors, sqrt(eigenvalues)]
     *   -- the square roots are what sampling and the inverse-square-root transform both actually need.
     */
    protected function eigenSqrt(array $covariance): array
    {
        $decomposed = $this->eigenDecomposition->decompose($covariance);
        $sqrtEigenvalues = array_map(static fn (float $eigenvalue): float => sqrt(max($eigenvalue, 0.0)), $decomposed['eigenvalues']);

        return [$decomposed['eigenvectors'], $sqrtEigenvalues];
    }

    /**
     * @param int $lambda
     * @param int $n
     * @param array<int, float> $mean
     * @param float $sigma
     * @param array<int, array<int, float>> $eigenvectors
     * @param array<int, float> $sqrtEigenvalues
     * @param array<int, float> $lowerBounds
     * @param array<int, float> $upperBounds
     *
     * @return array<int, array{z: array<int, float>, y: array<int, float>, x: array<int, float>}>
     */
    protected function sampleOffspring(
        int $lambda,
        int $n,
        array $mean,
        float $sigma,
        array $eigenvectors,
        array $sqrtEigenvalues,
        array $lowerBounds,
        array $upperBounds,
    ): array {
        $samples = [];

        for ($k = 0; $k < $lambda; $k++) {
            $z = [];

            for ($i = 0; $i < $n; $i++) {
                $z[$i] = $this->standardNormal();
            }

            // y = B * (D .* z) -- scale the isotropic sample by the eigenvalues' square roots, THEN
            // rotate into the covariance's own basis. No transformByTranspose here: z is already
            // isotropic (N(0,I)), so there is nothing to project out of eigenspace first.
            $y = $this->vectorMath->matrixVectorMultiply($eigenvectors, $this->vectorMath->applyDiagonal($sqrtEigenvalues, $z));
            $x = $this->clamp($this->vectorMath->addVectors($mean, $this->vectorMath->scaleVector($y, $sigma)), $lowerBounds, $upperBounds);

            $samples[$k] = ['z' => $z, 'y' => $y, 'x' => $x];
        }

        return $samples;
    }

    /**
     * Box-Muller transform — a standard, simple way to draw from N(0,1) without depending on any
     * PHP extension beyond what {@see \Random\Randomizer} already provides.
     *
     * @return float
     */
    protected function standardNormal(): float
    {
        $u1 = max($this->randomizer->getFloat(0.0, 1.0), PHP_FLOAT_EPSILON);
        $u2 = $this->randomizer->getFloat(0.0, 1.0);

        return sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
    }

    /**
     * @param array<int, array{x: array<int, float>}> $samples
     * @param array<int, int> $rankedIndexes Sample indexes, best (lowest value) first.
     * @param array<int, float> $weights
     * @param int $n
     *
     * @return array<int, float>
     */
    protected function weightedRecombination(array $samples, array $rankedIndexes, array $weights, int $n): array
    {
        $result = array_fill(0, $n, 0.0);

        foreach ($weights as $rank => $weight) {
            $sampleIndex = $rankedIndexes[$rank];

            foreach ($samples[$sampleIndex]['x'] as $dimension => $value) {
                $result[$dimension] += $weight * $value;
            }
        }

        return $result;
    }

    /**
     * @param array<int, float> $pathSigma
     * @param array<int, float> $yMean
     * @param array<int, array<int, float>> $eigenvectors
     * @param array<int, float> $sqrtEigenvalues
     * @param array<string, mixed> $strategy
     *
     * @return array<int, float>
     */
    protected function updatePathSigma(array $pathSigma, array $yMean, array $eigenvectors, array $sqrtEigenvalues, array $strategy): array
    {
        $inverseSqrtTransformed = $this->applyInverseSqrtCovariance($yMean, $eigenvectors, $sqrtEigenvalues);
        $factor = sqrt($strategy['cSigma'] * (2 - $strategy['cSigma']) * $strategy['muEff']);

        $decayed = $this->vectorMath->scaleVector($pathSigma, 1 - $strategy['cSigma']);
        $boosted = $this->vectorMath->scaleVector($inverseSqrtTransformed, $factor);

        return $this->vectorMath->addVectors($decayed, $boosted);
    }

    /**
     * C^(-1/2) * v = B * D^(-1) * B^T * v, using the already-known eigendecomposition of C.
     *
     * @param array<int, float> $vector
     * @param array<int, array<int, float>> $eigenvectors
     * @param array<int, float> $sqrtEigenvalues
     *
     * @return array<int, float>
     */
    protected function applyInverseSqrtCovariance(array $vector, array $eigenvectors, array $sqrtEigenvalues): array
    {
        $transformed = $this->vectorMath->transformByTranspose($eigenvectors, $vector);

        foreach ($transformed as $index => $value) {
            $transformed[$index] = $value / max($sqrtEigenvalues[$index], PHP_FLOAT_EPSILON);
        }

        return $this->vectorMath->matrixVectorMultiply($eigenvectors, $transformed);
    }

    /**
     * @param float $sigma
     * @param array<int, float> $pathSigma
     * @param array<string, mixed> $strategy
     *
     * @return float
     */
    protected function updateStepSize(float $sigma, array $pathSigma, array $strategy): float
    {
        $pathNorm = $this->vectorMath->vectorNorm($pathSigma);

        return $sigma * exp(($strategy['cSigma'] / $strategy['dSigma']) * ($pathNorm / $strategy['chiN'] - 1));
    }

    /**
     * @param array<int, float> $pathSigma
     * @param array<string, mixed> $strategy
     * @param int $n
     * @param int $generation
     *
     * @return float 1.0 or 0.0 -- stalls the covariance path update when the step-size path has grown
     *   unusually large, a standard CMA-ES stabilization against premature covariance blow-up.
     */
    protected function computeHeaviside(array $pathSigma, array $strategy, int $n, int $generation): float
    {
        $pathNorm = $this->vectorMath->vectorNorm($pathSigma);
        $expectedNorm = sqrt(1 - (1 - $strategy['cSigma']) ** (2 * $generation)) * $strategy['chiN'];
        $threshold = (1.4 + 2 / ($n + 1)) * $strategy['chiN'];

        return $pathNorm / max($expectedNorm, PHP_FLOAT_EPSILON) < $threshold ? 1.0 : 0.0;
    }

    /**
     * @param array<int, float> $pathC
     * @param array<int, float> $yMean
     * @param float $hSigma
     * @param array<string, mixed> $strategy
     *
     * @return array<int, float>
     */
    protected function updatePathC(array $pathC, array $yMean, float $hSigma, array $strategy): array
    {
        $factor = $hSigma * sqrt($strategy['cc'] * (2 - $strategy['cc']) * $strategy['muEff']);

        $decayed = $this->vectorMath->scaleVector($pathC, 1 - $strategy['cc']);
        $boosted = $this->vectorMath->scaleVector($yMean, $factor);

        return $this->vectorMath->addVectors($decayed, $boosted);
    }

    /**
     * @param array<int, array<int, float>> $covariance
     * @param array<int, float> $pathC
     * @param array<int, array{x: array<int, float>}> $samples
     * @param array<int, int> $rankedIndexes
     * @param array<string, mixed> $strategy
     * @param float $hSigma
     * @param float $sigma
     * @param array<int, float> $mean
     * @param int $n
     *
     * @return array<int, array<int, float>>
     */
    protected function updateCovariance(
        array $covariance,
        array $pathC,
        array $samples,
        array $rankedIndexes,
        array $strategy,
        float $hSigma,
        float $sigma,
        array $mean,
        int $n,
    ): array {
        $c1 = $strategy['c1'];
        $cMu = $strategy['cMu'];
        $deltaH = (1 - $hSigma) * $strategy['cc'] * (2 - $strategy['cc']);

        $retained = 1 - $c1 - $cMu + $c1 * $deltaH;
        $updated = [];

        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $updated[$i][$j] = $retained * $covariance[$i][$j] + $c1 * $pathC[$i] * $pathC[$j];
            }
        }

        foreach ($strategy['weights'] as $rank => $weight) {
            $sampleIndex = $rankedIndexes[$rank];
            $x = $samples[$sampleIndex]['x'];
            $y = $this->vectorMath->scaleVector($this->vectorMath->subtractVectors($x, $mean), 1.0 / $sigma);

            for ($i = 0; $i < $n; $i++) {
                for ($j = 0; $j < $n; $j++) {
                    $updated[$i][$j] += $cMu * $weight * $y[$i] * $y[$j];
                }
            }
        }

        return $updated;
    }
}
