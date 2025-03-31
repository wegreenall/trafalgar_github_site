import torch
import torch.distributions as D
from ortho.builders import (
    OrthoBuilder,
    OrthoBuilderState,
    get_weight_function_from_sample,
    get_moments_from_function,
    get_moments_from_sample,
    get_betas_from_moments,
    get_gammas_from_moments,
    get_gammas_betas_from_moments,
    get_gammas_betas_from_moments_gautschi,
    get_gammas_betas_from_modified_moments_gautschi,
    integrate_function,
    sample_from_function,
    get_poly_from_moments,
    get_gammas_from_sample,
    get_poly_from_sample,
    get_orthonormal_basis,
    get_orthonormal_basis_from_sample,
)
from ortho.basis_functions import Basis
from ortho.orthopoly import (
    OrthogonalPolynomial,
    OrthonormalPolynomial,
    SymmetricOrthogonalPolynomial,
    SymmetricOrthonormalPolynomial,
)
from ortho.polynomials import ProbabilistsHermitePolynomial
import math
from torch.quasirandom import SobolEngine

# from ortho.measure import coeffics_list
import unittest
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_printoptions(precision=10)


def function_for_sampling(x: torch.Tensor):
    """
    It's... the Gaussian!
    """
    return (1 / (math.sqrt(math.pi * 2))) * torch.exp(-(x ** 2) / 2)


def ks_test(
    sample: torch.Tensor,
    distribution: D.Distribution,
    input_space=torch.linspace(-5, 5, 500),
) -> torch.Tensor:
    """
    Should calculate the empirical cdf of a sample for the purpose
    of testing using the Kolmogorov-Smirnov test.
    """
    cdfs = distribution.cdf(input_space)
    sample_size = sample.shape[0]
    repeated_input_space = input_space.repeat(sample_size, 1)
    repeated_sample = sample.repeat(len(input_space), 1).t()
    ones = 1.0 * ((repeated_input_space - repeated_sample) > 0)
    empirical_cdfs = torch.mean(ones, dim=0)
    result = torch.max(torch.abs(empirical_cdfs - cdfs))
    return result


# def gauss_moment(n: int) -> int:
# """
# returns the n-th moment of a standard normal gaussian distribution.
# """
# if n == 0:
# return 1
# if n % 2 == 0:
# return double_fact(n - 1)
# else:
# return 0


def gauss_moment(order: int) -> list:
    """
    returns the n-th moment of a standard normal gaussian distribution.
    """
    if order == 0:
        return 1
    if order == 1:
        return 0
    else:
        return (order - 1) * gauss_moment(order - 2)


def double_fact(n: int) -> int:
    """
    Evaluates the double factorial of the input integer n
    """
    # assert n % 2 == 0, "n is not even!"
    # p = n - 1
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return n * double_fact(n - 2)


class TestOrthoBuilderGetPolynomials(unittest.TestCase):
    def setUp(self):
        self.order = 5
        self.builder = OrthoBuilder(self.order)
        self.sample_size = 1000
        self.sample = D.Normal(0.0, 1.0).sample((self.sample_size,))
        self.moments = torch.Tensor([1.0] * (2 * self.order + 1))

        self.betas = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        self.gammas = torch.Tensor([1.0, 1.0, 2.0, 3.0, 4.0])

        self.orthogonal_polynomial = OrthogonalPolynomial(
            self.order, self.betas, self.gammas
        )

    def test_get_orthogonal_polynomial(self):
        orthogonal_polynomial = self.builder.set_betas_and_gammas(
            self.betas, self.gammas
        ).get_orthogonal_polynomial()
        self.assertTrue(orthogonal_polynomial.order == self.order)
        self.assertTrue(
            isinstance(orthogonal_polynomial, OrthogonalPolynomial)
        )

    def test_get_orthonormal_polynomial(self):
        orthonormal_polynomial = self.builder.set_betas_and_gammas(
            self.betas, self.gammas
        ).get_orthonormal_polynomial()
        self.assertTrue(orthonormal_polynomial.order == self.order)
        self.assertTrue(
            isinstance(orthonormal_polynomial, OrthonormalPolynomial)
        )

    def test_get_orthonormal_basis(self):
        orthonormal_basis = (
            self.builder.set_betas_and_gammas(self.betas, self.gammas)
            .set_weight_function(lambda x: torch.exp(-(x ** 2) / 2))
            .get_orthonormal_basis()
        )
        self.assertTrue(orthonormal_basis.order == self.order)
        self.assertTrue(isinstance(orthonormal_basis, Basis))

    def test_get_symmetric_orthogonal_polynomial(self):
        symmetric_orthogonal_polynomial = self.builder.set_betas_and_gammas(
            self.betas, self.gammas
        ).get_symmetric_orthogonal_polynomial()
        self.assertTrue(symmetric_orthogonal_polynomial.order == self.order)
        self.assertTrue(
            isinstance(
                symmetric_orthogonal_polynomial, SymmetricOrthogonalPolynomial
            )
        )

    def test_get_symmetric_orthonormal_polynomial(self):
        symmetric_orthonormal_polynomial = self.builder.set_betas_and_gammas(
            self.betas, self.gammas
        ).get_symmetric_orthonormal_polynomial()
        self.assertTrue(symmetric_orthonormal_polynomial.order == self.order)
        self.assertTrue(
            isinstance(
                symmetric_orthonormal_polynomial,
                SymmetricOrthonormalPolynomial,
            )
        )

        pass


class TestOrthoBuilderStatesErrors(unittest.TestCase):
    def setUp(self):
        self.order = 5
        self.builder = OrthoBuilder(self.order)
        self.sample_size = 1000
        self.sample = D.Normal(0.0, 1.0).sample((self.sample_size,))
        self.moments = torch.Tensor([1.0] * (2 * self.order + 1))

        self.betas = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        self.gammas = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.orthogonal_polynomial = OrthogonalPolynomial(
            self.order, self.betas, self.gammas
        )

    def test_set_sample_state(self):
        self.builder.set_sample(self.sample)
        self.assertEqual(self.builder.state, OrthoBuilderState.BETAS_GAMMAS)

    def test_set_moments(self):
        self.assertEqual(self.builder.state, OrthoBuilderState.EMPTY)
        self.builder.set_moments(self.moments)
        self.assertEqual(self.builder.state, OrthoBuilderState.BETAS_GAMMAS)

    def test_error_state_sample(self):
        self.builder.set_sample(self.sample)
        with self.assertRaises(ValueError):
            self.builder.set_betas_and_gammas(self.betas, self.gammas)

    def test_error_state_moments(self):
        self.builder.state = OrthoBuilderState.SAMPLE
        with self.assertRaises(AssertionError):
            self.builder.set_moments(self.moments)

    def test_error_orthogonal_polynomial_generation(self):
        self.assertEqual(self.builder.state, OrthoBuilderState.EMPTY)
        with self.assertRaises(ValueError):
            self.builder.get_orthogonal_polynomial()

    def test_error_orthonormal_polynomial_generation(self):
        self.assertEqual(self.builder.state, OrthoBuilderState.EMPTY)
        with self.assertRaises(ValueError):
            self.builder.get_orthonormal_polynomial()

    def test_error_orthonormal_basis_generation(self):
        self.assertEqual(self.builder.state, OrthoBuilderState.EMPTY)
        with self.assertRaises(ValueError):
            self.builder.get_orthonormal_basis()

    def test_error_symmetric_orthogonal_polynomial_generation(self):
        self.assertEqual(self.builder.state, OrthoBuilderState.EMPTY)
        with self.assertRaises(ValueError):
            self.builder.get_symmetric_orthogonal_polynomial()

    def test_error_symmetric_orthonormal_polynomial_generation(self):
        self.assertEqual(self.builder.state, OrthoBuilderState.EMPTY)
        with self.assertRaises(ValueError):
            self.builder.get_symmetric_orthonormal_polynomial()

    @unittest.skip("Not implemented yet - add OrthogonalPolynomial")
    def test_modified_moments(self):
        # (
        # self.builder.
        # .set_moments(self.moments)
        # )
        (
            self.builder.set_modifying_polynomial(
                self.orthogonal_polynomial
            ).set_moments(self.moments)
        )
        self.assertEqual(self.builder.state, OrthoBuilderState.BETAS_GAMMAS)
        # self.builders.


class TestBuilders(unittest.TestCase):
    def setUp(self):
        self.order = 16
        distribution = D.Normal(0.0, 1.0)
        sample_size = 100000
        self.sample = distribution.sample((sample_size,))
        sobol = SobolEngine(dimension=1)
        base_sample = sobol.draw(sample_size)
        self.fixed_sample = D.Normal(0.0, 1.0).icdf(base_sample).squeeze()[2:]

        self.end_point = torch.tensor(10.0)
        fineness = 1000
        self.input_points = torch.linspace(
            -self.end_point, self.end_point, fineness
        )

        normal_moments = [
            gauss_moment(order) for order in range(2 * self.order + 2)
        ]
        self.normal_moments = torch.Tensor(normal_moments)

        self.prob_polynomials = [
            lambda x: torch.ones(x.shape),
            lambda x: x,
            lambda x: x ** 2 - 1,
            lambda x: x ** 3 - 3 * x,
            lambda x: x ** 4 - 6 * x ** 2 + 3,
            lambda x: x ** 5 - 10 * x ** 3 + 15 * x,
        ]

        self.weight_function = lambda x: torch.exp(-(x ** 2) / 2)
        self.example_betas = torch.cat(
            (
                torch.Tensor([1.0]),
                # torch.linspace(1.0, self.order - 1, self.order - 1)
                torch.zeros(self.order - 1),
            )
        )
        self.example_gammas = torch.cat(
            (
                torch.Tensor([1.0]),
                torch.linspace(1.0, self.order - 1, self.order - 1),
            )
        )

    def test_get_betas_from_moments(self):
        moments = self.normal_moments
        betas = get_betas_from_moments(moments, self.order)
        self.assertTrue(
            torch.allclose(
                betas,
                torch.zeros(self.order),
                1e-02,
            )
        )

    def test_get_gammas_from_moments(self):
        """
        Tests whether the gammas are correct.
        """
        moments = self.normal_moments
        gammas = get_gammas_from_moments(moments, self.order)
        # print("Gammas from moments:", gammas)
        self.assertTrue(
            torch.allclose(
                gammas,
                torch.cat(
                    (
                        torch.Tensor([1.0]),
                        torch.linspace(1.0, self.order - 1, self.order - 1),
                    )
                ),
                1e-02,
            )
        )

    @unittest.skip("Not close enough due to statistical noise")
    def test_get_moments_from_sample(self):
        moments = get_moments_from_sample(self.sample, self.order)
        # self.assertEqual(moments.shape, torch.Size([self.order + 1]))
        # self.assertTrue(torch.allclose(moments, self.normal_moments))
        self.assertTrue(moments.shape == torch.Size([self.order]))

    def test_get_moments_from_sample_shape(self):
        moments = get_moments_from_sample(self.sample, self.order)
        self.assertEqual(moments.shape, torch.Size([self.order]))

    @unittest.skip("bad example")
    def test_get_gammas_from_sample(self):
        gammas = get_gammas_from_sample(self.fixed_sample, self.order)  # [1:]
        # breakpoint()
        # print(gammas)

        self.assertTrue(torch.allclose(gammas, self.example_gammas), 5e-2)

    def test_get_gammas_betas_from_moments(self):
        betas, gammas = get_gammas_betas_from_moments(
            self.normal_moments, self.order
        )
        betas_2 = get_betas_from_moments(self.normal_moments, self.order)
        gammas_2 = get_gammas_from_moments(self.normal_moments, self.order)
        self.assertTrue(torch.allclose(betas, betas_2))
        self.assertTrue(torch.allclose(gammas, gammas_2))

    def test_get_gammas_betas_from_moments_gautschi(self):
        betas, gammas = get_gammas_betas_from_moments(
            self.normal_moments, self.order
        )
        betas_2, gammas_2 = get_gammas_betas_from_moments_gautschi(
            self.normal_moments[: 2 * self.order], self.order
        )
        self.assertTrue(torch.allclose(betas, betas_2))
        self.assertTrue(torch.allclose(gammas, gammas_2))

    def test_get_gammas_betas_from_modified_moments_gautschi(self):
        polynomial = ProbabilistsHermitePolynomial(2 * self.order)
        moments = torch.zeros(2 * self.order)
        moments[0] = 1
        betas, gammas = get_gammas_betas_from_modified_moments_gautschi(
            moments,
            self.order,
            polynomial,
        )
        # Check that the betas are those for the Normal distribution (Hermite polynomials)
        self.assertTrue(torch.allclose(betas, torch.zeros(betas.shape)))

        # Check that the gammas are those for the Normal distribution (Hermite polynomials)
        self.assertTrue(
            torch.allclose(
                gammas,
                torch.cat(
                    (
                        torch.Tensor([1.0]),
                        torch.linspace(1.0, self.order - 1, self.order - 1),
                    )
                ),
                1e-02,
            )
        )

    def test_integrate_function(self):
        integral = integrate_function(
            function_for_sampling,
            self.end_point,
            (1 / (math.sqrt(math.pi * 2))),
        )
        self.assertTrue(
            torch.allclose(integral, torch.tensor(1.0), rtol=1e-01)
        )

    def test_sample_from_function(self):
        new_sample = sample_from_function(
            function_for_sampling,
            self.end_point,
            (1 / (math.sqrt(math.pi * 2))),
            sample_size=int(10e4),
        )
        ks_test_statistic = ks_test(new_sample, D.Normal(0.0, 1.0))
        self.assertTrue(
            torch.allclose(ks_test_statistic, torch.tensor(0.0), atol=1e-02)
        )

    @unittest.skip("bad example")
    def test_get_orthonormal_basis_from_sample(self):
        basis = get_orthonormal_basis_from_sample(
            self.sample, self.weight_function, self.order
        )
        self.assertTrue(isinstance(basis, Basis))

    @unittest.skip("Not implemented")
    def test_get_orthonormal_basis(self):
        basis = get_orthonormal_basis(
            self.example_betas,
            self.example_gammas,
            self.order,
            self.weight_function,
        )

    @unittest.skip("")
    def test_get_poly_from_moments(self):
        poly = get_poly_from_moments(self.normal_moments, self.order)
        for i in range(6):
            with self.subTest(i=i):
                polyvals = poly(self.input_points, i, None)
                true_polyvals = self.prob_polynomials[i](self.input_points)
                self.assertTrue(torch.allclose(polyvals, true_polyvals))

    @unittest.skip("")
    def test_get_poly_from_sample(self):
        poly = get_poly_from_sample(self.sample, self.order)
        for i in range(6):
            with self.subTest(i=i):
                polyvals = poly(self.input_points, i, None)
                true_polyvals = self.prob_polynomials[i](self.input_points)
                self.assertTrue(torch.allclose(polyvals, true_polyvals))


if __name__ == "__main__":
    unittest.main()
