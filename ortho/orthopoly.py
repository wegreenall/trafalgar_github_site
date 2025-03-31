import torch
from typing import Dict
import matplotlib.pyplot as plt
from ortho.measure import MaximalEntropyDensity

# from time import sleep
torch.set_default_tensor_type(torch.DoubleTensor)

"""
Contains classes for construction of orthogonal polynomials, as well as methods
for getting the measure/weight function.
"""


class MultivariateOrthogonalPolynomial:
    def __init__(
        self, order: int, betas: torch.Tensor, gammas: torch.Tensor, leading=1
    ):
        """
        A class representing a multivariate orthogonal polynomial
        """
        self.order = order

        def __call__(x: torch.Tensor, deg: int, params: dict) -> torch.Tensor:
            return


class OrthogonalPolynomial:
    def __init__(
        self, order: int, betas: torch.Tensor, gammas: torch.Tensor, leading=1
    ):
        """
        A system of orthogonal polynomials has the property that:

        int P_j(x) P_k(x) dx = δ_ik

        The three-term recursion for orthogonal polynomials is written:

            P_n+1(x) = (β_{n+1} - x) P_n(x) - γ_{n+1} P_{n-1} (x)

        with:
            P_0(x) = 1 and P_1(x) = x

        Construction then of an arbitrary orthogonal polynomial sequence should
        involve setting the betas and the gammas appropriate for a given
        application
        """
        self.order = order

        if len(betas) < order or len(gammas) < order:
            raise TypeError(
                "Not enough betas or gammas for the order %d" % (order)
            )
        self.set_betas(betas)
        self.set_gammas(gammas)
        self.leading = leading
        return

    def __call__(
        self, x: torch.Tensor, deg: int, params: dict
    ) -> torch.Tensor:
        """
        This produces the value of the polynomial by constructing the
        fundamental recursion.
        """
        if deg > self.order:
            raise ValueError("The order of this polynomial is not high enough")
        if deg == 0:
            result = torch.ones(x.shape)
        elif deg == 1:
            result = self.leading * (x - self.betas[0])
        else:
            result = (self.leading * x - self.betas[deg - 1]) * self(
                x, deg - 1, params
            ) - self.gammas[deg - 1] * self(x, deg - 2, params)
        return result

    def get_order(self):
        """
        Returns the order of this polynomial.
        """
        return self.order

    def set_betas(self, betas):
        """
        Setter for betas on orthogonal polynomial.
        """
        # assert (betas >= 0).all(), "Please make sure all betas are positive"
        self.betas = betas

    def set_gammas(self, gammas):
        """
        Setter for gammas on orthogonal polynomial.
        Validates that the initial gamma is set to 1; they also all need
        to be positive so that the corresponding linear moment functional
        is positive definite.
        """
        # assert gammas[0] == 1, "Please make sure gammas[0] = 1"
        # print("In set_gammas in OrthogonalPolynomial, and γ_0 = ", gammas[0])
        assert (gammas >= 0).all(), "Please make sure gammas > 0"
        self.gammas = gammas

    def get_betas(self):
        return self.betas

    def get_gammas(self):
        return self.gammas

    def get_jacobi_operator(self):
        """
        Returns the Jacobi operator for the orthogonal polynomial.
        """
        matrix = torch.zeros((self.order, self.order))
        for i in range(self.order):
            matrix[i, i] = self.betas[i]
            if i > 0:
                matrix[i, i - 1] = torch.sqrt(self.gammas[i])
            if i < self.order - 1:
                matrix[i, i + 1] = torch.sqrt(self.gammas[i + 1])

        return matrix


class OrthonormalPolynomial(OrthogonalPolynomial):
    """
    Subclasses the OrthogonalPolynomial to allow for normalisation.
    The normalising constant for a given orthonormal basis is the reciprocal of
    the product of the gammas from the Favard recursion.
    """

    def __call__(self, x: torch.Tensor, deg: int, params: dict):
        result = super().__call__(x, deg, params)
        """
        WARNING: TESTING for DEG+1 AS THE PRODUCT FOR THE ORTHONORMALISATION
        """
        # normalising_coefficient = torch.prod(self.gammas[0 : deg + 1])
        normalising_coefficient = torch.prod(self.gammas[0:deg])
        return result / torch.sqrt(normalising_coefficient)


class OrthogonalBasisFunction(OrthogonalPolynomial):
    def __init__(
        self, order, betas: torch.Tensor, gammas: torch.Tensor, leading=1
    ):
        """
        Subclasses the OrthogonalPolynomialFunction
        to multiply by the weight function.
        """
        super().__init__(order, betas, gammas, leading)
        # moment_net = CatNet(order, betas, gammas)
        self.med = MaximalEntropyDensity(order, betas, gammas)
        pass

    def __call__(self, x: torch.Tensor, deg: int, params: dict):
        """
        Calls first the orthogonal polynomial component, and then
        builds the weight function component.
        """
        ortho_poly_term = super().__call__(x, deg, params)
        return ortho_poly_term  # * med_term

    def get_weight(self, x: torch.Tensor, params: dict):
        return torch.sqrt(self.med(x, self.betas, self.gammas))


class SymmetricOrthogonalPolynomial(OrthogonalPolynomial):
    def __init__(self, order: int, gammas: torch.Tensor):
        """
        If Pn is symmetric, i.e. Pn(-x) = (-1)^n Pn(x),
        then there exist coefficients γ_n != 0 for n>=1, s.t.
                P_{n+1} = xP_n(x) - γ_nP_{n-1}

        with initial conditions P_0(x) = 1 and P_1(x) = x
        This is equivalent to a polynomial sequence with β_n = 0 for each n
        """
        betas = torch.zeros(order)
        super().__init__(order, betas, gammas)


class SymmetricOrthonormalPolynomial(OrthonormalPolynomial):
    def __init__(self, order: int, gammas: torch.Tensor):
        """
        If Pn is symmetric, i.e. Pn(-x) = (-1)^n Pn(x),
        then there exist coefficients γ_n != 0 for n>=1, s.t.
                P_{n+1} = xP_n(x) - γ_nP_{n-1}

        with initial conditions P_0(x) = 1 and P_1(x) = x
        This is equivalent to an orthogonal polynomial sequence with β_n = 0
        for each n.
        """
        betas = torch.zeros(order)
        super().__init__(order, betas, gammas)


class OrthogonalPolynomialSeries:
    def __init__(self, coefficients, order, betas, gammas, leading=1):
        """Using Clenshaw recursion, we can build a series based on orthogonal
        polynomials by utilising Favard's three-term recursion theorem, and
        the Clenshaw recursion algorithm.

        From wikipedia:
            The Clenshaw recursion algorithm computes the weighted sum of a
            finite series of functions φ_k(x):
                S(x) = sum_{k=0}^n a_k φ_k(x)
            where:
                φ_{k+1}(x) = Θ_k(x) φ_k(x) + δ_k(x) φ_{k-1}

            First we compute a sequence {b_k}:
                b_{n+2}(x) = b_{n+1}(x) = 0
                b_k(x) = a_k + θ_k(x)b_{k+1} + δ_{k+1}(x) b_{k+2}(x)

            Then, once one has {b_k}, the sum can be written:
                S(x) = φ_0(x)(a_0 + β_1(x) b_2(x)) + φ_1(x)b_1(x)

        Since by the Favard's theorem, for orthogonal polynomials
            φ_1(x) = x and φ_0(x) = 1

        We just use (a_0 + β_1(χ)β_2(χ)) + x b_1(x)
        """
        self.coefficients = coefficients  # a_k
        self.order = order
        self.betas = betas
        self.gammas = gammas

        self.thetas = self._get_clenshaw_thetas()
        self.deltas = self._get_clenshaw_deltas()
        # breakpoint()
        self._compute_b_terms()
        pass

    def __call__(self, x):
        if self.order == 0:
            return self.coefficients[0] * torch.ones(x.shape)
        if self.order == 1:
            return (
                self.coefficients[0] * torch.ones(x.shape)
                + (x - self.betas[0]) * self.coefficients[1]
            )
        else:
            f_0_term = self.coefficients[0] + self.deltas[1](x) * self.b_terms[
                2
            ](x)
            f_1_term = (x - self.betas[0]) * self.b_terms[1](x)
            return f_0_term + f_1_term

    def _get_clenshaw_thetas(self):
        """
        Returns the sequence of θ (note the notation in the docstring for
        this class's __init__ function).

        The θ in this notation represent the _function_ of x that is the
        coefficient for P_n in the recurision. In the Favard case, we have:
                                θ_n(x) = (x - β_n)
        """
        thetas = list()
        for beta in self.betas:
            thetas.append(lambda x, n=beta: x - n)
        return thetas

    def _get_clenshaw_deltas(self):
        """
        Returns the sequence of δ (note the notation in the docstring for
        this class's __init__ function).

        The δ in this notation represent the _function_ of x that is the
        coefficient for P_n in the recurision. In the Favard case, we have:
                                δ_n(x) = ( - γ_n)
        """
        deltas = list()
        for gamma in self.gammas:
            print(gamma)
            deltas.append(lambda x, n=gamma: -n * torch.ones(x.shape))
        return deltas

    def _compute_b_terms(self):
        """
        The Clenshaw algorithm requires construction of the sequence of b terms
        (note the notation in the docstring for
        this class's __init__ function).

        Having constructed the b terms, we can build the sum of the functions.
        """
        self.b_terms = list()

        def b(k):
            if k == self.order + 2 or k == self.order + 1:
                return lambda x: torch.zeros(x.shape)
            elif k == self.order:
                return lambda x: self.coefficients[k] * torch.ones(x.shape)
            elif k == self.order - 1:
                return (
                    lambda x: self.coefficients[k]
                    + self.thetas[k](x) * b(k + 1)(x)
                    + self.deltas[k + 1](x) * b(k + 2)(x)
                )
            elif k <= self.order - 2:
                return (
                    lambda x: self.coefficients[k]
                    + self.thetas[k](x) * b(k + 1)(x)
                    + self.deltas[k + 1](x) * b(k + 2)(x)
                )

        for k in range(self.order + 2):
            self.b_terms.append(b(k))


def get_measure_from_poly(
    poly: OrthogonalPolynomial, initial_moment=1
) -> list:
    """
    Accepts a polynomial and produces from it the density of an absolutely
    continuous measure w.r.t which the polynomial is orthogonal.

    The Favard's theorem states the recurrence relation for the orthogonal
    polynomial sequence to be:
            P_{n+1} = (x-β_{n+1})P_n - γ_{n+1}P_{n-1}
        so: P_{n+1} - (x-β_{n+1})P_n - γ_{n+1}P_{n-1} = 0

    This is the same as:
            xP_n(x) = P_{n+1}(x) + β_nP_n(x) + γ_n P_{n-1}(x)

    Applying the linear moment functional to this relation gives the moment
    recursion implemented here.

    The moments of the measure, which uniquely define it, have the following
    property: for each k, <L, P_k(x)> is 0. However, this implies a sequence
    of relations for the moments w.r.t the coefficients of the polynomial:

        <L, 1> = μ_0      =1 for a probability distribution, the
                          Poisson-equivalent intensity for a Poisson
                          measure

        <L, P_1> = μ_1 - β_0μ_0 = 0
                   => μ_1 = β_0 μ_0   i.e. the mean is β_0.

        <L, P_2> = μ_2 - <L, xP_1(x)> + β_2<L, xP_1> + γ_{2}<L, P_1>
                   => μ_2 =

    The above define a sequence of relations that implicitly set the moment:

        <L, P_{k+1}> = μ_{Κ+1} - {other terms} = 0

    This function uses this implicit relation to acquire values for the moments
    given the betas and gammas on the input orthogonal polynomial.

    Once the moments are acquired, it is possible to construct the
    measure via an inverse laplace transform on the moment-generating
    function
        Σ_ι  μ_i x^i / i!

    """

    raise NotImplementedError


# def get_poly_from_moments(
# moments: torch.Tensor,
# order: int,
# ) -> "SymmetricOrthonormalPolynomial":
# """
# Accepts a list of moment values and produces from it a
# SymmetricOrthogonalPolynomial.

# The standard recurrence for an orthogonal polynomial series has n equations
# and 2n unknowns - this means that it is not feasible to construct, from a
# given sequence of moments, an orthogonal polynomial sequence that is
# orthogonal     w.r.t the given moments. However, if we impose
# symmetricality, we can build a sequence of symmetric orthogonal polynomials
# from a given set of moments.
# """
# gammas = torch.zeros(order)

# # to construct the polynomial from the sequnce of moments, utilise the
# # sequence of equations:
# gammas = get_gammas_from_moments(moments, order)

# return SymmetricOrthonormalPolynomial(order, gammas)
if __name__ == "__main__":
    check_ortho_poly = True
    check_sym_poly = True
    check_orthopolyseries = True
    from torch import distributions as D

    order = 20
    betas = D.Exponential(2.0).sample([order])
    gammas = D.Exponential(2.0).sample([order])
    gammas[0] = 1

    x = torch.linspace(-3, 3, 100)
    params: Dict = dict()
    if check_ortho_poly:
        poly = OrthogonalPolynomial(order, betas, gammas)

        func = poly(x, 3, params)
        plt.plot(x, func)
        plt.show()

    if check_sym_poly:
        sym_poly = SymmetricOrthogonalPolynomial(order, gammas)
        sym_func = sym_poly(x, 2, params)
        plt.plot(x, sym_func)
        plt.show()

    if check_orthopolyseries:
        betas = torch.zeros(order + 1)
        gammas = torch.linspace(0, order, order + 1)
        # gammas[0] = 0

        standard_orthopoly = OrthogonalPolynomial(order + 1, betas, gammas)
        summed_poly_series = torch.zeros(x.shape)
        for i in range(order + 1):
            # summed_poly_series +=
            print("polynomial order:", i)
            summed_poly_series += standard_orthopoly(x, i, dict())

        # plt.plot(x, summed_poly_series, color=black")
        print("number of summed terms:", i)

        # now do it with the OrthopolySeries
        orthopoly_series = OrthogonalPolynomialSeries(
            torch.ones(order + 1), order, betas, gammas
        )
        values = orthopoly_series(x)
        plt.plot(x, values - summed_poly_series, color="red")
        plt.show()
