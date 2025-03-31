import torch
from torch import nn
from torch.nn import Module
import torch.distributions as D
from typing import Tuple
import math
from ortho.roots import get_roots, get_polynomial_max
from ortho.utils import integrate_function

# from ortho.orthopoly import OrthogonalPolynomial
# from ortho.inverse_laplace import build_inverse_laplace_from_moments
# from typing import Callable
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(linewidth=200)


class MaximalEntropyDensity:
    """
    A class constructing the maximal entropy density corresponding to
    the moments implied by sequences β, γ.

    This is feasible since, there is a unique solution for a given sequence of
    moments {1, μ_1, μ_2, ...), such that the Hankel determinant is positive
    (i.e. they come from an OPS as given by Favard's theorem with  γ_n > 0 for
    n >= 1).

    To get this, we take e^(-m'M^{_1}x).
    where m is the vector of moments 1 to k, and M is the Hankel matrix of
    moments from 2 to 2k; x is the polynomial up to order k.
        i.e. {1, x, ..., x^k}

    The moment_net output should be equal in size to 2 * order.
    """

    def __init__(self, order: int, betas: torch.Tensor, gammas: torch.Tensor):
        assert betas.shape[0] == 2 * order, "Please provide 2 * order betas"
        assert gammas.shape[0] == 2 * order, "Please provide 2 * order gammas"
        self.betas = betas
        self.gammas = gammas
        self.order = order

        self.lambdas = self._get_lambdas()
        self.normalising_constant = self._get_normalising_coefficient()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the maximal entropy density at input x.

        This is e^(-m'M{^-1}x + η), where x signifies the
        matrix of x evaluated at powers {0, 1, ..., m}, and η is set to
        normalise to one. We calculate this through a Monte Carlo integration
        at initialisation.

            -> μ, Μ    => _get_moments()
            -> δ = μ'Μ => _get_lambdas()
            -> x       => _get_poly_term()
            -> δ'x     => _unnormalised_log_weight_function()
        """
        log_weight = self._log_unnormalised_weight_function(x)

        if (log_weight == math.inf).any():
            print("Inf in polyterm...")
            breakpoint()

        weight_function = torch.exp(log_weight) * self.normalising_constant
        if (weight_function == math.inf).any() or (
            weight_function == -math.inf
        ).any():
            print("\nInf in weight function...\n")
            print(
                "Masking infs with 0. This is not the right thing to do\
                but it is what I can think of right now"
            )
            breakpoint()

        return weight_function

    def _log_unnormalised_weight_function(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluates the polynomial term using Horner's method - this
        may improve the floating point behaviour.
        """
        poly = Polynomial(self.lambdas)
        return -poly(x).squeeze()

    def _get_lambdas(self) -> torch.Tensor:
        """
        Solves the system m = Mλ to acquire the Lagrangian multipliers for the
        maximal entropy density given the sequence of moments.

        Given that the Lagrangian multiplier signs do not matter,
        and to ensure "good" behaviour of the weight function, we can
        take the Lagrange multipliers to be the negative of the absolute
        values for the result here - this will allow us to ensure the "right"
        behaviour of the weight function whilst imposing the constraints.
        """
        # get the moments vector
        moment_vector, moment_matrix = self._get_moments()

        # ratios_matrix R_{ij} = j/(i+1)
        ratios_matrix = torch.einsum(
            "i, j -> ij",
            1 / torch.linspace(2, self.order + 1, self.order),
            torch.linspace(1, self.order, self.order),
        )
        system_matrix = ratios_matrix * moment_matrix
        lambdas = torch.linalg.solve(system_matrix, moment_vector)
        assert torch.allclose(system_matrix @ lambdas, moment_vector)
        return torch.cat([torch.Tensor([0.0]), lambdas])

    def _get_moments(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Acquires the moments, starting from the first (not the 0-th, trivially
        equal to 1) for the given betas and gammas, via the Catalan
        matrix formulation due to Aigner (2006).

        Returns:
            - the moment vector m containing the moments up to the order + 1
            - the Hankel moment matrix containing all moments up to
              2 * order + 2
        """
        cat_matrix = torch.zeros(2 * self.order + 2, 2 * self.order + 2)
        cat_matrix[0, 1] = 1

        """
        The following loop builds out the Catalan matrix. it does this
        inside a padded matrix so that the recursion is simple for the edge 
        cases (by having terms "outside" the matrix be 0).
        """
        for n in range(1, 2 * self.order + 1):
            cat_matrix[n, 1:-1] = (
                cat_matrix[n - 1, :-2].clone()
                + cat_matrix[n - 1, 1:-1].clone() * self.betas
                + cat_matrix[n - 1, 2:].clone() * self.gammas
            )
        """
        We construct the moments up to 2*self.order and then use unfold
        to loop the same vector round to make the Hankel matrix.
        """
        moment_vector = cat_matrix[1:-1, 1]  # contains up to 2m
        moment_matrix = moment_vector.unfold(0, self.order, 1)[
            1:, :
        ]  # moment matrix

        return (
            moment_vector[: self.order],
            moment_matrix,
        )

    def _get_normalising_coefficient(
        self, sampling_size=100000
    ) -> torch.Tensor:
        """
        Returns the normalising coefficient under the given sequence.
        It does this by first calculating the maximiser of the weight
        function, and then approximating the integral via Laplace's method.
        The fact that the density is the maximum
        entropy and goes to zero means this calculation should be
        relatively accurate.
        """
        # get the function maximum
        weight_maximiser = self._get_unnormalised_maximiser()
        integral = integrate_function(
            lambda x: torch.exp(self._log_unnormalised_weight_function(x)),
            torch.Tensor([5.0]),
            weight_maximiser,
        )
        if -torch.log(integral) == math.inf:
            breakpoint()

        print(
            "weight maximiser:",
            weight_maximiser,
            "order:",
            self.order,
            "integral",
            integral,
            "integral reciprocal",
            1 / integral,
        )
        return 1 / integral

    def _get_unnormalised_maximiser(self):
        """
        Returns the maximiser of this function when not normalised -
        this allows for simple calculation of the normalising constant
        because to do so we need to build the integral of the function.
        """
        coeffics = -(self.lambdas)
        poly_max = get_polynomial_max(coeffics, self.order + 1)
        return torch.exp(poly_max)

    def set_betas(self, betas):
        self.betas = betas

    def set_gammas(self, gammas):
        # print("In set_gammas in MaximalEntropyDensity, and γ_0 = ", gammas[0])
        assert (gammas > 0).all(), "Please ensure gammas are non-negative."
        self.gammas = gammas


class Polynomial:
    """
    A generic polynomial class for the purpose of implementing
    the efficient Horner method.

    This just allows for rapid implementation
    for in these libraries without recourse to external dependencies.
    """

    def __init__(self, coefficients: torch.Tensor):
        self.coefficients = coefficients
        self.order = len(self.coefficients)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the polynomial using the recursive eval_poly function.
        """
        return self._eval_poly(x, 0)

    def _eval_poly(self, x: torch.Tensor, deg: int) -> torch.Tensor:
        """
        Recursively evaluates the polynomial using Horner's method.

        This leads to higher stability numerically as it lowers the number
        of large summations.
        """
        if deg == self.order - 2:
            return self.coefficients[deg] + x * self.coefficients[deg + 1]
        else:
            return self.coefficients[deg] + x * self._eval_poly(x, deg + 1)

    def get_coefficients(self) -> torch.Tensor:
        return self.coefficients

    def set_coefficients(self, coefficients):
        assert (
            coefficients.shape[0] == self.order
        ), "Coefficients not the right length, please provide as many as\
        the order of the polynomial. The get_order method will provide this."
        self.coefficients = coefficients

    def get_order(self) -> int:
        return self.order


def jordan_matrix(s, t):
    order = len(s)
    ones = torch.ones(order)
    zeros = torch.zeros(order)
    matrix = torch.zeros((order + 1, order))
    block = torch.vstack((torch.ones(order), s, t)).t()
    for i in range(order + 1):
        # breakpoint()
        if i < order - 2:
            matrix[i, i : i + 3] = torch.tensor([ones[i], s[i], t[i]])
        elif i < order - 1:
            matrix[i, i : i + 2] = torch.tensor([ones[i], s[i]])
        elif i < order:
            matrix[i, i] = torch.tensor([ones[i]])
    # for i
    # print(matrix)
    # breakpoint()
    return matrix[:, :].t()


def weight_mask(order) -> (torch.Tensor, torch.Tensor):
    # order = len(s)
    ones = torch.ones(order)
    zeros = torch.zeros(order)
    matrix = torch.zeros((order + 1, order))
    matrix_2 = torch.zeros((order + 1, order))

    # block = torch.vstack((torch.ones(order), s, t)).t()
    for i in range(order + 1):
        # breakpoint()
        if i < order - 2:
            matrix[i, i : i + 3] = torch.tensor([zeros[i], ones[i], ones[i]])
            matrix_2[i, i : i + 3] = torch.tensor(
                [ones[i], zeros[i], zeros[i]]
            )
        elif i < order - 1:
            matrix[i, i : i + 2] = torch.tensor([zeros[i], ones[i]])
            matrix_2[i, i : i + 2] = torch.tensor([ones[i], zeros[i]])
        elif i < order:
            matrix[i, i] = torch.tensor([zeros[i]])
    return matrix[:, :].t(), matrix_2[:, :].t()


class CatNet(nn.Module):
    """
    A Catalan matrix represented as a Neural net, to allow
    us to differentiate w.r.t. its parameters easily.
    """

    def __init__(self, order, betas, gammas):
        super().__init__()

        # Check the right size of the initial s
        if (betas.shape[0] != 2 * order) or (gammas.shape[0] != 2 * order):
            raise ValueError(
                r"Please provide at least 2 * order parameters for beta and gamma"
            )
        # breakpoint()
        self.order = order
        self.mask, self.ones_matrix = weight_mask(len(betas))
        self.jordan = torch.nn.Parameter(jordan_matrix(betas, gammas))  # .t()
        self.layers = []
        self.shared_weights = []
        for i in range(1, 2 * order + 1):
            self.shared_weights.append(
                torch.nn.Parameter(self.jordan[: i + 1, :i])
            )
            layer = nn.Linear(i, i + 1, bias=False)
        with torch.no_grad():
            layer.weight = torch.nn.Parameter(self.jordan[: i + 1, :i])
        self.layers.append(layer)
        return

    def forward(self, x):
        """
        As is, with s = 0 and t = 1, the resulting values should be the Catalan
        numbers. The parameters need to be shared between all the layers...

        This needs to be done by sourcing the parameters from the Jordan matrix
        but i'm not sure that this is what is happening here.
        """
        catalans = torch.zeros(2 * self.order)
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                print("Shared weights are:", self.shared_weights[3])
                layer.weight *= self.mask[: i + 2, : i + 1]
                layer.weight += self.ones_matrix[: i + 2, : i + 1]
            # print(layer.weight.shape)
            x = layer(x)
            # breakpoint()
            catalans[i] = x[
                -1
            ]  # the last in the layer will be the corresponding catalan no.

        return torch.hstack(
            (
                torch.tensor(
                    1,
                ),
                catalans[:-1],
            )
        )


# class coeffics_list:
# """
# A small class to handle the use of object-by-reference in Python -
# Used in the construction of the tensor of coefficients for the construction
# of the sequence of moments for a given orthogonal polynomial. That is,
# for a given OrthogonalPolynomial class with given sequences of betas
# and gammas, the resulting moments result from the three-term recursion.
# """

# def __init__(self, order):
# """
# Initialises the data to be a sequence of zeroes of length order + 1;
# this is because an order 3 polynomial will need 4 coefficients in the
# matrix used to solve for the moments
# """
# self.order = order
# self.data = torch.zeros(order + 1)  # the length is the order + 1

# def update_val(self, k, value):
# self.data[k] += value  # we _update_ the value

# def get_vals(self):
# return self.data

# def __len__(self):
# return self.order

# def __repr__(self):
# return self.data.__repr__()


# def get_moments_from_poly(poly):
# """
# For a given polynomial, computes the moments of the measure w.r.t which the
# polynomial is orthogonal, utilising the three-term recurrence that defines
# the orthogonal polynomial, in conjunction with the Linear functional
# operator L.

# For more details, check the documentation for L

# """
# order = poly.get_order()
# final_coefficients = torch.zeros([order, order])

# for i in range(order):
# coeffics = coeffics_list(i)
# L((0, i), 1, coeffics, poly)
# final_coefficients[i, : (i + 1)] = coeffics.get_vals()

# coeffics_inv = torch.linalg.inv(final_coefficients)
# targets = torch.zeros(order)
# targets[0] = 1  # assuming we're getting a probability distribution
# mus = torch.einsum("ij,j->i", coeffics_inv, targets)

# return mus


# def get_measure_from_poly(poly) -> Callable:
# # get the moments:
# moments = get_moments_from_poly(poly)
# """
# The sequence of moments from the polynomial imply a unique linear moment
# functional, but multiple corresponding densities. To that extent, we can
# choose the one that maximises the Shannon informational entropy, subject
# to the constraints implied by the moments -
# see, "Inverse two-sided Laplace transform for probability density
# functions", (Tagliani 1998)
# """
# # to build the density, we solve the following problem:
# # min μ_0 ln(1/μ_0  \int exp(-Σλ_j t^j) dt ) + Σλ_j μ_j
# sigma = 2
# b = 1
# measure = build_inverse_laplace_from_moments(moments, sigma, b)
# return measure


# def L(
# loc: tuple,
# value: float,
# coeffics: coeffics_list,
# poly: OrthogonalPolynomial,
# ):
# """
# Recursively calculates the coefficients for the construction of the moments

# This is a representation of the linear functional operator:
# <L, x^m P_n>  === L(m, n)


# Places into coeffics the coefficients of order len(coeffics).

# The aim is to build the matrix of coefficients to solve for the moments:

# [1       0     0  ... 0] |μ_0|   |1|
# [-β_0    1     0       ] |μ_1|   |0|
# [-γ_1  -β_0    1  ... 0] |μ_2| = |0|
# [          ...         ] |...|   |0|
# [?       ?        ... 1] |μ_k|   |0|

# To find the coefficients for coefficient k, start with a list of length k,
# and pass (0,k) to the function L which represents the linear functional
# operator for the OPS.

# L takes (m,n) and a value and calls itself on [(m+1, n-1), 1*value],
# [(m, n-1), -β_{n-1}*value]
# [(m, n-2), -γ_{n-1}*value]
# Mutiply all the values passed by the value passed in.

# The function places in coeffics a list of length k, the coefficients
# for row k in the above described matrix.

# :param loc: a tuple (m, n) representing that this is L(m,n) == <L, x^m P_n>
# :param value: a coefficient value passed in will multiply the coefficient
# found in this iteration
# :param coeffics: the coeffics_list object that will be updated
# :param poly: the orthogonal polynomial system w.r.t which we are building
# the linear moment functional - i.e. this polynomial system is
# orthogonal with respect to the measure whose moments are
# produced by this linear moment functional
# """
# local_betas = poly.get_betas()
# local_gammas = poly.get_gammas()

# if loc[1] == 0:  # n is 0
# # i.e loc consists of a moment (L(m, 0))
# # add the value to the corresponding coefficient in the array
# coeffics.update_val(loc[0], value)

# # check if it's the first go:
# if not (loc[1] < 0 or (loc[0] == 0 and loc[1] != len(coeffics))):
# # m != 0, n != 0
# left_loc = (loc[0] + 1, loc[1] - 1)
# centre_loc = (loc[0], loc[1] - 1)
# right_loc = (loc[0], loc[1] - 2)
# L(left_loc, value, coeffics, poly)
# L(centre_loc, value * -local_betas[loc[1] - 1], coeffics, poly)
# L(right_loc, value * -local_gammas[loc[1] - 1], coeffics, poly)
# pass


if __name__ == "__main__":
    # build an orthogonal polynomial
    order = 10

    test_moment_gen_2 = True
    # build random betas and gammas

    if test_moment_gen_2:
        betas = 0 * torch.ones(2 * order)
        gammas = 1 * torch.ones(2 * order)

        med = MaximalEntropyDensity(order, betas, gammas)
        x = torch.linspace(-5, 5, 1000)
        vals = med(x)
        plt.plot(x, vals)
        plt.show()
        # moments = med._get_moments()
        # print(result)

    # # testing the moment acquirer
    # s = 0 * torch.ones(1 * order)
    # t = 1 * torch.ones(1 * order)
    # order = 8
    # my_net = CatNet2(order, betas, gammas)
    # my_net(torch.Tensor([1.0]))
    # catalans = my_net(torch.Tensor([1.0]))
    # # optimiser = torch.optim.SGD(my_net.shared_weights, 0.001)
    # print(catalans)
