# basis_functions.py
import math
from ortho.orthopoly import OrthogonalPolynomial

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributions as D

# from framework import special, utils
from ortho.polynomials import (
    chebyshev_first,
    chebyshev_second,
    generalised_laguerre,
)
from typing import Callable, Union, Tuple

from ortho.special import hermite_function

"""
This contains implementations of basis functions useful for constructing
kernels according to the Mercer representation theorem.

The basis functions are expected to have signature (x, i, params),
where:

x : input value
i : the degree of the given function (in most cases, the degree of the
polynomial that corresponds to that function)

params:  a dictionary of parameters that contains the appropriate titles etc.
for that set of polynomials.

Return value must be a torch.Tensor of dimension [n] only;
not [n, 1] as is the inputs vector - this reflects explicitly that the output
would be 1-d (2-d inputs would be of shape [n,2]). The result of this is that
multi-dimensional problems require the construction of the tensor product 
formulation of an orthonormal basis.
"""


class Basis:
    def __init__(
        self,
        basis_functions: Union[Callable, Tuple[Callable]],
        dimension: int,
        order: int,
        parameters: dict,
    ):
        """
        :param basis_functions: either a Callable function, w/ signature:
                    basis_function(x: torch.Tensor,
                                   deg: int,
                                   parameters: dict) -> torch.Tensor
                    or a tuple, of size dimension, of Callables, w/ signature:
                    basis_function(x: torch.Tensor,
                                   deg: int,
                                   parameters: dict) -> torch.Tensor
                    which will each be applied to separate dimensions of the
                    input.
        """
        self.dimension = dimension
        self.order = order

        # check whether the number of basis functions is correct
        if isinstance(basis_functions, Callable):
            basis_functions = (basis_functions,)

        if len(basis_functions) != dimension:
            raise ValueError(
                "The number of basis functions passed in must match the dimension parameter"
            )

        if isinstance(parameters, dict) or parameters is None:
            parameters = (parameters,)

        if parameters is not None and len(parameters) != self.dimension:
            raise ValueError(
                "The number of parameter dicts passed must match the dimension parameter"
            )

        self.basis_functions = basis_functions
        self.parameters = parameters
        return

    def get_dimension(self):
        """
        Getter method for the dimension of the model.
        """
        return self.dimension

    def get_order(self):
        """
        Getter method for the order of the model (i.e. number of basis functions).
        """
        return self.order

    def _get_intermediate_result(self, x: torch.Tensor):
        # check input shape
        if len(x.shape) <= 1:
            x = x.unsqueeze(-1)
        elif x.shape[1] != self.dimension:
            raise ValueError(
                "The dimension of the inputs should be {dim} for this,\
                              because that is the dim of the\
                              basis.".format(
                    dim=self.dimension
                )
            )
        result = []  # torch.zeros(x.shape[0], self.order, self.dimension)

        # construct the multidimensional basis evaluation
        for d in range(self.dimension):
            basis_function = self.basis_functions[
                d
            ]  # get the basis function in this dimension
            result.append(
                torch.vstack(
                    [
                        basis_function(x[:, d], deg, self.parameters[d])
                        for deg in range(self.order)
                    ]
                ).t()
            )  # [Nxm]
        return result

    def __call__(self, x):
        """
        Returns the whole basis evaluated at an input.

        input shape: [x.shape[0], self.dimension]
        output shape: [x.shape[0], self.order ** self.dimension] i.e. N x m
        """
        # result is a list of d N x m matrices
        result = self._get_intermediate_result(x)

        potential_result = reshaping(result)
        result = torch.reshape(
            potential_result, (x.shape[0], self.order**self.dimension)
        )

        if self.dimension == 1:
            result = result.squeeze(1)
        return result

    def get_parameters(self):
        return self.parameters

    def __add__(self, other):
        """
        When adding two bases, return a basis that
        is of the same order.
        """
        if self.order != other.order:
            raise ValueError(
                "Other basis is not of the same order as this basis. This Basis order: {this_order}; other Rasis order: {other_order}".format(
                    this_order=self.order, other_order=other.order
                )
            )
        if self.dim != other.dim:
            raise ValueError(
                "Other basis is not of the same dim as this basis. This Basis order: {this_dim}; other Rasis order: {other_dim}".format(
                    this_dim=self.dim, other_dim=other.dim
                )
            )
        new_basis = CompositeBasis(self, other)
        return new_basis


class CompositeBasis(Basis):
    def __init__(self, old_basis, new_basis: Basis):
        self.old_basis = old_basis
        self.new_basis = new_basis

    def __call__(self, x):
        return self.old_basis(x) + self.new_basis(x)

    def get_order(self):
        return self.order


class SmoothExponentialBasis(Basis):
    def __init__(
        self,
        dimension: int,
        order: int,
        params: dict,
    ):
        """
        The smooth exponential basis functions as constructed in Fasshauer (2012),
        "first" paramaterisation:
            a, b, c.

        The Hermite function here is that constructed with the Physicist's Hermite
        Polynomial as opposed to the Probabilist's.

        : param dimension: the dimension of the input space
        : param order: the degree of the basis functions
        : param params: dictionary of parameters.
                        Required keys:
                            - precision_parameter
                            - ard_parameter
        : param weight_function: a Callable function, w/ signature:
                    weight_function(x: torch.Tensor,
                                   deg: int,
                                   parameters: dict) -> torch.Tensor
                    which will be applied to the basis functions.
        """
        super().__init__(
            basis_functions=smooth_exponential_basis_fasshauer,
            dimension=dimension,
            order=order,
            parameters=params,
        )


class RandomFourierFeatureBasis(Basis):
    def __init__(
        self,
        dim: int,
        order: int,
        spectral_distribution: D.Distribution,
        use_2pi=False,
    ):
        """
        Random Fourier Feature basis for constructing the
        """
        # self.w_dist = D.Normal(torch.zeros(dim), torch.ones(dim))
        self.w_dist = spectral_distribution  # samples are the size of "dim" which is 1-d for stationary, 2-d for non-stationary.
        self.b_dist = D.Uniform(0.0, 2 * math.pi)
        self.b_sample_shape = torch.Size([order])
        self.w_sample_shape = torch.Size([order])

        self.b = self.b_dist.sample(self.b_sample_shape)
        self.w = self.w_dist.sample(self.w_sample_shape)  # order x dim

        self.order = order
        self.params = None

        self.use_2pi = use_2pi  # mark whether we're multiplying the argument in the cosine by 2π. I think it is necessary!

    def __call__(self, x: torch.Tensor):
        """
        Returns the value of these random features evaluated at the inputs x.

        output shape: [x.shape[0], self.order]
        """
        if len(x.shape) == 1:  # i.e. the single dimension is implicit
            x = x.unsqueeze(1)

        n = x.shape[0]
        b = self.b.repeat(n, 1).t()

        """
        Because the frequencies calculated in the DFT are integer frequencies, 
        they're essentially pre-scaled by 2 * π. Multiplying
        the argument in cos(x) by 2π gives all integer frequencies (in radians)
        meaning you get strong excessive periodic behaviour in the
        resulting basis functions. 
        """
        z = (math.sqrt(2.0 / self.order) * torch.cos(self.w @ x.t() + b)).t()

        return z

    def get_w(self):
        """
        Getter for feature spectral weights.
        """
        return self.w

    def get_b(self):
        """
        Getter for feature phase parameters.
        """
        return self.b


class OrthonormalBasis(Basis):
    def __init__(
        self,
        basis_functions: OrthogonalPolynomial,
        weight_functions: Callable,
        dimension: int,
        order: int,
        params: dict = None,
    ):
        # assert isinstance(
        # basis_functions, OrthogonalPolynomial
        # ), "the basis function should be of type OrthogonalPolynomial"

        super().__init__(basis_functions, dimension, order, params)
        if isinstance(weight_functions, tuple) and len(weight_functions) != dimension:
            raise ValueError(
                "The number of basis functions passed in must match the dimension parameter"
            )
        elif isinstance(weight_functions, Callable):
            # put the single basis function into a tuple for DRY in the call method
            weight_functions = (weight_functions,)
        self.weight_functions = weight_functions

    def __call__(self, x):
        """
        Returns the whole basis evaluated at an input. The difference between an
        orthonormal basis and a standard set of polynomials is that the weight
        function and normalising constant
        can be applied "outside" the tensor of orthogonal polynomials,
        so it is feasible to do this separately (and therefore faster).
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        ortho_poly_basis = super()._get_intermediate_result(x)
        for d, weight_function in enumerate(self.weight_functions):
            weights = torch.sqrt(
                weight_function(x[:, d])
            ).squeeze()  # the weight function is per-dimension
            # breakpoint()
            try:
                ortho_poly_basis[d] *= weights.repeat(self.order, 1).t()
            except:
                print("ERRORED!")
                breakpoint()
        # now that we have the
        reshaped_ortho_basis = reshaping(ortho_poly_basis)
        result = torch.reshape(
            reshaped_ortho_basis, (x.shape[0], self.order**self.dimension)
        )

        if self.dimension == 1:
            result = result.squeeze()
        return result
        # return torch.einsum("ij,i -> ij", ortho_poly_basis, result)

    def set_gammas(self, gammas):
        """
        Updates the gammas on the basis function and the
        weight function.
        """
        for i, basis_function in enumerate(self.basis_functions):
            basis_function.set_gammas(gammas[:, i])
        for i, weight_function in enumerate(self.weight_functions):
            weight_function.set_gammas(gammas[:, i])


def smooth_exponential_basis(x: torch.Tensor, deg: int, params: dict):
    print("THIS SHOULD NOT BE BEING USED ANYWHERE")
    """
    The smooth exponential basis functions as constructed in Fasshauer (2012),
    "second" paramaterisation. It is orthogonal w.r.t the measure ρ(x)
    described there.

    The Hermite function here is that constructed with the Physicist's Hermite
    Polynomial as opposed to the Probabilist's.


    : param x: the input points to evaluate the function at. Should be of
               dimensions [n,d]
    : param deg: degree of polynomial used in construction of func
    : param params: dict of parameters. keys set on case-by-case basis
    """
    if deg > 42:
        print(
            r"Warning: the degree of the hermite polynomial is relatively"
            + "high - this may cause problems with the hermvals function."
            + "Try lowering the order, or use a different basis function."
        )

    epsilon = torch.diag(params["ard_parameter"])  # ε  - of dimension d
    alpha = torch.diag(params["precision_parameter"])  # precision parameter
    # of the measure w.r.t the hermite functions are orthogonal

    sigma = torch.sqrt(params["variance_parameter"])

    # β = (1 + (2ε/α)^2)^(1/4)
    beta = torch.pow((1 + torch.pow((2 * epsilon / alpha), 2)), 0.25)

    log_gamma = 0.5 * (
        torch.log(beta)
        - (deg) * torch.log(torch.tensor(2, dtype=torch.float))
        - torch.lgamma(torch.tensor(deg + 1, dtype=torch.float))
    )

    delta_2 = (torch.pow(alpha, 2) / 2) * (torch.pow(beta, 2) - 1)

    multiplicative_term = torch.exp(
        log_gamma - (delta_2 * torch.pow(x, torch.tensor(2)))
    )

    # calculate the Hermite polynomial term
    # remember, deg = n-1
    hermite_term = hermite_function(alpha * beta * x, deg)

    # for numerical reasons, we can save the log of the absolute to get the
    #  right value,
    # then exponentiate and multiply by a mask of negative values to get the
    # right value.

    abs_hermite_term = torch.log(torch.abs(hermite_term))
    mask = torch.where(
        hermite_term < 0,
        -torch.ones(hermite_term.shape),
        torch.ones(hermite_term.shape),
    )
    phi_d = mask * torch.exp(abs_hermite_term) * multiplicative_term
    phi = sigma * phi_d
    # if deg == 56 or deg == 57:
    #     breakpoint()
    return phi


def smooth_exponential_eigenvalues(deg: int, params: dict):
    print("THIS SHOULD NOT BE BEING USED ANYWHERE")
    """
    Returns the vector of eigenvalues, up to length deg, using the parameters
    provided in params. This comes from Fasshauer2012 - where it is explained
    that the smooth exponential basis is orthogonal w.r.t a given measure
    (with precision alpha, etc.)

    :param deg: the degree up to which the eigenvalues should be computed.
    :param params: a dictionary of parameters whose keys included
    """

    eigenvalues = torch.zeros(deg)

    for i in range(1, deg + 1):
        epsilon = torch.diag(params["ard_parameter"])  # ε  - of dimension d
        alpha = torch.diag(params["precision_parameter"])  # precision
        # of the measure w.r.t the hermite functions are orthogonal

        beta = torch.pow((1 + torch.pow((2 * epsilon / alpha), 2)), 0.25)

        delta_2 = 0.5 * torch.pow(alpha, 2) * (torch.pow(beta, 2) - 1)

        denominator = torch.pow(alpha, 2) + delta_2 + torch.pow(epsilon, 2)

        left_term = alpha / torch.sqrt(denominator)
        right_term = torch.pow((torch.pow(epsilon, 2) / denominator), i - 1)

        lamda_d = left_term * right_term
        eigenvalue = torch.prod(lamda_d, dim=1)
        eigenvalues[i - 1] = eigenvalue
    return eigenvalues


def smooth_exponential_basis_fasshauer(
    x: torch.Tensor, deg: int, params: dict, dim=None
):
    """
    The smooth exponential basis functions as constructed in Fasshauer (2012),
    "first" paramaterisation:
        a, b, c.

    The Hermite function here is that constructed with the Physicist's Hermite
    Polynomial as opposed to the Probabilist's.

    : param x: the input points to evaluate the function at. Should be of
               dimensions [n, d]
    : param deg: degree of polynomial used in construction of func
    : param params: dict of parameters.
                    Required keys:
                        - precision_parameter
                        - ard_parameter
    : param dim: the dimension to evaluate the basis on

    In order to allow per-dimension basis function selection on Gaussian processes
    (see mercergp), we allow one to use the standard smooth exponential
    fasshauer basis here; using the dim parameter to select a dimension
    on which to evaluate the basis.
    """
    if deg > 42:
        print(
            r"Warning: the degree of the hermite polynomial is relatively"
            + "high - this may cause problems with the hermvals function."
            + "Try lowering the order, or use a different basis function."
        )

    if dim is not None and isinstance(dim, int):
        a = torch.diag(params["precision_parameter"])[dim]  # precision parameter
        b = torch.diag(params["ard_parameter"])[dim]  # ε  - of dimension d
    else:
        a = torch.diag(params["precision_parameter"])  # precision parameter
        b = torch.diag(params["ard_parameter"])  # ε  - of dimension d
    # of the measure w.r.t the hermite functions are orthogonal
    c = torch.sqrt(a**2 + 2 * a * b)
    # sigma = torch.sqrt(params["variance_parameter"])
    if (c != c).any():
        print("c in fasshauer is NaN!")
        breakpoint()

    # β = (1 + (2ε/α)^2)^(1/4)
    log_const_term = -0.5 * (
        deg * torch.log(torch.tensor(2))
        + torch.lgamma(torch.tensor(deg) + 1)
        + 0.5 * (torch.log(a) - torch.log(c))
    )

    # calculate the Hermite polynomial term
    hermite_term = hermite_function(
        torch.sqrt(2 * c) * x,
        deg,
    )

    phi_d = (
        torch.exp(log_const_term - ((c - a) * torch.pow(x, torch.tensor(2))))
        * hermite_term
    )
    return phi_d


def smooth_exponential_eigenvalues_fasshauer(order: int, params: dict):
    """
    If in one dimension, returns the vector of eigenvalues, up to length order, using the parameters
    provided in params. Otherwise, returns a matrix of [order, dimension]
    per-dimension eigenvalue vectors as columns. The calling EigenvalueGenerator
    class is then expected to convert these to tensor product length
    i.e. to become [order ** dimension].
    :param order: the degree up to which the eigenvalues should be computed.
    :param params: a dictionary of parameters whose keys included
    """
    b = torch.diag(params["ard_parameter"])  # ε  - of dimension d
    a = torch.diag(params["precision_parameter"])  # precision
    c = torch.sqrt(a**2 + 2 * a * b)
    left_term = torch.sqrt(2 * a / (a + b + c))
    right_term = b / (a + b + c)

    # construct the vector
    exponents = torch.linspace(0, order - 1, order).repeat(len(b), 1).t()
    eigenvalues = left_term * torch.pow(right_term, exponents)
    return eigenvalues.squeeze()


def get_linear_coefficients_fasshauer(
    intercept: torch.Tensor, slope: torch.Tensor, params: dict
):
    """
    Returns the coefficients that represent a function that has the
    same slope and intercept at 0 for the fasshauer parameterised smooth
    exponential basis
    ( i.e. smooth_exponential_basis_fasshauer().

    :   param intercept: the intercept of the linear function we want to
                         approximate
    :   param slope:     the slope of the linear function we want to
                         approximate
    """
    b = torch.diag(params["ard_parameter"])  # ε  - of dimension d
    a = torch.diag(params["precision_parameter"])  # precision
    c = torch.sqrt(a**2 + 2 * a * b)

    basis_intercept = intercept * torch.pow(a / c, 0.25)
    basis_slope = 0.5 * slope * torch.pow(a, 0.25) / torch.pow(c, 0.75)
    return (basis_intercept, basis_slope)


def standard_laguerre_basis(x: torch.Tensor, deg: int, params: dict):
    """
    Returns a Laguerre basis function, using the Generalised Laguerre
    polynomials, evaluated at x.

    :  param x:      input tensor to evaluate the function at
    :  param deg:    the degree of the basis function
    :  param params: dictionary of kernel arguments. Should include
                     an "alpha" parameter in order to get the generalised
                     laguerre polynomial. This polynomial includes
                     the standard laguerre polynomial as a special case when
                     alpha = 0.
    """
    laguerre_term = generalised_laguerre(x, deg, params)
    x_exponent = params["alpha"] / 2
    weight_term = (x ** (x_exponent)) * torch.exp(-x / 2)
    normalising_constant = torch.exp(
        (
            torch.lgamma(torch.tensor(deg + params["alpha"] + 1))
            - torch.lgamma(torch.tensor(deg + 1))
        )
        / 2
    )
    return laguerre_term * weight_term / normalising_constant


def standard_chebyshev_basis(x: torch.Tensor, deg: int, params: dict):
    """
    Returns a standard Chebyshev basis function, using Chebyshev
    polynomials of the first kind, evaluated at x, that does
    not impose the zero-endpoint condition of Babolian

    :  param x:      input tensor to evaluate the function at
    :  param deg:    the degree of the basis function
    :  param params: dictionary of kernel arguments. Should
                     contain keys 'upper_bound' and 'lower_bound':
                     tensors representing the upper and lower bound
                     over which the function is to be a basis respectively.
    """

    # check that x is of the right dimension:
    # assert len(x.shape) == 2,\
    #    "inputs to basis function should have 2 dimensions."

    if "lower_bound" not in params:
        lower_boundary = torch.tensor(-1, dtype=float)
    else:
        lower_boundary = params["lower_bound"]

    if "upper_bound" not in params:
        upper_boundary = torch.tensor(1, dtype=float)
    else:
        upper_boundary = params["upper_bound"]

    if "chebyshev" not in params:
        # default result: chebyshev polynomials/basis of the second kind
        chebyshev = "second"
    else:
        chebyshev = params["chebyshev"]

    assert chebyshev in {
        "first",
        "second",
    }, 'chebyshev should be either "first" or "second"'

    # Transform to [-1,1] for processing
    z = (2 * x - (upper_boundary + lower_boundary)) / (upper_boundary - lower_boundary)

    if chebyshev == "first":
        chebyshev_term = chebyshev_first(z, deg)
        # exponent of weight function (1-z**2)
        weight_power = torch.tensor(-0.25)

        # define the normalising constant
        spacing_term = math.sqrt((upper_boundary - lower_boundary) / 2)
        if deg == 0:
            normalising_constant = math.sqrt(1 / math.pi) / spacing_term
        else:
            normalising_constant = math.sqrt(2 / math.pi) / spacing_term

    elif chebyshev == "second":
        chebyshev_term = chebyshev_second(z, deg)
        # exponent of weight function (1-z**2)
        # weight_power = torch.Tensor([0.25])
        weight_power = torch.tensor(0.25)

        # define the normalising constant
        normalising_constant = 2 / math.sqrt(
            (upper_boundary - lower_boundary) * (math.pi)
        )

    # define weight function
    weight_term = torch.pow(1 - z**2, weight_power)
    if (chebyshev_term != chebyshev_term).any():
        print("Diagnostics for: Nans in Chebyshev. Code: {}".format(__name__))
        print("Line number: 592")
        print("Z values: (transformed to -1-ε, 1+ε)", z)
        print("X values: ", x)
        print("Got Nan in Chebyshev; check bounds")
        breakpoint()
        raise ValueError(
            "Chebyshev returning NaNs. Ensure"
            + "it is being evaluated within boundaries."
        )
    return weight_term * chebyshev_term * normalising_constant


def cosine_basis(x: torch.Tensor, deg: int, params: dict):
    """
    The cosine basis as found in (Walder and Bishop, 2017).
    """
    base_constant = math.sqrt(2.0 / math.pi)
    if deg == 0:
        normalising_constant = math.sqrt(1 / 2.0)
    else:
        normalising_constant = 1.0
    return torch.cos(deg * x) * base_constant * normalising_constant


def standard_haar_basis(x: torch.Tensor, deg: int, params: dict):
    """
    Returns a standard Chebyshev basis, using Chebyshev
    polynomials of the first kind, that does not impose
    the zero-endpoint condition of Babolian

    :  param x:      input tensor to evaluate the function at
    :  param deg:    the degree of the basis function
    :  param params: dictionary of kernel arguments. Should
                     contain keys 'upper_bound' and 'lower_bound':
                     tensors representing the upper and lower bound
                     over which the function is to be a basis respectively.
    """
    pass


def reshaping(tensors: torch.Tensor):
    """For a N x m x d tensor, returns the einsum resulting from:
    torch.einsum("na, nb, nc, nd -> nabcd", tensor[:,:,0], tensor[:,:,1], tensor[:,:,2], tensor[:,:,3])
    """
    einsum_string = ""
    used_chars = ""
    i = 0
    for i in range(len(tensors) - 1):
        einsum_string += "n" + chr(ord("a") + i) + ","
        used_chars += chr(ord("a") + i)
    einsum_string += (
        "n" + chr(ord("a") + i + 1) + "-> n" + used_chars + chr(ord("a") + i + 1)
    )
    result = torch.einsum(einsum_string, *tensors)
    return result


if __name__ == "__main__":
    # pass
    test_rff_basis = True
    test_rff_basis_multidim = False

    if test_rff_basis:
        dim = 1
        order = 5000
        point_count = 1000
        spectral_distribution = D.Normal(torch.zeros(dim), torch.ones(dim))
        rff = RandomFourierFeatureBasis(dim, order, spectral_distribution)
        # x = torch.linspace(-1, 1, 100)
        # x = torch.ones((order, dim))https://www.youtube.com/watch?v=T72TopWbXJg
        x = torch.linspace(-3, 3, point_count)

        data = rff(x)
        coeffics = D.Normal(0.0, 1.0).sample((order,))
        breakpoint()
        gp = coeffics @ data.t()

        # plt.plot(x.numpy().flatten(), data.numpy())
        plt.plot(x.numpy().flatten(), gp.numpy())
        plt.show()

    if test_rff_basis_multidim:
        dim = 2
        order = 5000
        point_count = 1000
        spectral_distribution = D.Normal(torch.zeros(dim), torch.ones(dim))
        # spectral_distribution = D.Normal(0.0, 1.0)
        rffbasis = RandomFourierFeatureBasis(dim, order, spectral_distribution)
        x = torch.linspace(-4, 4, point_count)
        plt.plot(x, rffbasis(x))
        plt.show()
