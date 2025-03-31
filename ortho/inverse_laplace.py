import torch
import matplotlib.pyplot as plt
import math
from ortho.vandermonde import Bjorck_Pereyra
from typing import Callable
from ortho.basis_functions import Basis
from ortho.polynomials import generalised_laguerre

"""
Implements the inverse Laplace transform algorithm as explained in:
    Computation of the inverse Laplace transform based on a collocation method
    which uses only real values

    Cuomo, D'Amore, Murli, and Rizzard, 2005.


To build the measure w.r.t which the orthogonal polynomial system is
orthogonal, we can take the inverse laplace transform of the moment generating
function that corresponds to the moments. Since the moments can be acquired
from a given orthogonal polynomial system, for given {β_n, γ_n}, we can build
the MGF using:
        Σ_i μ_i x^i / i!

Then, this means we can construct the inverse Laplace transform of this
constructed function, to get the measure w.r.t it's orthogonal.
"""


def chebyshev_roots(order: int) -> torch.Tensor:
    """
    Returns the roots of the Chebyshev polynomial of order N
    for the purpose of constructing the Laguerre-polynomials based inverse
    Laplace transform
    """
    k = torch.linspace(0, order - 1, order)  # the sequence of integers 0-order
    return torch.cos((2 * k + 1) * math.pi / (2 * order))


def mgf(x: torch.Tensor, moments: torch.Tensor):
    """
    Returns the moment generating function for a given linear functional, which
    is uniquely determined by the sequence of moments, is given as:
        Σ_i μ_i x^i / i!

    The inverse laplace transform of this should give the measure w.r.t
    a given orthogonal polynomial system will be orthogonal.

    x: shape of N, 1
    moments: shape of M, 1
    """
    order = len(moments)
    facs = torch.exp(torch.lgamma(torch.linspace(1, order, order)))
    V = torch.vander(x, N=order, increasing=True)
    return torch.einsum("ij,j->i", V, moments / facs)


def mobius_transform(sigma, b, w):
    return 2 * b / (1 - w) + sigma - b


def build_inverse_laplace(
    func: Callable, b: torch.Tensor, sigma: torch.Tensor, order, gamma=1
):
    """
    It works! But it is slow.
    """
    # step 1: evaluate the MGF at the Chebyshev nodes, and push through
    # the bilateral transform:
    w = chebyshev_roots(order)
    phi_w = ((2 * b / (1 - w)) ** gamma) * func(mobius_transform(sigma, b, w))

    # step 2: calculate the coefficients of the polynomial that interpolates
    # phi_w at w:
    interpolating_coeffics = Bjorck_Pereyra(w, phi_w)
    params = {"alpha": torch.tensor(0.0)}
    laguerre_basis = Basis(generalised_laguerre, 1, order, params)

    # When evaluating the laguerre basis, we get a matrix that is of size:
    # [N, m]
    # Assuming this is correct, we need:
    # 'm,nm -> exp(-bx)L(2bx), interpolating_coeffics'
    return lambda x: torch.exp((sigma - b) * x) * torch.einsum(
        "nm, m -> n",
        laguerre_basis(2 * b * x),
        interpolating_coeffics,
    )


def build_inverse_laplace_from_moments(
    moments: torch.Tensor, sigma: torch.Tensor, b: torch.Tensor, gamma=1
) -> Callable:
    """
    Constructs the inverse laplace transform of the MGF for the linear moment
    functional that corresponds to the set of moments passed in as 'moments'.
    """
    order = len(moments)
    result = build_inverse_laplace(
        lambda x: mgf(x, moments), sigma, b, order, gamma
    )
    return result


if __name__ == "__main__":
    N = 20
    roots = chebyshev_roots(N)
    moments = torch.ones(N)
    x = torch.linspace(0, 5, 100)
    mgf_values = mgf(x, moments)
    plt.plot(mgf_values)
    plt.plot(torch.exp(x))
    plt.show()

    s = 1  # "sigma" for laplace
    b = 1  # "b" for laplace
    # plt.scatter(roots, torch.zeros(roots.shape))
    # plt.show()

    inverse_laplace_function = build_inverse_laplace_from_moments(
        moments, torch.Tensor([1]), torch.Tensor([1])
    )
    inverse_laplace_function_2 = build_inverse_laplace(
        lambda x: mgf(x, moments), b, s, N
    )

    # Example: sin: sin(ωt) * u(t)
    mu = 1
    sigma = 1
    t = 2
    F = lambda x: t / (x ** 2 + t ** 2)
    my_sin = build_inverse_laplace(F, b, s, N)
    density_vals = my_sin(x)
    plt.plot(x, torch.sin(t * x).numpy())
    plt.plot(x, density_vals.numpy(), color="red")
    plt.show()
