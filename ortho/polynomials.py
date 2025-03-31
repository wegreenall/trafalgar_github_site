import torch
from ortho.orthopoly import OrthogonalPolynomial
import matplotlib.pyplot as plt

"""
This file implements various families of orthogonal polynomials as direct
instances of the Orthogonal Polynomial class.
"""


def chebyshev_first(x: torch.Tensor, deg: int) -> torch.Tensor:
    """
    Evaluates the Chebyshev polynomial of the first kind at x, for order i
    """
    return torch.cos(deg * torch.arccos(x))


def chebyshev_second(x: torch.Tensor, deg: int) -> torch.Tensor:
    """
    Evaluates the Chebyshev polynomial of the second kind at x, for order i
    """
    return torch.sin((deg + 1) * torch.arccos(x)) / torch.sin(torch.arccos(x))


def generalised_laguerre(x: torch.Tensor, deg: int, params: dict):
    """
    Implements the Generalized Laguerre polynomials.

    The generalised Laguerre polynomials can be written recursively:
        L_0^α(x) = 1
        L_1^α(x) = 1 + α - x
    and then:
        L_{k+1}^α(x) = ((2k + 1 + α  - x)L_k^α(x) - (k+α)L_{k-1}^α(x)) / k+1
    """
    assert (
        "alpha" in params
    ), "Missing parameter for generalised laguerre polynomial: alpha"
    alpha = params["alpha"]

    if deg == 0:
        return torch.ones(x.shape)
    elif deg == 1:
        return 1 + alpha - x
    else:
        # k = deg - 1
        coeffic_1 = 2 * (deg - 1) + 1 + alpha - x
        coeffic_2 = deg - 1 + alpha
        denom = deg
        return (
            coeffic_1 * generalised_laguerre(x, deg - 1, params)
            - coeffic_2 * generalised_laguerre(x, deg - 2, params)
        ) / denom


class ProbabilistsHermitePolynomial(OrthogonalPolynomial):
    def __init__(self, order):
        betas = torch.zeros(order)
        gammas = torch.linspace(0, order - 1, order)
        super().__init__(order, betas, gammas)


class WeightedProbabilistsHermitePolynomial(OrthogonalPolynomial):
    def __init__(self, order, weight: float):
        betas = torch.zeros(order)
        gammas = weight * torch.linspace(0, order - 1, order)
        super().__init__(order, betas, gammas)


class HermitePolynomial(OrthogonalPolynomial):
    def __init__(self, order):
        betas = torch.zeros(order)
        gammas = 2 * torch.linspace(0, order - 1, order)
        super().__init__(order, betas, gammas, leading=2)


class GeneralizedLaguerrePolynomial(OrthogonalPolynomial):
    def __init__(self, order, alpha):
        ns = torch.linspace(0, order - 1, order)
        betas = 2 * ns + alpha + 1
        gammas = (ns + 1) * (ns + alpha + 1)
        gammas[0] = 1
        super().__init__(order, betas, gammas)


class LaguerrePolynomial(GeneralizedLaguerrePolynomial):
    def __init__(self, order):
        super().__init__(order, 0)


if __name__ == "__main__":
    # Hermite polynomials
    order = 12
    x = torch.linspace(-2, 2, 1000)
    hermite = HermitePolynomial(order)
    params = dict()
    plt.plot(x, hermite(x, 4, params))
    plt.show()

    for i in range(order):
        plt.plot(x, hermite(x, i, params) ** 2)
    plt.show()
