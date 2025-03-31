import torch
import math
from typing import Callable
import matplotlib.pyplot as plt


torch.set_default_tensor_type(torch.DoubleTensor)
"""
This file containes functions for calculating the roots; maxima; etc.
for a polynomial for given coefficients, etc.
"""


def gaussian(
    x: torch.Tensor, polynomial_coeffics: torch.Tensor, root, second_derivative
):
    """
    calculates the Gaussian at the peak given by root
    """
    x_shaped = root.repeat(deg, 1).t()
    x_at_powers = x_shaped ** torch.linspace(0, deg - 1, deg)

    poly_max = torch.einsum("ij, j->i", x_at_powers, polynomial_coeffics)

    if second_derivative <= 0:
        gaussian = torch.exp(
            poly_max + 0.5 * second_derivative * (x - root) ** 2
        )
    else:
        gaussian = torch.zeros(x.shape)
    return gaussian


def exp_of_poly(x: torch.Tensor, polynomial_coeffics: torch.Tensor, deg):
    """
    Evaluates the function
            exp{p(x)}
    where p(x) = δ'x, and x is (1, x, x^2, x^3, ..., x^m)
    """
    x_shaped = x.repeat(deg, 1).t()
    x_at_powers = x_shaped ** torch.linspace(0, deg - 1, deg)

    polynomial = torch.einsum("ij, j->i", x_at_powers, polynomial_coeffics)
    return torch.exp(polynomial)


def polynomial_vals(x: torch.Tensor, coeffics):
    """
    Evaluates the polynomial built with parameter coeffics.
    """
    return


def get_roots(polynomial_coeffics: torch.Tensor, deg: int):
    """
    This function first constructs the companion matrix for the given polynomial,
    and then calculates its eigenvalues, as in the numpy implementation.
    Given the coefficients of a monic polynomial, with coefficients
                    a_0, a_1, a_2, ..., a_{m-1}, 1

    the companion matrix looks like:
                  | 0 0 0 ...  0    -a_0 |
                  | 1 0 0 ...  0    -a_1 |
                  | 0 1 0 ...  0    -a_2 |
                  | 0 0 1 ...  0    -a_3 |
                  | 0 0 0 ...  0    -a_4 |
                  |       ...  0         |
                  | 0 0 0 ...  1 -a_{m-1}|

    The eigenvalues of this matrix will be the roots of the polynomial.
    Obviously, for non-monic polynomials we can divide all coefficients by the
    leading coefficient, since if there is a polynomial P_n(x),
                P_n(x) = 0 <=> P_n(x)/a_n = 0

    """
    normalised_coeffics = polynomial_coeffics / polynomial_coeffics[-1]
    assert normalised_coeffics[-1] == 1, "Normalising of coefficients failed!"
    companion = torch.eye(deg - 2)
    companion = torch.vstack((torch.zeros(deg - 2).unsqueeze(0), companion))
    companion = torch.hstack(
        (companion, -normalised_coeffics[: deg - 1].unsqueeze(1))
    )

    roots = torch.linalg.eigvals(companion)
    real_roots = torch.real(roots[torch.isreal(roots)])
    return real_roots


def get_deriv_roots(polynomial_coeffics: torch.Tensor, deg: int):
    """
    Returns the roots of the derivative of the polynomial (i.e. the peaks
    of the polynomial).

    param deg: the order of the original polynmomial
    param polynomial_coeffics: the coefficients of the original polynomial
    """
    deriv_coeffics = polynomial_coeffics[1:] * torch.linspace(
        1, deg - 1, deg - 1
    )
    roots = get_roots(deriv_coeffics, deg - 1)
    return roots


def get_polynomial_maximiser(polynomial_coeffics: torch.Tensor, deg: int):
    """
    Returns the maximiser of a polynomial that has even term,
    over all the roots of its derivative. I.e., the highest value at a local
    maximum.

    This is useful for calculating the maximum
    of
                   exp(δ'X)

    where δ'Χ = δ_1 x + δ_2 x^2 + ... + δ_m x^m.

    This maximum is one of the roots of the polynomial δ_prime'x.
            δ_1 + 2 δ_2 x + ... + δ_m x^{m-1}

    Therefore, first we construct the coefficients of the derivative
    polynomial, and calculate the corresponding polynomial's roots.

    The resulting set of points will be the stationary points of the
    polynomial, and as such will include the maximiser of the weight
    function.
    """
    # first, build the coefficients for the derivative
    # deriv_coeffics = torch.zeros(polynomial_coeffics.shape)
    deriv_roots = get_deriv_roots(polynomial_coeffics, deg)

    func_max = -math.inf
    if len(deriv_roots) > 0:
        for root in deriv_roots:
            root_at_powers = torch.pow(root, torch.linspace(0, deg - 1, deg))
            test_max = polynomial_coeffics @ root_at_powers  # δ'X
            if func_max < test_max:
                max_root = root
                func_max = test_max
    else:
        print("No roots!")
        breakpoint()
    return max_root


def get_polynomial_max(polynomial_coeffics: torch.Tensor, deg: int):
    """
    Returns the largest value of a polynomial that has even term,
    of all the roots of its derivative. I.e., the highest value at a local
    maximum.

    This is useful for calculating the maximum
    of
                   exp(δ'X)

    where δ'Χ = δ_1 x + δ_2 x^2 + ... + δ_m x^m.

    This maximum is one of the roots of the polynomial δ_prime'x.
            δ_1 + 2 δ_2 x + ... + δ_m x^{m-1}

    Therefore, first we construct the coefficients of the derivative
    polynomial, and calculate the corresponding polynomial's roots.

    The resulting set of points will be the stationary points of the
    polynomial, and as such will include the maximiser of the weight
    function.
    """
    # first, build the coefficients for the derivative
    max_root = get_polynomial_maximiser(polynomial_coeffics, deg)
    root_at_powers = torch.pow(max_root, torch.linspace(0, deg - 1, deg))
    return root_at_powers @ polynomial_coeffics


def get_second_deriv_at_root(
    polynomial_coeffics: torch.Tensor, deg: int, root: torch.Tensor
):
    root_at_powers = torch.pow(root, torch.linspace(0, deg - 1, deg))

    # 2 get the second derivative at the maximiser
    first_deriv_powers = torch.linspace(0, deg - 1, deg)
    second_deriv_powers = torch.linspace(0, deg - 2, deg - 1)
    coeffics = (
        first_deriv_powers[1:] * second_deriv_powers * polynomial_coeffics[1:]
    )[1:]
    fprimeprime_root = root_at_powers[: deg - 2] @ coeffics
    return fprimeprime_root


def get_second_deriv_at_max(polynomial_coeffics: torch.Tensor, deg: int):
    """
    Returns f''(x_0)
    """
    maximiser = get_polynomial_maximiser(polynomial_coeffics, deg)
    # print("For the approx integral, using the maximiser:", maximiser)
    return get_second_deriv_at_root(polynomial_coeffics, deg, maximiser)


def calculate_integral_laplace(polynomial_coeffics: torch.Tensor, deg: int):
    """
    Estimates the integral of the weight function via the Laplace method.

    The Laplace method works by noting that a function that can be written
                            exp{Mf(x)}
    for large M will be dominated by the early terms - we can get close by
    approximating with a Taylor expansion to produce a Gaussian integral.

    The formula is:
        ∫ exp{Mf(x)}dx ≈ exp{Mf(x_0)} √(2π/Mf''(x_0))

    where x_0 is the maximiser of the function f. In our case, the function f
    is the polynomial in the exponent of the weight function.
    """
    # 1 get the maximum of the function f
    maximiser = get_polynomial_maximiser(polynomial_coeffics, deg)
    # print("For the approx integral, using the maximiser:", maximiser)
    root_at_powers = torch.pow(maximiser, torch.linspace(0, deg, deg + 1))
    f_max = root_at_powers @ polynomial_coeffics

    # 2 get the second derivative at the maximiser
    # first_deriv_powers = torch.linspace(0, deg, deg + 1)
    # second_deriv_powers = torch.linspace(0, deg - 1, deg)
    # coeffics = (
    # first_deriv_powers[1:] * second_deriv_powers * polynomial_coeffics[1:]
    # )[1:]
    # fprimeprime_max = root_at_powers[2:] @ coeffics
    fprimeprime_max = get_second_deriv_at_max(polynomial_coeffics, deg)
    # print("second_deriv:", fprimeprime_max)
    approx_integral = torch.exp(f_max) * torch.sqrt(
        2 * math.pi / -fprimeprime_max
    )
    return approx_integral


if __name__ == "__main__":
    coeffics = torch.Tensor([0.0, 0.5, 3.5, -4, -20])
    roots = torch.Tensor(
        [
            0.5,
            (math.sqrt(41) - 9) / 20,
            -(9 + math.sqrt(41)) / 20,
        ]
    )
    deg = 5
    test_roots = get_roots(coeffics, deg)
    print("final roots:", test_roots)
    test_max_root = get_polynomial_maximiser(coeffics, deg)

    print("final maximiser of polynomial:", test_max_root)
    poly_max = get_polynomial_max(coeffics, deg)
    exp_poly_max = torch.exp(poly_max)
    print("final maximum of polynomial", poly_max)
    print("final maximum of exponentiated polynomial", exp_poly_max)
    # approx_integral = calculate_integral_laplace(coeffics, deg)
    # print("Final approximate integral:", approx_integral)
    fineness = 1000
    x = torch.linspace(-1.5, 1.5, fineness)
    second_deriv = get_second_deriv_at_max(coeffics, deg)
    # print("Second deriv:", second_deriv)

    approximating_gaussian = (
        1  # / (math.sqrt(2 * math.pi * -second_deriv))
    ) * torch.exp(
        poly_max
        + 0.5
        * get_second_deriv_at_max(coeffics, deg)  # torch.abs(coeffics[-1])
        * (x - test_max_root) ** 2
    )

    plt.plot(x, approximating_gaussian)
    plt.plot(x, exp_of_poly(x, coeffics, deg))
    peaks = get_deriv_roots(coeffics, deg)
    funcsum = torch.zeros(x.shape)
    for root in peaks:
        func1 = gaussian(
            x,
            coeffics,
            root,
            get_second_deriv_at_root(coeffics, deg, root),
        )

        funcsum += func1

        plt.plot(x, func1)
    # plt.plot(x, funcsum * exp_poly_max / max(funcsum))
    plt.show()

    # monte_carlo_version = integrate_function(
    # lambda x: exp_of_poly(x, coeffics, deg),
    # torch.tensor(1.5),
    # exp_poly_max,
    # )
    # print("Monte carlo integral:", monte_carlo_version)
    # # print("exponential max of polynomial:", torch.exp(test_max))
