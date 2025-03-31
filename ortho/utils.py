import torch
from typing import Callable
import math
import matplotlib.pyplot as plt


def sample_from_function(
    target: Callable,
    end_point: torch.Tensor,
    func_max: torch.Tensor,
    sample_size=2000 ** 2,
):
    proposal_dist = torch.distributions.Uniform(-end_point, end_point)
    uniform_dist = torch.distributions.Uniform(0, 1)
    proposal_density = torch.Tensor([1 / (2 * end_point)])
    M = func_max / proposal_density

    usample = uniform_dist.sample((sample_size,))
    candidate_sample = proposal_dist.sample((sample_size,))
    relevant_locs = usample < target(candidate_sample) / (M * proposal_density)

    candidate_sample = candidate_sample[relevant_locs]
    return candidate_sample


def integrate_function(
    integrand: Callable,
    end_point: torch.Tensor,
    func_max: torch.Tensor,
    sample_size=2000 ** 2,
):
    """
    Integrates a function using Importance sampling - this is for testing
    purposes to see if the thing makes sense.
    """
    # now return the proportion of the points that is under the function...
    candidate_sample = sample_from_function(
        integrand, end_point, func_max, sample_size=2000 ** 2
    )
    proposal_density = torch.Tensor([1 / (2 * end_point)])
    M = func_max / proposal_density
    if sample_size == 1000 ** 2:
        print("INSIDE INTEGRATE FUNCTION WITH TEST!")
        breakpoint()
    # plt.hist(candidate_sample.numpy().flatten(), bins=300)
    # plt.show()
    # breakpoint()
    return M * len(candidate_sample) / sample_size


def get_condition_number(matrix: torch.Tensor):
    """
    Calculates the condition number for a given matrix, using the standard
    pytorch norm.
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix not square"
    matrix_norm = torch.linalg.norm(matrix, 2)
    inverse = torch.linalg.inv(matrix)
    inverse_norm = torch.linalg.norm(inverse, 2)
    # breakpoint()
    return matrix_norm * inverse_norm


def gauss_moment(n: int) -> int:
    """
    Returns the n-th moment of a standard normal Gaussian distribution.
    """
    if n % 2 == 0:
        return double_fact(n - 1)
    else:
        return 0


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


if __name__ == "__main__":
    f = (
        lambda x: (0.4 / math.sqrt(math.pi * 2))
        * torch.exp(-(x.t() ** 2) / 2).squeeze()
        + (0.3 / math.sqrt(math.pi * 2))
        * torch.exp(-((x.t() - 6) ** 2) / 2).squeeze()
        + (0.3 / math.sqrt(math.pi * 2))
        * torch.exp(-((x.t() + 6) ** 2) / 2).squeeze()
    )
    integral = integrate_function(
        f, torch.Tensor([20.0]), 0.7 / (math.sqrt(math.pi * 2))
    )
    print("integral of normal density:", integral)
