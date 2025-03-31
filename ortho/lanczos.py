import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from mercergp.kernels import SmoothExponentialKernel


def random_norm1_vector(size):
    v1 = D.Normal(0, 1).sample([size])
    # v1 /= torch.norm(v1)
    return v1 / torch.norm(v1)


def lanczos_tridiagonalisation(A: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Accepts a matrix A, and returns a matrix Q such that QAQ' = T,
        where T is tridiagonal. This may be useful for calculating orthogonal
        polynomial coefficients


    Let AQ = QT.
    Q = [q1, ..., qn] and

        [a1 b1  0,  0,  0]
        [b1 a2 b2,  0,  0]
    T = [0  b2 a3, b3,  0]
        [0   0 b3, a4, b4]
        [0   0  0, b4, a5]

    -> Aqj = b_{j-1}q_{j-1} + a_jq_j + b_j q_{j+1}
    """
    n = A.shape[0]  # the width of A
    v = torch.zeros(rank, rank)
    w = torch.zeros(rank, rank)
    # initialisation
    w1prime = A @ v[1, :]
    a1 = torch.inner(w1prime, v[1, :])
    w[0] = w1prime - a1 @ v[1, :]

    beta = torch.zeros(n)
    j = 0
    for j in range(1, rank):
        beta[j] = torch.norm(w[j - 1])
        if beta[j] != 0:
            v[j] = w[j - 1] / beta[j]
        else:
            v[j]
            # v[j] =

            # for i in range(n):
            # if
    return


if __name__ == "__main__":
    # the program begins here
    fineness = 10
    X = torch.linspace(-3, 3, fineness)
    params = {
        "ard_parameter": torch.Tensor([[1.0]]),
        "variance_parameter": torch.Tensor([1.0]),
        "noise_parameter": torch.Tensor([1.0]),
    }
    K = SmoothExponentialKernel(params)
    matrix = K(X, X)
    rank = 4
    lanczos_tridiagonalisation(matrix, rank)
