import torch
from ortho.polynomials import chebyshev_first, chebyshev_second
import matplotlib.pyplot as plt

"""
Here we implement the Bjorck and Pereya(1970) algorithm , based in pytorch for 
differentiability.
"""


def vander(x):
    n = len(x)
    vander = torch.zeros(n, n)
    for i, val in enumerate(alpha):
        vander[i, :] = torch.pow(val, torch.linspace(0, n - 1, n))
    return vander


def power_matrix(x, order):
    """
    Returns a matrix of the powers of 'x' up to 'order' for the purpose
    of building mgfs, vandermonde matrices, etc.
    """
    signs = torch.sign(x).repeat(order, 1).t()
    log_x = torch.log(torch.abs(x))
    powers = torch.linspace(0, order - 1, order)
    result = torch.exp(torch.einsum("i,j->ij", log_x, powers))
    return signs * result


def nchoosek(n, k):
    return torch.exp(
        torch.lgamma(torch.tensor(n + 1))
        - (
            torch.lgamma(torch.tensor(k + 1))
            + torch.lgamma(torch.tensor(n - k + 1))
        )
    )


def Bjorck_Pereyra(alpha, f):
    """
    Implements the Bjorck Pereyra algorithm "for the dual system"

        V'a = f

    Where:
        a = (a_0, a_1, ..., a_n)'  the set of coefficients we want to solve for
        f = (f_0, f_1, ..., f_n)'  the polynomial function values we will use
                                   to interpolate with
       V' = |1,         1, ...,     1|  the Vandermonde matrix evaluated at
            |α_0,     α_1, ...,   α_n|  (α_0, α_1, ... α_n)
            |            ...         |
            |α_0^n, α_1^n, ..., α_n^n|

    The algorithm builds up the coefficients by replacing, in each iteration,
    the vector of values with another one that is different by one term.

    First, it builds from c(0) = f -> c(n) = {a vector of divided differences}

    Then it builds from a(n) = c -> a(0) = {the solution}

    each a(0) (or c(0)) is a full, n-sized vector.
    """
    # step 1: set c(0) = f
    n = len(alpha)
    c = f.clone()  # "for k:=0 step 1 until n do a[k] := f[k]"
    for k in range(1, n):
        """
        From the Bjorck Pereyra paper:
        Step(i):
            Put c(0) = f, and, for k = 0,1,...,n-1, compute
            c_j^(k+1) = (c_j^(k) - c_{j-1}^(k))/(α_j-α_{j-k-1} j = k+1, ..., n
            c_j^(k+1) = c_i^(k)  , j = 0, 1, ... k  # i.e. leave it unchanged
        """
        c[k:n] = (c[k:n] - c[k - 1 : n - 1]) / (alpha[k:n] - alpha[0 : n - k])

    # step 2: set a(n) = c
    A = c.clone()
    for k in range(n - 1, 0, -1):  # i..e k decreasing from n-1 to 0
        """
        From the Bjorck Pereyra paper:
        Step(ii):
            Put a^(n) = c and, for k=n-1,...,1,0 compute
            a_j^(k) = a_j^(k+1), j=0, 1, ... ,k-1 , n  (remember, in the first
                                                        instance, k = n-1)
            a_j^(k) = a_j(k+1) - α_κa^{k+1}_{j+1}
        """
        # a(k) is the same as a(k+1) for all j up to n-1
        A[k - 1 : n - 1] = A[k - 1 : n - 1] - alpha[k - 1] * A[k:n]

    """
    From the Bjorck Pereyra paper:
        Now define the lower bidiagonal matrix L_k(α) of order n+1 by:
    """
    return A  # this should be the sequence of coefficients


if __name__ == "__main__":
    n = 4
    lb = -7
    ub = 7
    alpha = torch.linspace(lb, ub, n)
    a = torch.ones(n)  # the coefficients are all 1
    a = torch.distributions.Exponential(1 / 10).sample([n])

    # true_function
    z = torch.linspace(-7, 7, 100)
    true_function = torch.zeros(100)
    function_points = torch.zeros(n)
    for i in range(n):
        true_function += a[i] * z ** i
        function_points += a[i] * alpha ** i  # i.e. just the function points
    plt.plot(z, true_function)
    plt.scatter(alpha, function_points)

    # dual algorithm points
    result = Bjorck_Pereyra(alpha, function_points)  # can we get all ones?

    # now build that polynomial!
    test_polynomial = torch.zeros(len(z))
    for i in range(n):
        test_polynomial += result[i] * z ** i

    plt.plot(z, test_polynomial)
    plt.show()

    # what about vandermonde...
    vander = torch.zeros(n, n)
    for i, val in enumerate(alpha):
        vander[i, :] = torch.pow(val, torch.linspace(0, n - 1, n))

    vanderinv = torch.linalg.inv(vander)
    solved_a = torch.einsum("ij,j->i", vanderinv, function_points)
    print("the true coefficients", a)
    print("WIth vandermonde inverse:", solved_a)
    print("The result from my implementation:", result)
