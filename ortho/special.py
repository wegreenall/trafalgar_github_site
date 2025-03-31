import matplotlib
import numpy as np
import scipy.special as sc

if __name__ == "__main__":
    matplotlib.use("TkAgg")
import math

import matplotlib.pyplot as plt
import torch
from torch.autograd import Function, gradcheck

# from numpy.polynomial.hermite import hermval

torch.set_default_tensor_type(torch.DoubleTensor)

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 08:34:01 2019

@author: William Greenall
"""
"""
This contains special functions implemented in pytorch, with a view to keeping
them differentiable.
"""


class GammaIncInv(Function):
    @staticmethod
    def forward(ctx, a, input):
        ctx.save_for_backward(input)
        # print("input in forward pass", input)
        ctx.alpha = a
        output = sc.gammaincinv(a, x.detach())  # is this properly vectorised?
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        To implement the backward pass, you want to take the incoming gradient
        from the later layers and apply to it the effect of the current
        operation. This means that the grad_input should be:
        grad_output * {f'(input)} where f is this operation.

        In our case, we can leverage the fact that we are constructing the
        inverse of the incomplete gamma function:
        df^1/dx = 1/(f'(f^-1(x)))

        Since the incomplete gamma function is int_0^x t^(a-1) e^(-t)dt, its
        derivative is:
        x^(a-1)e^(-x). Evaluating this at f^(-1) gives us the result we need.
        :return:
        """
        input = ctx.saved_tensors[0]  # get the input; it keeps a tuple of the
        # saved tensors, so have to used [0]
        # alpha is a parameter, so I stash it this way.
        # See "https://pytorch.org/docs/stable/notes/extending.html)
        a = ctx.alpha

        f_inv = sc.gammaincinv(a, input.detach())  # get f_inv
        grad_input = None

        # if ctx.needs_input_grad[0]:
        f_prime = torch.pow(f_inv, a - 1) * torch.exp(-f_inv) / sc.gamma(a)
        grad_input = grad_output / f_prime

        return None, grad_input


def lbeta(argument, axis=0):
    # sum of log gammas (i.e. product of gammas)
    numerator = torch.sum(torch.lgamma(argument), axis=axis)
    # log gamma of sum of arguments (i.e. gamma of the sum
    denominator = torch.lgamma(torch.sum(argument, axis=axis))
    return numerator - denominator


def hermite_coefficients(N, whole_matrix=False):
    """
    Returns the physicist's polynomials coefficient for a given order
    """
    a = torch.zeros((N + 2, N + 2))

    a[0, 0] = 1
    a[1, 1] = 2

    for n in range(N + 1):
        for k in range(N + 1):
            if k == 0:
                a[n + 1, k] = -a[n, k + 1]
            else:
                a[n + 1, k] = 2 * a[n, k - 1] - (k + 1) * a[n, k + 1]

    # if whole_matrix
    if whole_matrix:
        return_val = a[:-1, :-1]
    else:
        return_val = a[N, :-1]
    return return_val


def hermval(x, c, prob=True):
    """
    This replicates the numpy hermval function for Hermite polynomials but for
    Pytorch tensors.

    :param x:
    :param c:
    :param prob: if True, returns the probabilist's Hermite polynomial;
                 if False, returns the Physicist's.
    :return:
    """
    # print("ABOUT TO HERMVAL!")
    x2 = x * 2  # for the physicist's version
    # c0 = c[-1] - c1*(2*(nd-1))
    if (x2 != x2).any():
        print("Input to hermval contains NaN.")
        breakpoint()
    assert (x2 == x2).all(), "Input to hermval contains NaN."
    # else:
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            if prob:
                c0 = c[-i] - c1 * (nd - 1)
                c1 = tmp + c1 * x
            else:
                c0 = c[-i] - c1 * (2 * (nd - 1))
                c1 = tmp + c1 * x2

            # c1 = torch.where(c1!=c1, torch.zeros(c1.shape), c1)
            if (tmp != tmp).any() or (c0 != c0).any() or (c1 != c1).any():
                print("tmp:", tmp)
                print("c0:", c0)
                print("c1:", c1)
                print("One of the hermval components has become NaN.")

                # c1 = torch.where(c1!=c1, torch.zeros(c1.shape), c1)
                breakpoint()
    if prob:
        return_val = c0 + c1 * x
    else:
        return_val = c0 + c1 * x2

    return_val = torch.where(
        return_val != return_val, torch.zeros(return_val.shape), return_val
    )

    if (return_val != return_val).any():
        breakpoint()
    return return_val


def hermite_function(x, n):
    """
    Returns the value of the Hermitian function
    (i.e. the Gauss-Hermitian function constructed from the Hermitian
    polynomials).

    The Hermite polynomials used here are the physicist's polynomials
    evaluated at x. This is useful for the construction of basis functions.
    :return:
    """

    # might want to do all the following in logs for numerical stability

    # build the coefficient vector needed to get the right hermite polynomial
    # value based on the way hermval gets it
    coeffic_vector = torch.zeros([n + 1])
    coeffic_vector[n] = 1
    hermite_result = hermval(x, coeffic_vector, prob=False)
    if (hermite_result != hermite_result).any():
        print("hermite component has nans!")
        breakpoint()

    return hermite_result


if __name__ == "__main__":
    a = 50
    x = torch.linspace(0.05, 0.99, 1000)
    x.requires_grad = True
    gammaincinv = GammaIncInv.apply
    f = gammaincinv(a, x)
    # plt.plot(x.detach().numpy(), f.detach().numpy())
    print("Function vals:", f)

    # test this
    print("ABout to do a gradcheck test...")
    test = gradcheck(gammaincinv, (a, x), eps=1e-6, atol=1e-4)
    print("Gradcheck test:", test)

    print("Beginning comparisons/plotting")
    # n = 20
    n = 0
    z = torch.linspace(-5, 5, 1000)
    zeros_vector = np.zeros([n])
    coeffic_vector = np.concatenate([zeros_vector, np.array([1])])
    # y0 = hermite_function(z, 0)
    # y1 = hermite_function(z, 1)
    y2 = hermite_function(z, 2)

    def y2prime(x):
        return (4 * (x ** 2) - 2) * torch.exp(-0.5 * (x ** 2))

    def y3prime(x):
        return (8 * (x ** 3) - 12 * x) * torch.exp(-0.5 * (x ** 2))

    y3 = hermite_function(z, 3)
    # y3prime = lambda x:     # y3 = hermite_function(z, 3)
    # y4 = hermite_function(z, 4)
    # y5 = hermite_function(z, 5)

    # plt.plot(z, y0)
    # plt.plot(z, y1)
    plt.plot(z.numpy().flatten(), y2.numpy().flatten(), color="red")
    plt.plot(z.numpy().flatten(), y2prime(z).numpy().flatten(), color="black")
    # plt.plot(z, y0+y1+y2+y3+y4+y5)
    plt.show()

    plt.plot(z.numpy().flatten(), y3.numpy().flatten(), color="red")
    plt.plot(z.numpy().flatten(), y3prime(z).numpy().flatten(), color="black")
    # plt.plot(z, y0+y1+y2+y3+y4+y5)
    plt.show()
    # breakpoint()
    # y2 = wg_hermval(x,n)
