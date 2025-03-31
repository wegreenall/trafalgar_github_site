# WGK.py
# import pdb
# import opt_einsum as oe
# import matplotlib
import torch
import torch.distributions as D

import matplotlib.pyplot as plt
import math

from ortho.basis_functions import (
    Basis,
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
    standard_chebyshev_basis,
)


"""
This module contains classes for operating kernels related to the Gaussian
processes.

Each kernel has a standard interface, and adding new kernels requires
implementation of the evaluate_kernel method.
The exception to this rule is the Mercer Kernel, which operates in a different
way by being constructed from a set of basis
functions multiplied by the eigenvalues of the "Mercer"  Hilbert-Schmidt
integral operator on the basis functions. That kernel is used in the WGMGP
module for colloquially named Mercer Gaussian Processes.
"""


class KernelError(Exception):
    def __init__(self):
        print("No Available Kernel Specified")
        pass


class UpdateError(Exception):
    def __init__(self):
        print("Update vectors not of same length")
        pass


class StationaryKernel:
    required_kernel_arguments = {"noise_parameter", "variance_parameter"}

    """
    A base class for stationary kernels.

    Stationary kernels depend only on |x-x'|. It contains a method for
    generating a blank gram matrix  w/ data (ie, X-X') which is then
    evaluated at the kernel function (so, exp(-(X-X')(X-X')' / l^2)"""

    def __init__(self, kernel_args: dict):
        """Initialises the kernel

        when subclassing, append to required_args the necessary parameters.

        all elements of the dict kernel_args should be torch.Tensors
        """
        self.check_params(kernel_args)
        self.kernel_args = kernel_args

    def check_params(self, kernel_args):
        if not set(kernel_args.keys()).issuperset(
            self.required_kernel_arguments
        ):
            raise ValueError(
                "The kernel parameters dictionary does not contain all the required parameters."
            )

    def __add__(self, other):
        new_kernel = CompositeKernel([self, other])
        return new_kernel

    def __call__(self, input_points: torch.Tensor, test_points: torch.Tensor):
        """returns a Tensor representing the Gram matrix of the kernel
        function evaluated at the differences between input and test points.

        Tensor shape: bxnxm

        b: batch length
        n: dimension count (so n=2 for 2-d data)
        m: data length
        for each tensor(input or test), the last dimension should
        be the length of the data"""
        input_points = self._validate_shape(input_points)
        test_points = self._validate_shape(test_points)
        data_differences = self.get_data_differences(input_points, test_points)
        evaluated_kernel = self.evaluate_kernel(data_differences)
        return evaluated_kernel

    def _validate_shape(self, points):
        """
        Validates the shape of potential input points to be N x 1 as
        opposed to bare N.
        """
        if len(points.shape) == 1:
            return points.unsqueeze(1)
        else:
            return points

    def kernel_inverse(self, input_points: torch.Tensor):
        """
        Returns the kernel function inverse evaluated at the given input
        points. It receives only one tensor as input since the kernel inverse
        is square.
        """
        kernel = self(input_points, input_points)
        kernel += self.kernel_args["noise_parameter"] ** 2 * torch.eye(
            input_points.shape[0]
        )  # sigma_e
        return torch.inverse(kernel)

    def get_params(self):
        # kernel_params = [self.kernel_args[param] for param in
        # self.kernel_args]
        return self.kernel_args

    @staticmethod
    def get_data_differences(
        input_points: torch.Tensor, test_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Creates the 'data-differences' matrix made up of input points and test
        points, where input is x, test points are y;
        |x1-y1  x1-y2 x1-y3 ... |
        |x2-y1  x2-y2 x2-y3 ... |
        |   .                   |
        |   :                   |

        This is the precursor to constructing a distance (Gram) matrix.

        This allows one to make:the input-point (variance-covariance) block of
        the kernel; the input-test point (covariance) block; and the test-point
        (variance-covariance) block of the Gram matrix.

        Perhaps this operation could be called the outer differences following
        the outer product being a matrix/tensor whose elements are the products
        of all the pairs of individual elements of the two vectors.
        The result of this function has elements that are the differences of
        all the pairs of individual elements in the two argument vectors.

        :param input_points:
        :param test_points:
        :return:
        """
        # The data dimension should be 0 and we should work based on that.
        tp_shape = test_points.shape  # should be the data dimension
        ip_shape = input_points.shape  # should be the data dimension
        #    test_points = test_points.reshape([-1,1])
        input_points_repeated = input_points.repeat(tp_shape[0], 1, 1)
        test_points_repeated = test_points.repeat(ip_shape[0], 1, 1)
        # test_points_repeated_transpose = torch.einsum('ijk->jik',\
        # test_points_repeated)
        test_points_r_t = torch.einsum("ijk->jik", test_points_repeated)
        vector_diffs = input_points_repeated - test_points_r_t  # data
        return vector_diffs

    def evaluate_kernel(self, data_differences):
        """This method should be overwritten to evaluate the function
        for the kernel.
        :param data_differences: a matrix of (x-x') points over which to
                                 calculate the VCV matrix.
        :return: the Gram matrix; the input matrix provided but evaluated by
                                  the kernel function,
        e.g. e^-(1/2 (x-x')Σ^-1 (x-x')')
        """
        return NotImplementedError


class ProductKernel(StationaryKernel):
    def __init__(
        self, left_kernel: StationaryKernel, right_kernel: StationaryKernel
    ):
        self.left_kernel = left_kernel
        self.right_kernel = right_kernel
        return

    def __call__(self, input_points: torch.Tensor, test_points: torch.Tensor):
        left_matrix = self.left_kernel(input_points, test_points)
        right_matrix = self.right_kernel(input_points, test_points)
        return left_matrix * right_matrix

    def _validate_shape():
        return


class SmoothExponentialKernel(StationaryKernel):
    required_kernel_arguments = {
        "noise_parameter",
        "variance_parameter",
        "ard_parameter",
    }

    def evaluate_kernel(self, data_differences):
        # parameters
        ard = self.kernel_args["ard_parameter"]
        sigma = self.kernel_args["variance_parameter"]
        ard_inv = torch.inverse(ard)

        try:
            interim = torch.einsum(
                "ijk,kl,ijk->ij", data_differences, ard_inv, data_differences
            )

        except RuntimeError:
            if data_differences.shape[2] != ard_inv.shape[0]:
                print("Will add better error handling here later -")
            #   at the moment it looks like\
            #   the dimensions of the inverse and the tensor-difference matrix
            #   do not match.")
            breakpoint()

        half_interim = -interim / 2
        kernels = sigma * torch.exp(half_interim)
        return kernels


# class HeteroskedasticSmoothExponentialKernel(StationaryKernel):


class ExponentialKernel(StationaryKernel):  # need to fix this matrix's 'n'
    def evaluate_kernel(self, data_differences):
        # parameters
        ard = self.kernel_args["ard_parameter"]
        sigma = self.kernel_args["variance_parameter"]
        theta = torch.pow(ard, 2)

        # get the differences of the vectors for the kernel
        kernels = sigma * torch.exp(
            -torch.sum(torch.abs(data_differences), 1) / theta
        )  # check this is correct
        return kernels


class RationalQuadraticKernel(StationaryKernel):
    required_kernel_arguments = {
        "ard_parameter",
        "variance_parameter",
        "mixture_parameter",
    }

    def evaluate_kernel(self, data_differences):
        # parameters
        ard = self.kernel_args["ard_parameter"]
        sigma = self.kernel_args["variance_parameter"]
        alpha = self.kernel_args["mixture_parameter"]
        # theta = torch.pow(ard, 2)
        ard_inv = torch.inverse(ard)

        # print("in evaluate_kernel for RationalQuadratic")
        interim = torch.einsum(
            "ijk,kl,ijk->ij", data_differences, ard_inv, data_differences
        )

        # get the differences of the vectors for the kernel
        kernels = sigma * torch.pow((1 + interim / (2 * alpha)), -alpha)
        return kernels


class ConstantKernel(StationaryKernel):
    def evaluate_kernel(self, data_differences):
        sigma = self.kernel_args["variance_parameter"]
        kernels = sigma * torch.ones(data_differences.squeeze().shape)
        return kernels


class NoiseKernel(StationaryKernel):
    def evaluate_kernel(self, data_differences):
        sigma = self.kernel_args["variance_parameter"]
        kernels = sigma * torch.where(
            data_differences.squeeze() == 0,
            torch.ones(data_differences.squeeze().shape),
            torch.zeros(data_differences.squeeze().shape),
        )
        return kernels


class MercerKernel(StationaryKernel):
    required_kernel_arguments = {
        # "ard_parameter",
        # "precision_parameter",
        "noise_parameter",
        "variance_parameter",
    }

    def __init__(
        self, order: int, basis: Basis, eigenvalues: torch.Tensor, kernel_args
    ):
        """
        Initialises a Mercer Kernel approximation.

        The  Mercer Kernel approximation builds an approximate kernel by
        utilising the Mercer expansion. If you have an eigensystem {λ_l, φ_l},
        you can construct k(x_i,x_j) ~= Σ_l^m λ_l φ_l(x_i) φ_l(x_j).

        : param m: the number of terms in the summation, and the highest
        degree of the basis functions used for approximating the kernel

        :param basis function: a callable with signature (x, m, params) that
        returns the value of the basis function for a given degree 'i',
         at value 'x', with parameters 'params'.

        eigenvalue_function: a function with signature (m, params) producing a
                             tensor the eigenvalues in shape (degree,
                              dimension)

        kernel_args: the parameters we will use to construct the trained
                       kernel version
        """

        self.order = order  # the degree of the approximation

        # basis_function expected to have signature:
        # (x:torch.Tensor, degree: int, basis_args: dict)
        self.basis = basis

        self.eigenvalues = eigenvalues
        # for param in self.required_kernel_arguments:
        # if param not in kernel_args:
        # raise KeyError(str(param) + " not found in kernel args")

        if isinstance(kernel_args, dict):
            parameters_list = [kernel_args]
        elif isinstance(kernel_args, list) or isinstance(kernel_args, tuple):
            parameters_list = kernel_args
            if len(kernel_args) != self.basis.dimension:
                breakpoint()
                raise ValueError(
                    "The number of parameter dicts passed must match the \
                     dimension parameter of the eigenvalue generator"
                )
            for params in parameters_list:
                for param in self.required_kernel_arguments:
                    if param not in params:
                        raise ValueError(
                            "The parameter "
                            + param
                            + " is required but not found"
                        )
        self.kernel_args = kernel_args

    @staticmethod
    def get_data_differences(
        input_points: torch.Tensor, test_points: torch.Tensor
    ):
        return input_points, test_points

    def evaluate_kernel(self, data_differences):
        """
        Returns the gram matrix of the kernel evaluated at a given input,
        test point combination.


        :param data_differences:
        :return:
        """

        input_points, test_points = data_differences

        """
        The construction of the kernel approximation is done in stages.
        The kernel is represented as:
        Σ λn en(zi)en(zj)

        In order to build this, first we build en(zi) and en(zj). These
        are vectors of length {dimension} and they are essentially ωt ωτ'
        """
        # first, evaluate the eigenfunctions at the vector inputs
        # so φ(input_points), φ(test_points)
        # should be of size (input_size, degree), (test_size, degree)
        input_ksi = self.get_ksi(input_points)
        test_ksi = self.get_ksi(test_points).T
        diag_l = torch.diag(self.eigenvalues)
        # intermediate_term = torch.mm(input_ksi, diag_l)

        # transposing because I appear to have the shape wrong
        # kernel = torch.mm(intermediate_term, test_ksi).t()

        # matrix = torch.zeros(input_ksi.shape[0], test_ksi.shape[1])
        # for i,e in enumerate(self.eigenvalues):
        #     matrix += e * torch.outer(input_ksi[:,i] , test_ksi[i, :])
        if len(input_ksi.shape) != 2 or len(test_ksi.shape) != 2:
            breakpoint()

        if self.basis.dimension == 1:
            try:
                kernel = self.kernel_args["variance_parameter"] * torch.einsum(
                    "ij,jk...,kl -> il...", input_ksi, diag_l, test_ksi
                )
            except RuntimeError as e:
                print("Calculating the kernel failed; the error was:")
                print(e)
                breakpoint()

        else:
            net_variance_parameter = torch.prod(
                torch.Tensor(
                    [args["variance_parameter"] for args in self.kernel_args]
                )
            )
            kernel = net_variance_parameter * torch.einsum(
                "ij,jk...,kl -> il...", input_ksi, diag_l, test_ksi
            )
        return kernel

    def get_ksi(self, input_points):
        """
        Constructs and returns Ξ, which is the matrix whose rows are a
        given degree of basis function, evaluated at the input points vector

        specifically, Ξ is the Nxm matrix of the m basis functions
        evaluated at N data points

        : param input_points: the set of input points at which to
        evaluate the basis functions

        we create this method in case it is necessary to complete extra pre-
        or post- calculations on this basis
        """
        ksi = self.basis(input_points)
        return ksi

    def kernel_inverse(self, input_points: torch.Tensor):
        """
        Returns the inverse of the kernel Gram matrix evaluated at
        tensor input_points,
        using the Sherman-Morrison-Woodbury Formula to get the matrix inverse.

        The matrix inverse is built as σ^-2(Ι - Ξ(σ^2 Λ^-1  + Ξ'Ξ)^(-1)Ξ')
        where Ξ is an Nxm feature matrix, and Λ is an mxm diagonal matrix
        of operator eigenvalues.
        """
        if self.basis.dimension == 1:
            sigma_e = self.kernel_args[
                "noise_parameter"
            ]  # get noise parameter
        else:
            sigma_e = self.kernel_args[0]["noise_parameter"]

        # inv_diag_l = torch.diag(1/self.eigenvalues)  # Λ^-1

        ksi = self.get_ksi(input_points)  # Ξ
        interim_inv = self.get_interim_matrix_inverse(input_points)

        # 1/σ^2 (Ι - Ξ(σ^2 Λ^-1 + Ξ'Ξ)^{-1}Ξ')
        kernel_inv = (
            1
            / (sigma_e**2)
            * (torch.eye(input_points.shape[0]) - ksi @ interim_inv @ ksi.t())
        )
        return kernel_inv

    def get_interim_matrix_inverse(self, input_points):
        """
        Returns the (σ^2 Λ^-1 + Ξ'Ξ)^{-1} matrix as required in the kernel_inverse
        method (and in general by the WSM formula).
        """
        ksi = self.get_ksi(input_points)
        if self.basis.dimension == 1:
            sigma_e = self.kernel_args["noise_parameter"]
        else:
            sigma_e = self.kernel_args[0]["noise_parameter"]

        inv_diag_l = torch.diag(1 / self.eigenvalues)  # Λ^-1
        ksiksi = torch.mm(ksi.T, ksi)  # Ξ'Ξ

        # ( σ^(2) Λ^-1 + Ξ'Ξ)
        interim_matrix = (sigma_e**2) * inv_diag_l + ksiksi
        # ( σ^(2) Λ^-1 + Ξ'Ξ)^(-1)
        interim_inv = torch.inverse(interim_matrix)
        return interim_inv

    def set_eigenvalues(self, new_eigenvalues):
        """
        Allows for direct setting of the eigenvalues for the Mercer kernel
        representation. This allows them to be trained/solved for externally
        and then passed to a model for prediction.
        """
        self.eigenvalues = new_eigenvalues

        # fix infinite eigenvalues, for zero-d mercer eigenvalue inverses.
        mask = torch.nonzero((self.eigenvalues.isinf()))
        self.eigenvalues[mask] = 0
        return

    def get_eigenvalues(self):
        """
        Getter method to recall the current values of the eigenvalues for the
        Mercer kernel representation.
        """
        return self.eigenvalues

    def update_params(self, basis, eigenvalues, variance_parameter):
        self.set_eigenvalues(eigenvalues)
        self.basis = basis

    def set_variance(self, variance_parameter: torch.Tensor):
        """
        Allows for direct setting of the variance parameter for the Mercer kernel.
        """
        self.kernel_args["variance_parameter"] = variance_parameter

    def set_noise(self, noise_parameter: torch.Tensor):
        """
        Allows for direct setting of the noise parameter for the Mercer kernel.
        """
        self.kernel_args["noise_parameter"] = noise_parameter


class FasshauerKernel(MercerKernel):
    """
    An implementation of the Fasshauer-basis Mercer kernel for rapid
    prototyping.

    I assume in this case that I am working in 1 dimension.
    """

    def __init__(self, order, kernel_args):
        dimension = 1
        basis = Basis(
            smooth_exponential_basis_fasshauer, dimension, order, kernel_args
        )
        eigenvalues = smooth_exponential_eigenvalues_fasshauer(
            order, kernel_args
        )
        super().__init__(order, basis, eigenvalues, kernel_args)
        return


class RandomFourierFeaturesKernel(MercerKernel):
    required_kernel_arguments = {"noise_parameter", "variance_parameter"}

    def __init__(
        self,
        feature_count: int,
        sampling_distribution: torch.distributions.Distribution,
        dim: int,
        kernel_args: dict,
    ):
        """Sets up the Random Fourier feature kernel;
        it pre-generates ω and b.
        """
        self.sampling_distribution = sampling_distribution
        self.dim = dim
        self.order = feature_count
        self.feature_count = feature_count
        self.b_sample_shape = torch.Size([feature_count])
        self.w_sample_shape = torch.Size([feature_count, dim])
        self.kernel_args = kernel_args

        self.b_dist = D.Uniform(0.0, 2 * math.pi)
        self.w_dist = sampling_distribution
        self._generate_kernel()

    def _generate_kernel(self):
        """
        Generates the ω and b vectors for this specific kernel.
        """
        self.w = self.w_dist.sample(self.w_sample_shape)
        self.b = self.b_dist.sample(self.b_sample_shape)

    def __call__(self, input_points, test_points):
        """
        Evaluates the kernel.

        input shape: input_points: [N, d]
                     test_points:  [N, d]

        w shape: [d,  1]
        b shape: [d,  1]

        output shape: [N, N]

        """
        z = self.basis(input_points)
        y = self.basis(test_points)
        zz = z.t() @ y
        return zz

    def basis(self, input_points):
        """
        Evaluates the features basis at the input points.

        input_shape:  [N, 1]
        output_shape: [N, d]
        """

        n = input_points.shape[0]
        b = self.b.repeat(n, 1).t()  # R * n
        z = math.sqrt(2.0 / self.feature_count) * torch.cos(
            (self.w @ input_points.t()).squeeze() + b
        )
        return z

    def get_ksi(self, input_points):
        return self.basis(input_points)

    def kernel_inverse(self, input_points):
        return torch.inverse(self(input_points, input_points))

    def get_interim_matrix_inverse(self, input_points):
        return torch.inverse(
            self(input_points, input_points)
            + self.kernel_args["noise_parameter"] ** 2
            * torch.eye(input_points.shape[0])
        )


class SmoothExponentialRFFKernel(RandomFourierFeaturesKernel):
    def __init__(self, feature_count, dim):
        super().__init__(
            feature_count, D.Normal(torch.zeros(dim), torch.ones(dim)), dim
        )


# Composite/exotic kernels
class CompositeKernel(StationaryKernel):
    def __init__(self, kernels):
        """
        the CompositeKernel is a tool for combining kernels via either
        summation or multiplication, since sums of kernels are kernels, as are
        products of kernels. I think this makes kernels a ring (?)

        Largely a WIP

        :param kernels: an iterable containing kernels
        """
        self.kernel_args = [kernel.kernel_args for kernel in kernels]
        self.kernels = kernels

    # def __add__(self, other):
    #     new_kernel = CompositeKernel([self, other])
    #     return new_kernel

    def evaluate_kernel(self, data_differences):
        # i.e. return something that is 2-d, i.e. a matrix
        result = torch.zeros(data_differences.shape).squeeze()
        for kernel in self.kernels:
            new_kernel_value = kernel.evaluate_kernel(data_differences)
            result += new_kernel_value

        return result

    def kernel_inverse(self, input_points: torch.Tensor):
        # how will i do multiple kernels?
        return super().kernel_inverse(input_points)

    def get_params(self):
        """
        Adapts the get_params method to handle the fact that it is composite.
        :return:
        """
        kernel_params = []
        for kernel in self.kernels:
            kernel_params.extend(kernel.get_params())  # essentially,
            # concatenate the lists of  kernel parameters for each kernel in
            # the composite kernel.
        return kernel_params


if __name__ == "__main__":
    test_rff = False
    test_chebys = False
    test_mercer_kernel = True
    l_se = torch.Tensor([[2]])
    sigma_se = torch.Tensor([3])
    sigma_e = torch.Tensor([1])
    epsilon = torch.Tensor([1])
    if test_chebys:
        # generate a kernel and show it as a heatmap
        l_se = torch.Tensor([[2]])
        sigma_se = torch.Tensor([3])
        sigma_e = torch.Tensor([1])
        epsilon = torch.Tensor([1])
        lb = 0.0
        ub = 10.0
        chebyshev_args = {
            "upper_bound": torch.tensor(ub, dtype=torch.float32),
            "lower_bound": torch.tensor(lb, dtype=torch.float32),
        }

        mercer_args = {
            "ard_parameter": l_se,
            "variance_parameter": sigma_se,
            "noise_parameter": sigma_e,
            "precision_parameter": epsilon,
        }

        test_points = torch.linspace(lb + 0.01, ub - 0.01, 100)
        for order in range(5, 75, 20):
            basis = Basis(standard_chebyshev_basis, 1, order, chebyshev_args)
            eigenvalues = smooth_exponential_eigenvalues_fasshauer(
                order, mercer_args
            )
            kernel = MercerKernel(order, basis, eigenvalues, mercer_args)
            result = kernel(test_points, test_points)
            plt.imshow(result, cmap="viridis")
            plt.show()

    if test_rff:
        order = 1000
        order = 1000
        rff_kernel = SmoothExponentialRFFKernel(order, 1)
        x_axis = torch.linspace(-3, 3, 500).unsqueeze(1)
        kernel_matrix = rff_kernel(x_axis, x_axis)
        plt.imshow(kernel_matrix)
        plt.colorbar()
        plt.show()

    if test_mercer_kernel:
        fineness = 500
        # get the standard kernel
        x_axis = torch.linspace(-3, 3, fineness).unsqueeze(1)

        def smooth_exponential_kernel(x, y):
            return torch.exp(-torch.cdist(x, y) ** 2 / 2)

        kernel_matrix = smooth_exponential_kernel(
            x_axis, x_axis
        ) + 1e-6 * torch.eye(x_axis.shape[0])
        plt.imshow(kernel_matrix)
        plt.show()

        # mercer kernel
        mercer_args = {
            "ard_parameter": l_se,
            "variance_parameter": sigma_se,
            "noise_parameter": sigma_e,
            "precision_parameter": epsilon,
        }
        print(torch.linalg.det(kernel_matrix))
        chol = torch.linalg.cholesky(kernel_matrix)
        z = D.Normal(torch.zeros(fineness), torch.ones(fineness)).sample()
        gp_sample = chol @ z
        plt.plot(x_axis, gp_sample)
        plt.show()
        breakpoint()

        order = 15
        basis = Basis(
            smooth_exponential_basis_fasshauer, 1, order, mercer_args
        )
        eigenvalues = smooth_exponential_eigenvalues_fasshauer(
            order, mercer_args
        )
        mercer_kernel = MercerKernel(
            order,
            basis,
            eigenvalues,
            mercer_args,
        )
        mercer_kernel_matrix = mercer_kernel(x_axis, x_axis)
        plt.imshow(mercer_kernel_matrix)
        plt.show()
