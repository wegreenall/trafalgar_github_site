from math import floor
import torch
import torch.distributions as D
from ortho.measure import MaximalEntropyDensity
from ortho.orthopoly import (
    OrthogonalPolynomial,
    OrthonormalPolynomial,
    SymmetricOrthogonalPolynomial,
    SymmetricOrthonormalPolynomial,
)
from ortho.polynomials import ProbabilistsHermitePolynomial, HermitePolynomial
from ortho.basis_functions import OrthonormalBasis
from ortho.utils import sample_from_function, integrate_function, gauss_moment
from typing import Callable
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt
from termcolor import colored
from typing import Tuple, Union
from enum import Enum

"""
This file contains builder functions for various components
required for building out the Favard kernel.
"""


class OrthoBuilderState(Enum):
    """
    Enum for the state of the builder.
    """

    EMPTY = 0
    SAMPLE = 1
    MOMENTS = 2
    BETAS_GAMMAS = 4


class OrthoBuilder:
    def __init__(self, order, dim=1):
        """
        This class is a combination Builder and Factory
        for the various forms of Orthogonal polynomial that one
        might want.

        It operates as a state machine that maintains an indexing state for
        the data that is available. For example, if you set a sample, the class
        will automatically calculate the moments and set the state to MOMENTS,
        then calculate the betas and gammas and set the state to BETAS_GAMMAS.
        This way, we can avoid misguided setting.
        """
        self.order = order
        self.dim = dim

        # state holder
        self.state = OrthoBuilderState.EMPTY

        # data
        self.sample = None
        self.moments = None
        self.weight_function = None
        self.modifying_polynomial = None

        # parameters
        self.betas = None
        self.gammas = None

    def set_sample(self, sample: torch.Tensor):
        """
        External setter for sample.
        Since moments are aggregated from the sample,
        this method also sets the moments.

        Builder method: returns self.
        """
        if self.state == OrthoBuilderState.EMPTY:
            self.sample = sample
            self.state = OrthoBuilderState.SAMPLE
            return self.set_moments(None)
        else:
            raise ValueError(
                "Moments have already been set; adding moments is inconsistent"
            )

    def set_moments(self, moments: Union[torch.Tensor, None]):
        """
        External setter for moments.

        Builder method: returns self.
        """
        if self.state == OrthoBuilderState.SAMPLE:
            assert (
                moments is None
            ), "Sample has already been set, don't set moments as well"
            self.moments = get_moments_from_sample(self.sample, 2 * self.order)
        elif self.state == OrthoBuilderState.EMPTY:
            self.moments = moments

        # update the state and progress to the next stage.
        self.state = OrthoBuilderState.MOMENTS
        return self.set_betas_and_gammas(None, None)

    def set_betas_and_gammas(
        self,
        betas: Union[torch.Tensor, None],
        gammas: Union[torch.Tensor, None],
    ):
        """
        Setter for betas and gammas.

        if a modifying polynomial has been set, this will produce
        the "modified moments" betas and gammas as described
        in the Gautschi paper.

        Builder method: returns self.
        """
        if self.state == OrthoBuilderState.MOMENTS:
            if self._check_modifying_polynomial():
                get_gammas_betas_from_modified_moments_gautschi(
                    self.moments, self.order, self.modifying_polynomial
                )
            else:
                self.betas, self.gammas = get_gammas_betas_from_moments(
                    self.moments, self.order
                )
        elif self.state == OrthoBuilderState.EMPTY:
            self.betas, self.gammas = betas, gammas
        else:
            raise ValueError(
                "Tried to set betas and gammas but not empty.\
                             State is: {}".format(
                    self.state
                )
            )
        assert self.betas is not None and self.gammas is not None
        self.state = OrthoBuilderState.BETAS_GAMMAS
        return self

    def set_weight_function(self, weight_function: Callable):
        """
        External setter for weight_function.

        Builder method: returns self.
        """
        self.weight_function = weight_function
        return self

    def set_modifying_polynomial(
        self, modifying_polynomial: OrthogonalPolynomial
    ):
        """
        External setter for modifying_polynomial.

        If modifying_polynomial has been set, betas and gammas are calculated
        using the modified moments. See Gautschi paper for details.

        Builder method: returns self.
        """
        self.modifying_polynomial = modifying_polynomial
        return self

    def _check_weight_function(self) -> bool:
        return self.weight_function is not None

    def _check_modifying_polynomial(self) -> bool:
        return self.modifying_polynomial is not None

    def get_orthogonal_polynomial(self) -> OrthogonalPolynomial:
        """
        If the builders is ready (i.e. has betas and gammas)
        and has a weight function, we can build an orthgonal basis
        """
        if self.state == OrthoBuilderState.BETAS_GAMMAS:
            return OrthogonalPolynomial(self.order, self.betas, self.gammas)
        else:
            raise ValueError(
                "Not ready to build orthogonal polynomial. Current State: {} ".format(
                    self.state
                )
            )

    def get_orthonormal_polynomial(self) -> OrthonormalPolynomial:
        if self.state == OrthoBuilderState.BETAS_GAMMAS:
            return OrthonormalPolynomial(self.order, self.betas, self.gammas)
        else:
            raise ValueError(
                "Not ready to build orthonormal polynomial. Current State: {} ".format(
                    self.state
                )
            )

    def get_orthonormal_basis(self) -> OrthonormalBasis:
        """
        If the builders is ready (i.e. has betas and gammas AND weight
        function), we can build an orthonormal basis.
        """
        if (
            self._check_weight_function()
            and self.state == OrthoBuilderState.BETAS_GAMMAS
        ):  # i.e. we're ready
            return OrthonormalBasis(
                OrthonormalPolynomial(self.order, self.betas, self.gammas),
                self.weight_function,
                self.dim,
                self.order,
            )
        else:
            raise ValueError(
                "Not ready to build orthonormal basis. Current State: {} ".format(
                    self.state
                )
            )

    def get_symmetric_orthogonal_polynomial(
        self,
    ) -> SymmetricOrthogonalPolynomial:
        if self.state == OrthoBuilderState.BETAS_GAMMAS:
            return SymmetricOrthogonalPolynomial(self.order, self.gammas)
        else:
            raise ValueError(
                "Not ready to build symmetric orthogonal polynomial. Current State: {} ".format(
                    self.state
                )
            )

    def get_symmetric_orthonormal_polynomial(
        self,
    ) -> SymmetricOrthonormalPolynomial:
        if self.state == OrthoBuilderState.BETAS_GAMMAS:
            return SymmetricOrthonormalPolynomial(self.order, self.gammas)
        else:
            raise ValueError(
                "Not ready to build symmetric orthonormal polynomial. Current State: {} ".format(
                    self.state
                )
            )


def get_orthonormal_basis_from_sample_multidim(
    input_sample: torch.Tensor,
    weight_functions: Callable,
    order: int,
    parameters=None,
) -> OrthonormalBasis:
    """
    Returns an OrthonormalBasis made of a set of functions
        φ_i = c_i P_i w^{1/2}.
    For multiple dimensions, it expects a tuple of weight functions
    of the same size as the number of dimensions,
    in order to capture the ability to have different functions in
    different dimensions
    """
    if len(input_sample.shape) > 1:
        dimension = input_sample.shape[-1]
    else:
        dimension = 1

    if isinstance(weight_functions, Callable):
        weight_functions = (weight_functions,)

    if len(weight_functions) != dimension:
        raise ValueError(
            "The number of weight functions passed in must match the dimension parameter"
        )

    polynomials = []
    for d in range(dimension):
        betas, gammas = get_gammas_betas_from_moments(
            get_moments_from_sample(
                input_sample[:, d], 2 * order, weight_functions[d]
            ),
            order,
        )
        polynomials.append(OrthonormalPolynomial(order, betas, gammas))
    return OrthonormalBasis(
        polynomials, weight_functions, dimension, order, parameters
    )


"""
ONE DIMENSIONAL VERSION TO BE SAVED
"""


def get_orthonormal_basis_from_sample(
    input_sample: torch.Tensor,
    weight_function: Callable,
    order: int,
    parameters=None,
) -> OrthonormalBasis:
    """
    Returns an OrthonormalBasis made of a set of functions
        φ_i = c_i P_i w^{1/2}.
    For multiple dimensions, it expects a tuple of weight functions
    of the same size as the number of dimensions,
    in order to capture the ability to have different functions in
    different dimensions
    """
    betas, gammas = get_gammas_betas_from_moments_gautschi(
        get_moments_from_sample(input_sample, 2 * order, weight_function),
        order,
    )

    poly = OrthonormalPolynomial(order, betas, gammas)
    return OrthonormalBasis(poly, weight_function, 1, order)


def get_orthonormal_basis(
    betas: torch.Tensor,
    gammas: torch.Tensor,
    order: int,
    weight_function: Callable,
) -> OrthonormalBasis:
    """
    For given order, betas and gammas, generates
    an OrthonormalBasis.
    """
    poly = OrthonormalPolynomial(order, betas, gammas)
    # weight_function = MaximalEntropyDensity(order, betas, gammas)
    return OrthonormalBasis(poly, weight_function, 1, order)


def get_symmetric_orthonormal_basis(order, gammas) -> OrthonormalBasis:
    """
    For given order, betas and gammas, generates
    an OrthonormalBasis.
    """
    poly = SymmetricOrthonormalPolynomial(order, gammas)
    weight_function = MaximalEntropyDensity(
        order, torch.zeros(2 * order), gammas
    )
    return OrthonormalBasis(poly, weight_function, 1, order)


def get_weight_function_from_sample(
    sample: torch.Tensor, order
) -> "MaximalEntropyDensity":
    """
    For a given sample, returns the maximal entropy weight function that
    correponds to the moments that the sample has - i.e. it creates the
    maximal entropy weight function corresponding to the weight function.
    """
    gammas = get_gammas_from_sample(sample, 2 * order)
    return MaximalEntropyDensity(order, torch.zeros(2 * order), gammas)


def get_moments_from_function(
    target: Callable,
    end_point: torch.Tensor,
    func_max: torch.Tensor,
    order: int,
    sample_size=2000**2,
):
    sample = sample_from_function(
        target, end_point, func_max, sample_size
    ).squeeze()
    return get_moments_from_sample(sample, order)


def get_moments_from_sample_logged(
    sample: torch.Tensor,
    moment_count: int,
    weight_function=lambda x: torch.ones(x.shape),
) -> torch.Tensor:
    """
    Returns a sequence of _moment_count_ moments calculated from a sample -
    including the first element which is the moment of order 0.

    Note that the resulting sequence will be of length order + 1, but we need
    2 * order to build _order_ gammas and _order_ betas,
    so don't forget to take this into account when using the function.

    Example: I need to calculate 10 betas and 10 gammas: (i.e. I want to
    build the basis up to order 10). I take:
    moments = get_moments_from_sample(sample, 20, weight_function)
    """
    stretched_sample = torch.einsum(
        "i,ij->ij",
        sample,
        torch.ones(sample.shape[0], moment_count + 1),
    )
    exponents = torch.linspace(0, moment_count, moment_count + 1)
    signs = torch.sign(stretched_sample) ** exponents
    logged_stretched_sample = torch.log(torch.abs(stretched_sample))

    # powered_sample = torch.pow(stretched_sample, exponents)
    # logged_powered_sample = torch.einsum(
    # "ij,j -> ij", stretched_sample, exponents
    # )
    logged_powered_sample = logged_stretched_sample * exponents
    powered_sample = torch.exp(logged_powered_sample) * signs  # ** exponents
    moments = torch.mean(powered_sample, axis=0)
    # moments = torch.cat((torch.Tensor([1.0]), moments))
    return moments


def get_orthogonal_moments_from_sample(
    sample: torch.Tensor,
    moment_count: int,
    orthogonal_polynomial: OrthogonalPolynomial,
    weight_function=lambda x: torch.ones(x.shape),
) -> torch.Tensor:
    """
    Returns a sequence of _moment_count_ ORTHOGONAL moments calculated from a sample -
    including the first element which is the moment of order 0.

    Note that the resulting sequence will be of length order + 1, but we need
    2 * order to build _order_ gammas and _order_ betas,
    so don't forget to take this into account when using the function.

    Example: I need to calculate 10 betas and 10 gammas: (i.e. I want to
    build the basis up to order 10). I take:
    moments = get_moments_from_sample(sample, 20, weight_function)
    """
    params = dict()
    orthogonal_moments = torch.zeros(moment_count)
    weighted_sample = weight_function(sample)
    for deg in range(moment_count):
        print("current deg", deg)
        orthogonal_sample = orthogonal_polynomial(sample, deg, params)
        orthogonal_moments[deg] = torch.mean(
            orthogonal_sample * weighted_sample
        )
    return orthogonal_moments


def get_moments_from_sample(
    sample: torch.Tensor,
    moment_count: int,
    weight_function=lambda x: torch.ones(x.shape),
) -> torch.Tensor:
    """
    Returns a sequence of _moment_count_ + 1 moments calculated from a sample -
    including the first element which is the moment of order 0.

    Note that the resulting sequence will be of length order + 1, but we need
    2 * order to build _order_ gammas and _order_ betas,
    so don't forget to take this into account when using the function.

    Example: I need to calculate 10 betas and 10 gammas: (i.e. I want to
    build the basis up to order 10). I take:
    moments = get_moments_from_sample(sample, 20, weight_function)
    """
    stretched_sample = torch.einsum(
        "i,ij->ij", sample, torch.ones(sample.shape[0], moment_count)
    )
    exponents = torch.linspace(0, moment_count - 1, moment_count)
    powered_sample = torch.pow(stretched_sample, exponents)
    stretched_weight = torch.einsum(
        "i,ij->ij",
        weight_function(sample),
        torch.ones(sample.shape[0], moment_count),
    )
    moments = torch.mean(powered_sample * stretched_weight, axis=0)
    return moments


# def get_moments_from_sample(
# sample: torch.Tensor,
# moment_count: int,
# weight_function=lambda x: torch.ones(x.shape),
# ) -> torch.Tensor:
# """
# Returns a sequence of _moment_count_ moments calculated from a sample - including
# the first element which is the moment of order 0.

# Note that the resulting sequence will be of length order + 1, but we need
# 2 * order to build _order_ gammas and _order_ betas,
# so don't forget to take this into account when using the function.

# Example: I need to calculate 10 betas and 10 gammas: (i.e. I want to
# build the basis up to order 10). I take:
# moments = get_moments_from_sample(sample, 20, weight_function)

# """
# powers_of_sample = sample.repeat(
# moment_count + 1, 1
# ).t() ** torch.linspace(0, moment_count, moment_count + 1)

# weight = weight_function(sample).repeat(moment_count + 1, 1).t()
# estimated_moments = torch.mean(powers_of_sample * weight, dim=0)

# # build moments
# # moments = torch.zeros(2 * moment_count + 2)
# # moments[0] = 1
# # for i in range(1, 2 * moment_count + 2):
# # if i % 2 == 0:  # i.e. even
# # moments[i] = estimated_moments[i]

# # breakpoint()
# return estimated_moments


def get_gammas_from_moments(moments: torch.Tensor, order: int) -> torch.Tensor:
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the gammas that correspond to them;
    i.e., the gammas from the orthogonal polynomial series that
    is orthogonal w.r.t the linear moment functional with those moments.
    """
    dets = torch.zeros(order + 2)
    dets[0] = dets[1] = 1.0
    gammas = torch.zeros(order)
    assert (
        len(moments) >= 2 * order
    ), "Please provide at least 2 * order moments. Don't forget to include the zeroth moment"
    for i in range(order):
        hankel_matrix = moments[: 2 * i + 1].unfold(0, i + 1, 1)  # [1:, :]
        dets[i + 2] = torch.linalg.det(hankel_matrix)

    gammas = dets[:-2] * dets[2:] / (dets[1:-1] ** 2)
    return gammas


def get_betas_from_moments(moments: torch.Tensor, order: int) -> torch.Tensor:
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the gammas that correspond to them;
    i.e., the gammas from the orthogonal polynomial series that
    is orthogonal w.r.t the linear moment functional with those moments.
    """

    # first build the prime determinants:
    prime_dets = torch.zeros(order + 2)
    prime_dets[0] = 0.0
    prime_dets[1] = moments[1]  # i.e. the first moment.

    # now build the standard determinants
    dets = torch.zeros(order + 2)
    dets[0] = dets[1] = 1.0
    betas = torch.zeros(order)

    # build out the determinants of the matrices
    for i in range(order):
        hankel_matrix = moments[: 2 * i + 1].unfold(0, i + 1, 1)
        if i > 1:
            prime_hankel_matrix = torch.hstack(
                (hankel_matrix[:-1, :-2], hankel_matrix[:-1, -1].unsqueeze(1))
            )
            prime_dets[i + 1] = torch.linalg.det(prime_hankel_matrix)
        dets[i + 2] = torch.linalg.det(hankel_matrix)

    # for i in range(1, order - 1):
    # prime_moments = torch.cat(
    # (moments[: 2 * i - 1], moments[2 * i].unsqueeze(0))
    # )
    # breakpoint()
    # prime_hankel_matrix = prime_moments.unfold(0, i + 1, 1)
    # prime_dets[i + 2] = torch.linalg.det(prime_hankel_matrix)
    # breakpoint()
    betas = prime_dets[2:] / dets[2:] - prime_dets[1:-1] / dets[1:-1]
    return betas


def get_gammas_betas_from_moments(
    moments: torch.Tensor, order: int
) -> (torch.Tensor, torch.Tensor):
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the betas and gammas that correspond to them.

    This is known as the "Chebyshev algorithm" (Gautschi 1982).
    """
    if order + 1 == len(moments):
        print("excess moment: truncating")
        moments = moments[1:]
    gammas = get_gammas_from_moments(moments, order)
    betas = get_betas_from_moments(moments, order)
    return (betas, gammas)


def get_betas_from_moments_gautschi(
    moments: torch.Tensor, order: int
) -> torch.Tensor:
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the gammas that correspond to them;
    i.e., the gammas from the orthogonal polynomial series that
    is orthogonal w.r.t the linear moment functional with those moments.

    This method calculates the gammas via the recursion described in
    Walter Gautschi:
        On Generating Orthogonal Polynomials (1982). (section 2.3)

    """
    betas, _ = get_gammas_betas_from_moments_gautschi(moments, order)
    return betas


def get_gammas_from_moments_gautschi(
    moments: torch.Tensor, order: int
) -> torch.Tensor:
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the gammas that correspond to them;
    i.e., the gammas from the orthogonal polynomial series that
    is orthogonal w.r.t the linear moment functional with those moments.

    This method calculates the gammas via the recursion described in
    Walter Gautschi:
        On Generating Orthogonal Polynomials (1982). (section 2.3)

    """
    _, gammas = get_gammas_betas_from_moments_gautschi(moments, order)
    return gammas


def get_gammas_betas_from_moments_gautschi(
    moments: torch.Tensor, order: int
) -> torch.Tensor:
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the gammas that correspond to them;
    i.e., the gammas from the orthogonal polynomial series that
    is orthogonal w.r.t the linear moment functional with those moments.

    This method calculates the gammas via the recursion described in
    Walter Gautschi:
        On Generating Orthogonal Polynomials (1982). (section 2.3)

    """
    # breakpoint()
    # initialisation
    betas = torch.zeros(order)
    gammas = torch.zeros(order)
    sigma = torch.zeros(2 * order, 2 * order)
    # sigma[0, :] = 0
    # sigma[0, :] = moments
    # breakpoint()
    sigma[0, :] = moments
    betas[0] = moments[1] / moments[0]
    gammas[0] = moments[0]

    for k in range(1, order):
        for l in range(k, 2 * order - k - 1):
            sigma[k, l] = (
                sigma[k - 1, l + 1]
                - betas[k - 1] * sigma[k - 1, l]
                - gammas[k - 1] * (sigma[k - 2, l] if k - 2 > -1 else 0)
            )
            betas[k] = (
                sigma[k, k + 1] / sigma[k, k]
                - sigma[k - 1, k] / sigma[k - 1, k - 1]
            )
            gammas[k] = sigma[k, k] / sigma[k - 1, k - 1]
    return betas, gammas


def get_betas_from_modified_moments_gautschi(
    moments: torch.Tensor, order: int, polynomial: OrthogonalPolynomial
) -> torch.Tensor:
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the betas that correspond to them;
    i.e., the betas from the orthogonal polynomial series that
    is orthogonal w.r.t the linear moment functional with those moments.

    This method calculates the betas via the recursion described in
    Walter Gautschi:
        On Generating Orthogonal Polynomials (1982). (section 2.3)

    """
    betas, _ = get_gammas_betas_from_modified_moments_gautschi(
        moments, order, polynomial
    )
    return betas


def get_gammas_from_modified_moments_gautschi(
    moments: torch.Tensor, order: int, polynomial: OrthogonalPolynomial
) -> torch.Tensor:
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the gammas that correspond to them;
    i.e., the gammas from the orthogonal polynomial series that
    is orthogonal w.r.t the linear moment functional with those moments.

    This method calculates the gammas via the recursion described in
    Walter Gautschi:
        On Generating Orthogonal Polynomials (1982). (section 2.3)

    """
    _, gammas = get_gammas_betas_from_modified_moments_gautschi(
        moments, order, polynomial
    )
    return gammas


def get_gammas_betas_from_modified_moments_gautschi(
    moments: torch.Tensor, order: int, polynomial: OrthogonalPolynomial
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the gammas that correspond to them;
    i.e., the gammas from the orthogonal polynomial series that
    is orthogonal w.r.t the linear moment functional with those moments.

    This method calculates the gammas via the recursion described in
    Walter Gautschi:
        On Generating Orthogonal Polynomials (1982). (section 2.3)

    """
    # initialisation
    input_betas = polynomial.get_betas()
    input_gammas = polynomial.get_gammas()
    betas = torch.zeros(order)
    gammas = torch.zeros(order)
    sigma = torch.zeros(2 * order, 2 * order)
    sigma[0, :] = 0
    sigma[0, :] = moments
    betas[0] = moments[1] / moments[0]
    gammas[0] = moments[0]

    # continuation
    for k in range(1, order):
        for l in range(k, 2 * order - k - 1):
            sigma[k, l] = (
                sigma[k - 1, l + 1]
                - (betas[k - 1] - input_betas[l]) * sigma[k - 1, l]
                - gammas[k - 1] * (sigma[k - 2, l] if k - 2 > -1 else 0)
                + input_gammas[l] * sigma[k - 1, l - 1]
            )
            betas[k] = (
                sigma[k, k + 1] / sigma[k, k]
                - sigma[k - 1, k] / sigma[k - 1, k - 1]
            )
            gammas[k] = sigma[k, k] / sigma[k - 1, k - 1]
    return betas, gammas


def get_poly_from_moments(
    moments: torch.Tensor,
    order: int,
) -> SymmetricOrthonormalPolynomial:
    """
    Accepts a list of moment values and produces from it a
    SymmetricOrthogonalPolynomial.

    The standard recurrence for an orthogonal polynomial series has n equations
    and 2n unknowns - this means that it is not feasible to construct, from a
    given sequence of moments, an orthogonal polynomial sequence that is
    orthogonal     w.r.t the given moments. However, if we impose
    symmetricality, we can build a sequence of symmetric orthogonal polynomials
    from a given set of moments.
    """
    gammas = torch.zeros(order)

    # to construct the polynomial from the sequnce of moments, utilise the
    # sequence of equations:
    gammas = get_gammas_from_moments(moments, order)

    return SymmetricOrthonormalPolynomial(order, gammas)


def get_gammas_from_sample(sample: torch.Tensor, order: int) -> torch.Tensor:
    """
    Composes get_gammas_from_moments and get_moments_from_sample to
    produce the gammas from a sample. This just allows for simple
    calls to individual functions to construct the necessary
    component in any given situation.
    """
    return get_gammas_from_moments_gautschi(
        get_moments_from_sample(sample, 2 * order), order
    )


def get_poly_from_sample(
    sample: torch.Tensor, order: int
) -> SymmetricOrthonormalPolynomial:
    """
    Returns a SymmetricOrthogonalPolynomial calculated by:
         - taking the moments from the sample, with odd moments set to 0;
         - constructing from these the gammas that correspond to the
           SymmetricOrthogonalPolynomial recursion
         - generating the SymmetricOrthogonalPolynomial from these gammas.
    Hence we have a composition of the three functions:
          get_moments_from_sample -> get_poly_from_moments
    """
    moments = get_moments_from_sample(sample, order)
    return get_poly_from_moments(moments, order)


if __name__ == "__main__":
    test_modified_moments = True
    test_moments_from_sample = False
    use_sobol = True
    # normal moments:
    order = 15
    if test_moments_from_sample:
        dist = D.Normal(0.0, 1.0)
        # sample = dist.sample([4000 ** 2])
        sample_size = 20000
        if use_sobol:
            sobol = SobolEngine(dimension=1, scramble=True)
            base_sample = sobol.draw(sample_size)
            sample = dist.icdf(base_sample).squeeze()[2:]
        else:
            sample = dist.sample((sample_size,))
        checked_moments = get_moments_from_sample(
            sample, order + 2
        )  # [: order + 2]
        checked_moments_logged = get_moments_from_sample_logged(
            sample, order + 2
        )

        plt.plot(checked_moments, color="red")
        plt.plot(checked_moments_logged, color="blue")
        # plt.plot(checked_moments - checked_moments_logged, color="black")
        plt.plot(normal_moments, color="green")
        plt.show()

    # now I build an example with some known assymetric moments and
    # known betas and gammas:
    order = 6
    catalan = False
    if not catalan:
        true_moments = torch.Tensor(
            [
                1.0,
                1.0,
                2.0,
                4.0,
                9.0,
                21.0,
                51.0,
                127.0,
                323.0,
                835.0,
                2188.0,
                5798.0,
            ]
        )
        true_betas = torch.ones(order)
        true_gammas = torch.ones(order)
    else:
        true_moments = torch.Tensor(
            [
                1.0,
                0.0,
                1.0,
                0.0,
                2.0,
                0.0,
                5.0,
                0.0,
                14.0,
                0.0,
                42.0,
                0.0,
            ]
        )
        true_betas = torch.zeros(order)
        true_gammas = torch.ones(order)
    # order = floor(len(true_moments) / 2)

    gammas = get_gammas_from_moments(true_moments, order)
    betas = get_betas_from_moments(true_moments, order)
    print("calculated betas:", betas)
    print("calculated gammas:", gammas)
    print("true betas:", true_betas)
    print("true gammas:", true_gammas)
    normal_moments = []
    for n in range(1, 2 * order):
        normal_moments.append(gauss_moment(n))
    normal_moments = torch.cat(
        (torch.Tensor([1.0]), torch.Tensor(normal_moments))
    )
    # breakpoint()
    betas, gammas = get_gammas_betas_from_moments_gautschi(
        normal_moments, order
    )
    betas, gammas = get_gammas_betas_from_moments_gautschi(true_moments, order)
    print("\n")
    print(
        colored("Gautschi betas and gammas:", "green"),
        betas,
        gammas,
    )

    # test modified moments
    if test_modified_moments:
        order = 12
        # generate the moments and the sample
        dist = D.Normal(0.0, 1.0)
        sample_size = 20000
        if use_sobol:
            sobol = SobolEngine(dimension=1, scramble=True)
            base_sample = sobol.draw(sample_size)
            sample = dist.icdf(base_sample).squeeze()[2:]
        else:
            sample = dist.sample((sample_size,))

        hermite_polynomial = ProbabilistsHermitePolynomial(2 * order)
        # hermite_polynomial = HermitePolynomial(2 * order)
        orthogonal_moments = get_orthogonal_moments_from_sample(
            sample, 2 * order, hermite_polynomial
        )
        print(
            "calculated orthogonal moments:",
            colored(orthogonal_moments, "blue"),
        )
        plt.plot(torch.log(1 + orthogonal_moments))
        plt.show()

        orthogonal_moments = torch.zeros(2 * order)
        orthogonal_moments[0] = 1
        print(colored(orthogonal_moments, "green"))
        betas, gammas = get_gammas_betas_from_modified_moments_gautschi(
            orthogonal_moments, order, hermite_polynomial
        )

        print(
            colored("Gautschi betas and gammas:", "green"),
            betas,
            gammas,
        )
