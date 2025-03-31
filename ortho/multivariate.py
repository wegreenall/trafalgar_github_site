import torch
import torch.distributions as D
from itertools import product
import matplotlib.pyplot as plt
from termcolor import colored


class MultivariateMonomial:
    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d

    def __call__(self, x: torch.Tensor):
        """
        In the beginning, we will have this output the entire r_n shaped vector
        of monomials. If this does not work we will return to choosing
        the multi-index.

        return shape:
            (x.shape[0], r_n)
        """
        # create the stretched value of the right shape
        stretched_x = x.unsqueeze(2).expand(
            -1, -1, MultivariateStieltjes.r(self.n, self.d)
        )

        # create the multi-index
        multi_index = self._multi_index(self.n, self.d)
        monomial = torch.pow(stretched_x, multi_index).prod(dim=1)
        assert monomial.shape == (x.shape[0], MultivariateStieltjes.r(self.n, self.d))
        return monomial

    def _multi_index(self, n: int, d: int) -> torch.Tensor:
        """
        Returns the multi-index of the monomial of exactly degree n in d
        dimensions. Released in lexicographical order.

        Return shape:
            (d, r_n)
        """
        result = filter(lambda x: sum(x) == n, product(list(range(n + 1)), repeat=d))
        return torch.Tensor(list(result)).t()


class MultivariateOrthogonalPolynomialSequence:
    """
    Returns a sequence of MultivariateOrthogonalPolynomial objects of degree 0 to N
    """

    def __init__(self, N: int, d: int, sample: torch.Tensor):
        self.N = N
        self.d = d
        self.monomials = MultivariateMonomialSequence(N, d)
        self.sample = sample

        self.L_inv = self._get_moment_matrix(sample)

    def __call__(self, x: torch.Tensor):
        return self.monomials(x) @ self.L_inv

    def _get_moment_matrix(self, sample: torch.Tensor):
        """
        Returns the Gram matrix:
            (G_n)_{i,j} = m(i,j)
            G_n \in R^{R_n x R_n}
            m(i,j) = int φ_i(x) φ_j(x) dμ(x)

        so \hat{m}(i,j) = \frac{1}{n} \sum_{k=1}^n φ_i(x_k) φ_j(x_k)

        Return shape:
            (R_n, R_n)
        """
        value = self.monomials(sample)
        moment_matrix_value = value.t() @ value / sample.shape[0]
        L = torch.linalg.cholesky(moment_matrix_value)
        L_inv = torch.inverse(L)
        return L_inv


class MultivariateMonomialSequence:
    """
    Returns a sequence of MultivariateMonomial objects of degree 0 to N
    """

    def __init__(self, N: int, d: int):
        self.N = N
        self.d = d

        self.monomials = [
            MultivariateMonomial(n, d) for n in range(N + 1)
        ]  # an iterator of the monomials
        # print(self.monomials)

    def __call__(self, n: int, x: torch.Tensor):
        result = torch.zeros(x.shape[0], int(MultivariateStieltjes.R(n, self.d)))
        for n, monomial in enumerate(self.monomials[: n + 1]):
            monom_data = monomial(x)
            result[
                :,
                MultivariateStieltjes.R(n - 1, self.d) : MultivariateStieltjes.R(
                    n, self.d
                ),
            ] = monom_data
        return result


def moment_matrix(
    n: int,
    d: int,
    sample: torch.Tensor,
    multivariate_monomial_sequence: MultivariateMonomialSequence,
):
    """
    Returns the Gram matrix:
        (G_n)_{i,j} = m(i,j)
        G_n \in R^{R_n x R_n}
        m(i,j) = int φ_i(x) φ_j(x) dμ(x)

    so \hat{m}(i,j) = \frac{1}{n} \sum_{k=1}^n φ_i(x_k) φ_j(x_k)

    Return shape:
        (R_n, R_n)
    """
    value = multivariate_monomial_sequence(sample)
    moment_matrix_value = value.t() @ value / sample.shape[0]
    # print(moment_matrix_value.shape)
    # plt.imshow(moment_matrix_value.numpy(), cmap="viridis")
    # plt.show()

    L = torch.linalg.cholesky(moment_matrix_value)
    L_inv = torch.inverse(L)
    p = L_inv @ value.t()
    return p


"""
In all cases, the phrase "the paper" refers to:
    A Stieltjes algorihtm for generating multivariate orthogonal polynomials,
    Liu and Narayan, 2023.

To get A_{n+1},i, B_{n+1},i, the following calculation process is used, given
A_{n},i, B_{n},i, and B_{n-1},i:

    1. Calculate the moment matrix S_{n,i} and T_{n,i,j} for the given sample.
    2. calculate A_{n+1}, i = S_{n, i}
    3. Calculate \tilde{p}_{n+1} = x_i p_n - A_{n+1}, i p_n - B'_{n, i} p_{n-1}
    4. Calculate T_{n, i, j} = \int \tilde{p}_{n+1} \tilde{p}_{n+1}^t dμ(x)
    5. Calculate B_{n+1, i}:
        5a. Compute U_{n+1, i}, Σ_{n+1}, from B's SVD.
        5b. Compute V_{n+1, i}:
            5b1. Compute subblocks \hat{V}
            5b2. Compute V_{n+1, i} = V_{n+1, i} - A_{n+1, i} U_{n+1, i}
"""


class MultivariateStieltjes:
    def __init__(self, n: int, d: int, sample: torch.Tensor):
        # parameters
        self.n = n
        self.d = d
        self.sample = sample
        self.monomial_sequence = MultivariateMonomialSequence(n, d)

        # matrices
        self.As = []
        self.Bs = []

        # calculate the moment matrix

    def calculate(self):
        """
        The main function to calculate the A and B matrices.
        """
        self.As = []
        self.Bs = []
        self.Us = []  # these are the transforms to push into canonical form
        self.canonical_As = []
        self.canonical_Bs = []
        self.L_invs = []

        # first, get the zero-th
        self.As.append(torch.zeros(self.r(0, self.d), self.r(0, self.d), self.d))
        self.Bs.append(torch.zeros(self.r(0, self.d), self.r(1, self.d), self.d))
        self.Us.append(torch.eye(self.r(0, self.d)))
        self.canonical_As.append(
            torch.zeros(self.r(0, self.d), self.r(0, self.d), self.d)
        )
        self.canonical_Bs.append(
            torch.zeros(self.r(0, self.d), self.r(1, self.d), self.d)
        )
        self.L_invs.append(torch.eye(self.r(0, self.d)))

        for i in range(0, self.n):
            n = i + 1
            print(f"Calculating A and B for n={n}")
            A_n = self.A(n, self.d)
            # print("A shape:", A_n.shape, "n:", n)
            # print("should be:", (self.r(n - 1, self.d), self.r(n - 1, self.d), self.d))
            assert A_n.shape == (
                self.r(n - 1, self.d),
                self.r(n - 1, self.d),
                self.d,
            ), f"A_n shape is incorrect, should be {(self.r(n - 1, self.d), self.r(n - 1, self.d), self.d)}"

            # now that we have A, convert to canonical form before appending

            self.As.append(A_n)
            canonical_An = torch.einsum(
                "ab, bcD, cd -> adD", self.Us[n - 1], A_n, self.Us[n - 1]
            )
            self.canonical_As.append(canonical_An)

            B_n = self.B(n, self.d)
            # print("B shape:", B_n.shape, "n:", n)
            # print("should be:", (self.r(n - 1, self.d), self.r(n, self.d), self.d))
            assert B_n.shape == (
                self.r(n - 1, self.d),
                self.r(n, self.d),
                self.d,
            ), f"B_n shape is incorrect, should be {(self.r(n - 1, self.d), self.r(n, self.d), self.d)}"
            self.Bs.append(B_n)

            # now that we have B, construct the canonical form matrices
            BB = torch.einsum("abd, ced -> be", B_n, B_n)
            assert BB.shape == (self.r(n, self.d), self.r(n, self.d))
            eigens, V = torch.linalg.eig(BB)
            self.Us.append(V.t().real)
            self.L_invs.append(torch.diag(eigens.real))

            canonical_Bn = torch.einsum(
                "ab, bcD, cd -> adD", self.Us[n - 1], B_n, self.Us[n]
            )
            self.canonical_Bs.append(canonical_Bn)

    def ops(self, n: int, inputs: torch.Tensor):
        """
        Returns the orthogonal polynomial sequence at the given input.
        """
        # assert inputs.shape[0] == self.d
        assert len(self.canonical_As) >= n, f"The matrix A_n does not exist for n={n}"
        assert inputs.shape[1] == self.d

        if n == -1:
            return torch.zeros(inputs.shape[0], self.r(0, self.d))

        elif n == 0:
            return torch.ones(inputs.shape[0], self.r(0, self.d))

        elif n == 1:  # calculating p_1
            A_n = self.canonical_As[1]  # A_{1, i} \forall i
            B_n = self.canonical_Bs[1]  # B_{1, i} \forall i
            p_0 = self.ops(0, inputs)

            # einsum index key:
            # R = r_{n}
            # r = r_{n-1}
            # d = d (dimensions)
            # n = n (sample_size)
            term_1 = torch.einsum("nd, rRd, nr -> nR", inputs, B_n, p_0)
            term_2 = torch.einsum("rRd, rrd, nr -> nR", B_n, A_n, p_0)
            result = term_1 - term_2

        elif n > 1:
            A_n = self.canonical_As[n]  # A_{1, i} \forall i
            B_n = self.canonical_Bs[n]  # B_{1, i} \forall i
            B_n_1 = self.canonical_Bs[n - 1]  # B_{1, i} \forall i
            p_n_1 = self.ops(n - 1, inputs)
            p_n_2 = self.ops(n - 2, inputs)

            # einsum index key:
            # R = r_{n}
            # r = r_{n-1}
            # p = r_{n-2}
            # d = d (dimensions)
            # n = n (sample_size)
            term_1 = torch.einsum("nd, rRd, nr -> nR", inputs, B_n, p_n_1)
            term_2 = torch.einsum("rRd, rrd, nr -> nR", B_n, A_n, p_n_1)
            term_3 = torch.einsum("rRd, prd, np -> nR", B_n, B_n_1, p_n_2)

            result = term_1 - term_2 - term_3

        # get the inverse eigenvalue matrix:
        # B_n_transpose = B_n.permute(1, 0, 2)
        # L = torch.einsum("abd, bcd -> ac", B_n_transpose, B_n)
        L_inv = torch.inverse(self.L_invs[n])
        assert L_inv.shape == (
            self.r(n, self.d),
            self.r(n, self.d),
        ), "L_inv shape is incorrect"

        assert result.shape == (
            inputs.shape[0],
            self.r(n, self.d),
        ), f"ops shape is incorrect, should be: {(inputs.shape[0], self.r(n, self.d))}"

        return result @ L_inv

    def ops_tilde(self, n: int, inputs: torch.Tensor):
        """
        In order to calculate the A, B matrices, we need to use p_tilde, which
        is constructed as:
                p_tilde = x_i p_n - A_{n+1} p_n - B'_{n, i} p_{n-1}
        """
        A_n_plus_1 = self.canonical_As[n]
        B_n = self.canonical_Bs[n - 1]
        term_1 = torch.einsum("nd, nr -> nrd", inputs, self.ops(n - 1, inputs))
        term_2 = -torch.einsum("rrd, nr -> nrd", A_n_plus_1, self.ops(n - 1, inputs))
        term_3 = -torch.einsum("rRd, nr -> nRd", B_n, self.ops(n - 2, inputs))
        return term_1 + term_2 + term_3

    @staticmethod
    def r(n: int, d: int) -> int:
        """
        This is the number of monomials of degree n in d dimensions.
        """
        top = torch.Tensor([n + d - 1])
        bottom = torch.Tensor([n])
        return int(
            torch.exp(
                (
                    torch.lgamma(top + 1)
                    - torch.lgamma(bottom + 1)
                    - torch.lgamma(top - bottom + 1)
                )
            )
        )

    @staticmethod
    def r_delta(n: int, d: int) -> int:
        """
        This is the number of monomials of degree n in d dimensions minus the
        number of monomials of degree n-1 in d dimensions.
        """
        return MultivariateStieltjes.r(n, d) - MultivariateStieltjes.r(n - 1, d)

    @staticmethod
    def R(n: int, d: int) -> int:
        """
        This is the dimensionality of the space spanned by multinomials up to
        degree n in d dimensions.
        """
        top = torch.Tensor([n + d])
        bottom = torch.Tensor([n])
        return int(
            torch.exp(
                (
                    torch.lgamma(top + 1)
                    - torch.lgamma(bottom + 1)
                    - torch.lgamma(top - bottom + 1)
                )
            )
        )

    def S(self, i: int, d: int):
        """
        Returns the moment matrix S_n,i for the given sample, where:
                     S_n,i = \int x_i p_n p_n^t dμ(x)
        """
        polynomial_evaluation = self.ops(i, self.sample)

        if i == 1:
            breakpoint()

        result = (
            torch.einsum(
                "nd, ni, nj -> ijd",
                self.sample,
                polynomial_evaluation,
                polynomial_evaluation,
            )
            / self.sample.shape[0]
        )

        assert result.shape == (
            self.r(i, d),
            self.r(i, d),
            d,
        ), f"S_{i} shape is incorrect, should be {(self.r(i, d), self.r(i, d), d)}"
        return result

    def T(self, n: int, d: int):
        """
        Returns the moment matrix T_n,i,j for the given sample, where:
                     T_n,i,j = \int x_i \tilde{p}_{n+1} \tilde{p}_{n+1}^t dμ(x)
        """
        polynomial_evaluation = self.ops_tilde(n, self.sample)

        result = (
            torch.einsum(
                "nrd, nRD -> rRdD", polynomial_evaluation, polynomial_evaluation
            )
            / self.sample.shape[0]
        )

        assert result.shape == (
            self.r(n - 1, d),
            self.r(n - 1, d),
            d,
            d,
        ), f"T_{{n, i, j}} shape is incorrect, should be {(self.r(n, d), self.r(n, d), d)}"
        return result

    def A(self, n: int, d: int):
        """
        Returns the matrix A_{n, i} in the recurrence for a given polynomial.

        Necessary for this calculation is P_{n-1}.
        """
        # polynomial_evaluation = self.ops(n-1, self.sample)
        result = self.S(n - 1, d)
        return result

    def B(self, n: int, d: int):
        """
        Calculates the matrix B_n,i in the recurrence.
        """

        if n == 0:  # "fall back" to the constant polynomial
            result = torch.ones(self.r(n, d), self.r(n + 1, d), d)
        elif n == 1:  # "fall back" to the moment matrices
            i = 0
            """
            See section 5.2.5 of the paper.

            ... Therefore, when d > 2 and n = 0, we use (18) to compute the B_{1,i}
                matrices.

            (18) A_{n+1,i} = \tilde{L}_n^{-1} G_{n,i} \tilde{L}_n^{-T}
                 B_{n+1,i} = \tilde{L}_n^{-1} \tilde{G}_{n+1,i} \tilde{L}_{n+1}^{-T}

            """
            Gn = self.G(i, d)
            Gn_plus_1 = self.G(i + 1, d)

            Ln = torch.linalg.cholesky(Gn)
            Ln_inv = torch.inverse(Ln)
            Ln_tilde = Ln_inv[-self.r(i, d) :, :]
            assert Ln_tilde.shape == (self.r(i, d), self.R(i, d))
            # print("just made Ln_tilde")

            Ln_plus_1 = torch.linalg.cholesky(Gn_plus_1)
            Ln_plus_1_inv = torch.inverse(Ln_plus_1)
            Ln_plus_1_tilde = Ln_plus_1_inv[-self.r(n, d) :, :]
            assert Ln_plus_1_tilde.shape == (self.r(n, d), self.R(n, d))

            # Gni = int x_i p_0 p_0^t dμ(x)
            Gni_plus_1 = (
                torch.einsum(
                    "nd, ni, nj -> ijd",
                    self.sample,
                    self.monomial_sequence(n, self.sample),
                    self.monomial_sequence(n, self.sample),
                )
                / self.sample.shape[0]
            )
            assert Gni_plus_1.shape == (self.R(n, d), self.R(n, d), self.d)

            Gni_plus_one_tilde = Gni_plus_1[: self.R(i, d), :]
            assert Gni_plus_one_tilde.shape == (self.R(i, d), self.R(n, d), self.d)

            # key:
            # r = r_n
            # p = r_n+1
            # R = R_n
            # P = R_{n+1}
            # d = dimension
            B_n_plus_1 = torch.einsum(
                "rR, RPd, pP -> rpd", Ln_tilde, Gni_plus_one_tilde, Ln_plus_1_tilde
            )
            result = B_n_plus_1

        elif n > 1:
            """
            See section 5.2.6 of the paper.
            """
            T = self.T(n, d)
            print("T shape:", T.shape)
            Ss = torch.zeros(self.r(n - 1, d), self.r(n - 1, d), d)
            Us = torch.zeros(self.r(n - 1, d), self.r(n - 1, d), d)
            V_hat = torch.zeros(self.r(n - 1, d), self.r(n - 1, d), d)
            V_tilde = torch.zeros(self.r_delta(n, d), self.r(n - 1, d), d)
            for i in range(d):
                for j in range(d):
                    if i == j:
                        T_ii = T[:, :, i, i]
                        U, S, V = torch.svd(T_ii)
                        # take the square root of the Σ matrix
                        S = torch.sqrt(torch.diag(S))
                        Ss[:, :, i] = S
                        Us[:, :, i] = U

            # calculate the V_hat matrix
            Ss = Ss.permute(2, 0, 1)
            try:
                Ss_inv = torch.linalg.inv(Ss)
            except torch._C._LinAlgError:
                print(colored("Linalg error on Ss", "red"))
                breakpoint()
            Ss = Ss.permute(1, 2, 0)
            Ss_inv = Ss_inv.permute(1, 2, 0)

            """
            From the paper:
                Using mixed moments  T_{n,i,j}  we can compute the square
                matrices V_hat_{n+1} whicha re subblocks of V_{n+1},

                (Recall from (22) that V_hat_{n+1},i = is already known for i = 1)

            """
            V_hat[:, :, 0] = torch.eye(self.r(n - 1, d))
            V_hat[:, :, 1:] = torch.einsum(
                "ab, bc, cdD, deD, efD -> afD",
                Ss_inv[:, :, 0],
                Us[:, :, 0],
                T[:, :, 0, 1:],
                Us[:, :, 1:],
                Ss_inv[:, :, 1:],
            )

            # Calculate the V_tilde matrices - let's see
            if d == 2:
                """
                See section 5.2.4 of the paper.

                When d = 2, we need only compute V_tilde_{n+1,i} for i = 2.
                Since Δr_{n} = 1 for every n when d = 2.

                We fix:
                    y := V^T_tilde_{n+1,2} \in R^{r_{n}}

                Then,
                    yy' = I_{r_{n}} - V^T_hat_{n+1,2} V_hat_{n+1,2}

                                z = chol(yy')^T

                """
                yy = torch.eye(self.r(n - 1, d)) - V_hat[:, :, 1].t() @ V_hat[:, :, 1]
                yy += 1e-4 * torch.eye(
                    self.r(n - 1, d)
                )  # perturbation to get around linalg error
                if (yy != yy).any():
                    raise ValueError("NaN in yy")
                try:
                    z = torch.linalg.cholesky(yy)[:, 0].t()
                except torch._C._LinAlgError:
                    print(colored("Linalg error on yy", "red"))
                    print("yy:", yy)
                    breakpoint()

                # z = torch.linalg.cholesky(yy)[:, 0].t()
                V_tilde[:, :, 1] = z
            elif d > 2:
                pass
            Vs = torch.cat((V_hat, V_tilde))
            B_n_plus_1 = torch.einsum("abD, bcD, dcD -> adD", Us, Ss, Vs)
            result = B_n_plus_1

        """
        B_{n+1}, i = \in R^{r_{n-1} x r_{n}}
        B_n = \in R^{dr_{n-1} x r_{n-1}}
        """
        assert result.shape == torch.Size(
            (self.r(n - 1, d), self.r(n, d), d)
        ), f"Shape of B_n is {result.shape}, and it should be {(self.r(n-1, d), self.r(n, d), d)}"
        return result

    def U(self, n: int, i: int, d: int):
        """
        The U matrix is the first part of the SVD of the B matrix.
        """
        pass

    def G(self, n: int, d: int):
        """
        The G matrix is the moment matrix used to construct the B matrix.
        """
        Gn = (
            torch.einsum(
                "ni, nj -> ij",
                self.monomial_sequence(n, self.sample),
                self.monomial_sequence(n, self.sample),
            )
            / self.sample.shape[0]
        )
        # breakpoint()
        return Gn

    def V_hat(self, n: int, i: int, d: int):
        """
        The V_hat matrix is the first r_n columns of the SVD of the B matrix at n+1.
        """
        pass

    def V_tilde(self, n: int, i: int, d: int):
        pass

    def Sigma(self, n: int, d: int):
        pass


if __name__ == "__main__":
    test_multivariate_monomial = False
    test_stieltjes = True
    if test_multivariate_monomial:
        multivariate_monomial = MultivariateMonomial(5, 2)

        # plot multivariate polynomial
        fineness = 200

        # sample = D.Normal(0, 6).sample((100000, 2))
        sample = D.MultivariateNormal(
            torch.zeros(2), torch.Tensor([[5.0, 0.0], [0.0, 2.0]])
        ).sample((100000,))

        # monomial_sequence = MultivariateMonomialSequence(5, 2)
        # moment_matrix(5, 2, sample, monomial_sequence)

        multivariate_orthogonal_polynomial_sequence = (
            MultivariateOrthogonalPolynomialSequence(5, 2, sample)
        )

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        x = torch.linspace(-5, 5, fineness)
        y = torch.linspace(-5, 5, fineness)
        X, Y = torch.meshgrid(x, y)
        Z = multivariate_orthogonal_polynomial_sequence(
            torch.stack([X.flatten(), Y.flatten()], dim=1)
        ).reshape(fineness, fineness, -1)
        print(Z.shape)
        ax.plot_surface(
            X.numpy(),
            Y.numpy(),
            Z[:, :, 12].numpy(),
            rstride=1,
            cstride=1,
            cmap="viridis",
        )
        plt.show()

    if test_stieltjes:
        sample_size = 1000000
        max_degree = 5
        dimension = 2
        dist = D.MultivariateNormal(torch.zeros(dimension), torch.eye(dimension))
        rho = 0.7
        correlated_covariance = torch.Tensor([[1.0, rho], [rho, 1.0]])
        correlated_dist = D.MultivariateNormal(
            torch.zeros(dimension), torch.eye(dimension)
        )
        msm = MultivariateStieltjes(max_degree, dimension, dist.sample((sample_size,)))
        msm.calculate()
