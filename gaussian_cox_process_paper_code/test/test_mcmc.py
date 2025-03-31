import torch
import unittest
from gcp_rssb.mcmc.lgcp_mcmc_poisson_samples import (
    IntensityLagrangePolynomials,
    IntensityKernelInterpolation,
    IntensityPiecewiseConstant,
)
from mercergp.kernels import SmoothExponentialKernel
import matplotlib.pyplot as plt


def approximand(x: torch.Tensor) -> torch.Tensor:
    return x**2


class TestIntensityPointwiseConstant(unittest.TestCase):
    def setUp(self):
        self.x_axis = torch.linspace(-1, 1, 20).unsqueeze(1)
        self.sample = 3 * self.x_axis + 2

    def test_intensity_pointwise_constant(self):
        intensity = IntensityPiecewiseConstant(self.sample, self.x_axis, 1)
        fineness = 100
        other_x_axis = torch.linspace(-0.95, 0.95, 100).unsqueeze(1)
        test_output = intensity(other_x_axis[int(fineness / 2)])
        test_output_epsilon = intensity(other_x_axis[int(fineness / 2)] + 1e-3)
        self.assertTrue(
            torch.allclose(test_output, test_output_epsilon, atol=1e-03)
        )


class TestIntensityLagrange(unittest.TestCase):
    def setUp(self):
        self.input_count = 10
        self.inputs = torch.linspace(-1, 1, self.input_count).unsqueeze(1)
        self.outputs = approximand(self.inputs)
        self.dimension = 1

        self.intensity = IntensityLagrangePolynomials(
            self.outputs, self.inputs, self.dimension
        )

    def test_denominator_tensor(self):
        denominator = self.intensity.bare_denominator_tensor
        self.assertEqual(
            denominator.shape,
            (self.input_count, self.input_count, self.dimension),
        )

    def test_intensity_lagrange_polynomials_fixedpoints(self):
        test_output = self.intensity(self.inputs)
        self.assertEqual(test_output.shape, (self.input_count, 1))
        self.assertTrue(torch.allclose(test_output, self.outputs, atol=1e-03))

    @unittest.skip("Not implemented")
    def test_intensity_lagrange_polynomials(self):
        other_inputs = torch.linspace(
            -0.7, 0.7, 2 * self.input_count
        ).unsqueeze(1)
        test_output = self.intensity(other_inputs)
        self.assertEqual(test_output.shape, (self.input_count, 1))
        self.assertTrue(torch.allclose(test_output, self.outputs))


class TestIntensityInterpolation(unittest.TestCase):
    def setUp(self):
        self.input_count = 10
        self.inputs = torch.linspace(-1, 1, self.input_count).unsqueeze(1)
        self.outputs = approximand(self.inputs)
        self.dimension = 1

        # kernel args
        kernel_args = {
            "ard_parameter": torch.Tensor([[5.0]]),
            "variance_parameter": torch.Tensor([[0.2]]),
            "noise_parameter": torch.Tensor([[1e-5]]),
        }
        kernel = SmoothExponentialKernel(kernel_args)

        self.intensity = IntensityKernelInterpolation(
            self.outputs, self.inputs, self.dimension, kernel
        )

    def test_intensity_fixedpoints(self):
        test_output = self.intensity(self.inputs)
        self.assertEqual(test_output.shape, (self.input_count, 1))
        # breakpoint()
        self.assertTrue(torch.allclose(test_output, self.outputs, atol=1e-03))

    @unittest.skip("Not implemented")
    def test_intensity_lagrange_polynomials(self):
        other_inputs = torch.linspace(
            -0.7, 0.7, 2 * self.input_count
        ).unsqueeze(1)
        test_output = self.intensity(other_inputs)
        self.assertEqual(test_output.shape, (self.input_count, 1))
        self.assertTrue(torch.allclose(test_output, self.outputs, atol=1e-03))


if __name__ == "__main__":
    unittest.main()
