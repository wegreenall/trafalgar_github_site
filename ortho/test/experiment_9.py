import torch
import os
import torch.distributions as D
import matplotlib.pyplot as plt
import csv
import re

from mercergp.builders import (
    train_smooth_exponential_mercer_params,
    build_smooth_exponential_mercer_gp,
    train_mercer_params,
    build_mercer_gp,
)
from mercergp.MGP import MercerGP

from mercergp.eigenvalue_gen import (
    SmoothExponentialFasshauer,
    # FavardEigenvalues,
)

# from mercergp.kernels import SmoothExponentialKernel, MercerKernel
from ortho.builders import get_orthonormal_basis_from_sample
from termcolor import colored
from experiment_functions import test_function, weight_function, get_training_inputs

# from ortho.basis_functions import OrthonormalBasis


def get_latest_experiment_number():
    files = os.listdir("./experiment_9_data/")
    numbers = []
    number_regex = re.compile(r"(?P<number>[0-9]*).pt")
    for file in files:
        # get the number in the filename
        number_matches = number_regex.search(file)
        if number_matches is not None:
            number = number_matches.group("number")
        else:
            return 0
        try:
            numbers.append(int(number))
            # print(number)
        except ValueError:
            pass
    return max(numbers)


"""
Given a sample from a Gaussian process,  we want to see how the Favard kernel
idea obtains the correct length scale. If we are able to calculate the right 
length-scale, and the standard Mercer fails, this will be good evidence.
"""


def get_funcs(
    sample_shape,
    gaussian_input_sample,
    non_gaussian_input_sample,
):
    """
    Given a kernel and a sample shape, outputs sample from a GP
    """
    # z = D.Normal(0.0, 1.0).sample(sample_shape)
    noise_sample = torch.sqrt(kernel_args["noise_parameter"]) * D.Normal(
        0.0, 1.0
    ).sample(sample_shape)

    gaussian_input_func = test_function(gaussian_input_sample) + noise_sample
    non_gaussian_input_func = test_function(non_gaussian_input_sample) + noise_sample
    return gaussian_input_func, non_gaussian_input_func


def train_gaussian_gp(gaussian_input_sample, gaussian_output_sample, order):
    # For each of the function samples, estimate the length_scale parameter
    gaussian_initial_ard_parameter = torch.Tensor([0.1]).clone()
    gaussian_initial_precision_parameter = torch.Tensor([[1.0]]).clone()
    gaussian_initial_noise_parameter = torch.Tensor([1.0]).clone()
    gaussian_initial_ard_parameter.requires_grad = True
    gaussian_initial_noise_parameter.requires_grad = True
    gaussian_parameters = {
        "noise_parameter": gaussian_initial_noise_parameter,
        "precision_parameter": torch.Tensor([1 / torch.std(gaussian_input_sample)]),
        "ard_parameter": gaussian_initial_ard_parameter,
    }
    gaussian_optimiser = torch.optim.Adam(
        [gaussian_initial_ard_parameter, gaussian_initial_noise_parameter],
        lr=0.001,
    )

    # train the parameters!
    trained_gaussian_input_params = train_smooth_exponential_mercer_params(
        gaussian_parameters,
        order,
        gaussian_input_sample,
        gaussian_output_sample,
        gaussian_optimiser,
    )
    gaussian_mercer_gp = build_smooth_exponential_mercer_gp(
        trained_gaussian_input_params,
        order,
    )
    gaussian_mercer_gp.add_data(gaussian_input_sample, gaussian_output_sample)
    return gaussian_mercer_gp


def train_non_gaussian_gp(
    non_gaussian_input_sample, non_gaussian_output_sample, order
) -> MercerGP:
    non_gaussian_initial_ard_parameter = torch.Tensor([0.1]).clone()
    non_gaussian_initial_precision_parameter = torch.Tensor([1.0]).clone()
    non_gaussian_initial_noise_parameter = torch.Tensor([1.0]).clone()
    non_gaussian_initial_ard_parameter.requires_grad = True
    non_gaussian_initial_noise_parameter.requires_grad = True
    non_gaussian_parameters = {
        "noise_parameter": non_gaussian_initial_noise_parameter,
        # "precision_parameter": non_gaussian_initial_precision_parameter,
        "precision_parameter": torch.Tensor([1 / torch.std(non_gaussian_input_sample)]),
        "ard_parameter": non_gaussian_initial_ard_parameter,
    }
    non_gaussian_optimiser = torch.optim.Adam(
        [
            non_gaussian_initial_ard_parameter,
            non_gaussian_initial_noise_parameter,
        ],
        lr=0.001,
    )

    trained_non_gaussian_input_params = train_smooth_exponential_mercer_params(
        non_gaussian_parameters,
        order,
        non_gaussian_input_sample,
        non_gaussian_output_sample,
        non_gaussian_optimiser,
    )
    non_gaussian_mercer_gp = build_smooth_exponential_mercer_gp(
        trained_non_gaussian_input_params,
        order,
    )
    non_gaussian_mercer_gp.add_data(
        non_gaussian_input_sample, non_gaussian_output_sample
    )
    return non_gaussian_mercer_gp


def train_favard_gp(
    non_gaussian_input_sample, non_gaussian_output_sample, order
) -> MercerGP:
    favard_initial_ard_parameter = torch.Tensor([0.1]).clone()
    favard_initial_precision_parameter = torch.Tensor([1.0]).clone()
    favard_initial_noise_parameter = torch.Tensor([1.0]).clone()
    favard_initial_ard_parameter.requires_grad = True
    favard_initial_noise_parameter.requires_grad = True
    favard_parameters = {
        "noise_parameter": favard_initial_noise_parameter,
        # "precision_parameter": favard_initial_precision_parameter,
        "precision_parameter": torch.Tensor([1 / torch.std(non_gaussian_input_sample)]),
        "ard_parameter": favard_initial_ard_parameter,
        "degree": 6,  # the eigenvalue polynomial exponent for the decay
    }

    """
    trained_non_gaussian_input_params is the result of training with the 
,    non-Gaussian input measure on the Gaussian eigenfunctions. 
    trained_gaussian_input_params is the result of training with the Gaussian 
    input measure sample on the Gaussian eigenfunctions. Now we produce the 
    parameters corresponding to training a Favard kernel version.
    This will use the non-Gaussian inputs.
    """
    # eigenvalue_generator = FavardEigenvalues(order)
    favard_optimiser = torch.optim.Adam(
        [favard_initial_ard_parameter, favard_initial_noise_parameter],
        lr=0.001,
    )

    print("Training favard gp!")

    smooth_exponential_eigenvalues = SmoothExponentialFasshauer(order)
    basis = get_orthonormal_basis_from_sample(
        non_gaussian_input_sample, weight_function, order
    )
    trained_favard_params = train_mercer_params(
        favard_parameters,
        non_gaussian_input_sample,
        non_gaussian_output_sample,
        basis,
        favard_optimiser,
        smooth_exponential_eigenvalues,
    )
    favard_mercer_gp = build_mercer_gp(
        trained_favard_params, order, basis, smooth_exponential_eigenvalues
    )
    favard_mercer_gp.add_data(non_gaussian_input_sample, non_gaussian_output_sample)
    return favard_mercer_gp


train_gaussian = True
train_non_gaussian = True
train_favard = True
plot_initial_samples = False

# first, get a sample function from a GP with a given length scale
sample_size = 800
sample_shape = torch.Size([sample_size])
kernel_args = {
    "variance_parameter": torch.Tensor([1.0]),
    "ard_parameter": torch.Tensor([[1.0]]),
    "noise_parameter": torch.Tensor([0.2]),
}
# se_kernel = SmoothExponentialKernel(kernel_args)

experiment_count = 1000

# prepare the general test points for all experiments
test_points_size = 60
test_points_shape = torch.Size([test_points_size])
# test_points = D.Normal(0.0, 4.0).sample(test_points_shape)
# test_points_outputs = test_function(test_points)

start_number = get_latest_experiment_number()
for i in range(start_number + 1, experiment_count):
    # test_points for predictive densities
    test_points = D.Normal(0.0, 4.0).sample(test_points_shape)
    test_points_outputs = test_function(test_points)
    print(colored("Current experiment number:{}".format(i), "red"))
    (
        gaussian_input_sample,
        non_gaussian_input_sample,
    ) = get_training_inputs(sample_shape)

    gaussian_input_func, non_gaussian_input_func = get_funcs(
        sample_shape, gaussian_input_sample, non_gaussian_input_sample
    )

    # GP zs
    """
    Now that we have a sample from each of the Gaussian processes,
    we can use these as data from which we will extract the ARD parameter.
    """
    if plot_initial_samples:
        plt.scatter(gaussian_input_sample, gaussian_input_func)
        plt.scatter(non_gaussian_input_sample, non_gaussian_input_func)
        plt.show()

    # parameters for the gaussian process _model_
    order = 15
    if train_gaussian:
        # For each of the function samples, estimate the length_scale parameter
        trained_gaussian_gp = train_gaussian_gp(
            gaussian_input_sample, gaussian_input_func, order
        )

        gaussian_predictive_density = (
            trained_gaussian_gp.get_marginal_predictive_density(test_points)
        )
        gaussian_predictive_densities = torch.exp(
            gaussian_predictive_density.log_prob(test_points_outputs)
        )
        gaussian_saves = (
            "./experiment_9_data/exp_9_gaussian_parameters_" + str(i) + ".pt"
        )
        torch.save(gaussian_predictive_densities, gaussian_saves)

    if train_non_gaussian:
        trained_non_gaussian_gp = train_non_gaussian_gp(
            non_gaussian_input_sample, non_gaussian_input_func, order
        )

        non_gaussian_predictive_density = (
            trained_non_gaussian_gp.get_marginal_predictive_density(test_points)
        )
        non_gaussian_predictive_densities = torch.exp(
            non_gaussian_predictive_density.log_prob(test_points_outputs)
        )
        non_gaussian_saves = (
            "./experiment_9_data/exp_9_non_gaussian_parameters_" + str(i) + ".pt"
        )
        torch.save(non_gaussian_predictive_densities, non_gaussian_saves)

    if train_favard:
        trained_favard_gp = train_favard_gp(
            non_gaussian_input_sample, non_gaussian_input_func, order
        )
        favard_predictive_density = trained_favard_gp.get_marginal_predictive_density(
            test_points
        )
        favard_predictive_densities = torch.exp(
            favard_predictive_density.log_prob(test_points_outputs)
        )
        favard_saves = "./experiment_9_data/exp_9_favard_parameters_" + str(i) + ".pt"
        torch.save(favard_predictive_densities, favard_saves)

    # gaussian_saves = open("exp_8_gaussian_parameters.pt", "a")
    # non_gaussian_saves = open("exp_8_non_gaussian_parameters.pt", "a")
    # favard_saves = open("exp_8_non_gaussian_parameters.pt", "a")
