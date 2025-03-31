"""
In this script we attempt to construct an autoencoder on MNIST, so that we
can see if our classification method works in the constructed latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gcp_rssb.methods.gcp_ose import GCPOSEHyperparameters
from gcp_rssb.methods.gcp_ose_bayesian import PriorParameters
from gcp_rssb.methods.gcp_ose_classifier import GCPClassifier

from ortho.basis_functions import Basis, standard_chebyshev_basis


class AE(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, dimension),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    get_predictions = True
    plot = False
    train_encoder_decoder = False
    plot_posterior_mean = False

    # if train_encoder_decoder:
    dimension = 2  # the dimension of the latent space
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # if torch.cuda.is_available():
    # torch.set_default_device("cuda:0")
    print("Using device:", device)
    transform = transforms.Compose([transforms.ToTensor()])
    print("got transforms")
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    relevant_indices = []
    labels = [3, 8]

    # relevant label choices
    label_counts = [0] * len(labels)
    for i, target in enumerate(dataset.targets):
        if int(target) not in labels:
            continue
        index = labels.index(int(target))
        label_counts[index] += 1
        relevant_indices.append(i)

    relevant_data = dataset.data[relevant_indices]
    relevant_targets = dataset.targets[relevant_indices]
    dataset.data = relevant_data
    dataset.targets = relevant_targets
    dataset.classes = [str(i) for i in labels]

    # print("got dataset")
    batch_size = 64
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device=device),
    )
    debatched_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        generator=torch.Generator(device=device),
    )

    model = AE(dimension).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    if train_encoder_decoder:
        epochs = 400
        outputs = []
        losses = []
        if get_predictions:
            for epoch in range(epochs):
                for i, (image, class_label) in enumerate(loader):
                    print(f"Epoch {epoch}, batch {i}")
                    image = image.reshape(-1, 784).to(device)
                    recon = model(image)
                    loss = criterion(recon, image)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

            # build out the encoded value sets
            encoded_value_sets = []
            for label, label_count in zip(labels, label_counts):
                encoded_values = torch.zeros((label_count, dimension))
                encoded_value_sets.append(encoded_values)

            k_zeros = 0
            k_ones = 0
            k_counts = [0] * len(labels)
            for i, (image, class_label) in enumerate(debatched_loader):
                image = image.reshape(-1, 784).to(device)
                encoding = model.encoder(image)  # [1, 2]
                label_index = labels.index(class_label)
                encoded_value_sets[label_index][
                    k_counts[label_index], :
                ] = encoding.detach()
                k_counts[label_index] += 1

            # save the encoded value sets
            torch.save(encoded_value_sets, "encoded_value_sets.pt")

            # save the encoder and decoder models
            torch.save(model.encoder.state_dict(), "encoder.pt")
            torch.save(model.decoder.state_dict(), "decoder.pt")
        else:
            encoded_value_sets = torch.load("encoded_value_sets.pt")
    else:
        # load the model
        dimension = 2
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        # torch.set_default_device("cuda:0")
        # print("Using device:", device)
        model = AE(dimension)
        model.encoder.load_state_dict(torch.load("encoder.pt"))
        model.decoder.load_state_dict(torch.load("decoder.pt"))
        model.eval()
        encoded_value_sets = torch.load("encoded_value_sets.pt")

    # now present the results
    if plot:
        for i, encoded_values in enumerate(encoded_value_sets):
            plt.scatter(
                encoded_values[:, 0].cpu().detach().numpy(),
                encoded_values[:, 1].cpu().detach().numpy(),
                label=labels[i],
                linewidths=0.1,
                s=0.5,
            )
        plt.show()

    # classifier
    order = 20
    basis_params = [
        {
            "upper_bound": torch.Tensor([18.0]),
            "lower_bound": torch.Tensor([-18.0]),
        }
    ] * dimension
    basis_functions = [standard_chebyshev_basis] * dimension
    basis = Basis(basis_functions, dimension, order, basis_params)
    parameters = PriorParameters(mean=0.0, alpha=1.5, beta=1.0, nu=0.1)
    hyperparameters = GCPOSEHyperparameters(basis, dimension)
    classifier = GCPClassifier(
        len(labels),
        parameters,
        hyperparameters,
        stabilising_epsilon=0.0,
    )

    print("initialised the classifier")
    encoded_value_sets_cpu = [
        value_set.detach().cpu() for value_set in encoded_value_sets
    ]
    data_count = 1000
    for c, encoded_values in enumerate(encoded_value_sets):
        encoded_values = encoded_values.to(device)
        breakpoint()
        classifier.add_data(encoded_values[:data_count, :], c)
        # classifier.add_data(encoded_value_sets[0][:1000, :], 0)
        # classifier.add_data(encoded_value_sets[1][:1000, :], 1)

    # plot the OSE
    fineness = 50
    x_axis = torch.linspace(-16.0, 16.0, fineness)
    y_axis = torch.linspace(-16.0, 16.0, fineness)
    if plot_posterior_mean:
        class_0_posterior_mean = classifier.cox_processes[
            0
        ]._get_posterior_mean()
        class_1_posterior_mean = classifier.cox_processes[
            1
        ]._get_posterior_mean()

        # contour plot the 2d OSE
        x_mesh, y_mesh = torch.meshgrid(x_axis, y_axis)
        test_points = torch.vstack((x_mesh.ravel(), y_mesh.ravel())).t()
        test_points = test_points.to(device)
        test_points_cpu = test_points.cpu()
        plt.contourf(
            x_mesh.cpu().numpy(),
            y_mesh.cpu().numpy(),
            class_0_posterior_mean(test_points)
            .reshape(x_mesh.shape)
            .cpu()
            .numpy(),
        )
        for i, encoded_values in enumerate(encoded_value_sets):
            plt.scatter(
                encoded_values[:data_count, 0].cpu().detach().numpy(),
                encoded_values[:data_count, 1].cpu().detach().numpy(),
                label=labels[i],
                linewidths=0.1,
                s=0.7,
            )
        plt.colorbar()
        plt.show()

    print("memory allocated before dels")
    if device == "cuda":
        print(torch.cuda.memory_summary(device=device))

    # now delete the model from the GPU and get rid other stuff
    del model
    del criterion
    del optimizer
    del loader
    # del debatched_loader
    del dataset
    # gc.collect()
    torch.cuda.empty_cache()
    print("memory allocated after dels")
    if device == "cuda":
        print(torch.cuda.memory_summary(device=device))
    # breakpoint()
    test_axes_x, test_axes_y = torch.meshgrid(0.5 * x_axis, 0.5 * y_axis)
    test_points = torch.vstack((test_axes_x.ravel(), test_axes_y.ravel())).t()
    plt.scatter(
        test_points.cpu()[:, 0].numpy(),
        test_points.cpu()[:, 1].numpy(),
        s=0.5,
        c="black",
    )
    plt.plot()
    plt.show()
    if get_predictions:
        prediction = classifier.predict_point(test_points, det_count=1000)

        # save the prediction
        torch.save(prediction, "prediction.pt")
    else:
        prediction = torch.load("prediction.pt")

    plt.contourf(
        test_axes_x.cpu().numpy(),
        test_axes_y.cpu().numpy(),
        prediction[:, 0].reshape(test_axes_x.shape).cpu().numpy(),
    )
    plt.colorbar()
    plt.show()
    plt.contourf(
        test_axes_x.cpu().numpy(),
        test_axes_y.cpu().numpy(),
        prediction[:, 1].reshape(test_axes_x.shape).cpu().numpy(),
    )
    plt.colorbar()
    plt.show()
    print("0.0 prediction", prediction)
