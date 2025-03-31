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
from mnist_classification import AE
from itertools import islice

dimension = 2  # the dimension of the latent space
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
transform = transforms.Compose([transforms.ToTensor()])
print("got transforms")

# load the encoder
model = AE(dimension).to(device)
model.encoder.load_state_dict(torch.load("encoder.pt"))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
print("got transforms")
encoded_value_sets = torch.load("encoded_value_sets.pt")
dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
debatched_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    generator=torch.Generator(device=device),
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

# classifier
order = 20
basis_params = [
    {
        "upper_bound": torch.Tensor([18.0]).to(device),
        "lower_bound": torch.Tensor([-18.0]).to(device),
    }
] * dimension
basis_functions = [standard_chebyshev_basis] * dimension
basis = Basis(basis_functions, dimension, order, basis_params)
parameters = PriorParameters(mean=0.0, alpha=1.5, beta=1.0, nu=0.1)
hyperparameters = GCPOSEHyperparameters(basis, dimension)
classifier = GCPClassifier(
    len(labels), parameters, hyperparameters, stabilising_epsilon=0.0
)

print("initialised the classifier")

data_count = 550
for c, encoded_values in enumerate(encoded_value_sets):
    classifier.add_data(encoded_values[:data_count, :], c)

# relevant label choices
label_counts = [0] * len(labels)
for i, target in enumerate(dataset.targets):
    if int(target) not in labels:
        continue
    index = labels.index(int(target))
    label_counts[index] += 1
    relevant_indices.append(i)

for i, (image, class_label) in islice(enumerate(debatched_loader), 5):
    input_image = image.reshape(-1, 784).to(device)
    encoding = model.encoder(input_image)
    del input_image
    result = (
        classifier.predict_point(encoding, det_count=1000)
        .cpu()
        .detach()
        .numpy()[0]
    )
    plt.title(f"Predicted: Prob(3): {result[0]}, Prob(8): {result[1]}")
    plt.xlabel(f"Actual: {class_label}")
    plt.imshow(image.squeeze().cpu().numpy())
    plt.savefig(f"digit_prediction_{i}.png")
    # plt.show()
