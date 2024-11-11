import torch
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

def sparsify_threshold_based(model, sparsity_level):
    """
    Applies static sparsification to a given `model` by pruning its weights.
    Uses a threshold-based sparsification (magnitude-based) by removing weights with the
    smallest magnitudes, under the assumption that smaller weights contribute less to the output.

    sparsity_level: percentage between 0 and 100, represents the amount reduced
    """

    for name, param in model.named_parameters():
        if "weight" in name:
            # Flatten weights and calculate threshold for sparsification
            param_data = param.data.view(-1)
            threshold = torch.quantile(param_data.abs(), sparsity_level)

            # Zero out weights below the threshold
            param_data[param_data.abs() < threshold] = 0



def sparsify_random_based(model, sparsity_level):
    """
    Applies static sparsification to a given `model` by pruning its weights.
    Uses a threshold-based sparsification (magnitude-based) by removing weights with the
    smallest magnitudes, under the assumption that smaller weights contribute less to the output.

    sparsity_level: percentage between 0 and 100, represents the amount reduced
    """

    for name, param in model.named_parameters():
        if "weight" in name:
            param_data = param.data.view(-1)
            mask = torch.rand_like(param_data) > sparsity_level
            param_data *= mask


def assess_sparsity_structure(model):

    sparsity_data = []  # List to store sparsity data for each layer

    # Go through each parameter in the model
    for name, param in model.named_parameters():
        if "weight" in name:  # Filter to include only weight parameters
            total_elements = param.numel()
            non_zero_elements = param.nonzero().size(0)
            sparsity_fraction = non_zero_elements / total_elements  # Fraction of weights > 0
            sparsity_data.append((name, sparsity_fraction, param.cpu().detach().numpy()))

    # Plot sparsity data
    layer_names = [x[0] for x in sparsity_data]
    non_zero_fractions = [x[1] for x in sparsity_data]

    # Plot fraction of non-zero weights by layer
    plt.figure(figsize=(12, 6))
    plt.bar(layer_names, non_zero_fractions)
    plt.xticks(rotation=90)
    plt.ylabel("Fraction of Non-Zero Weights")
    plt.title("Non-Zero Weight Fractions by Layer")
    plt.show()

    # Plot weight distributions for feedforward layers
    for name, sparsity_fraction, weights in sparsity_data:
        plt.figure(figsize=(10, 5))
        plt.hist(weights.flatten(), bins=50, range=(-0.1, 0.1))
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.title(f"Weight Distribution for Layer: {name}")
        plt.show()

