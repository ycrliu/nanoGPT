import torch
import matplotlib.pyplot as plt
from model import GPTConfig, GPT
import numpy as np
from collections import defaultdict


def sparsify_threshold_based(model, sparsity_level):
    """
    Applies static sparsification to a given `model` by pruning its weights.
    Uses a threshold-based sparsification (magnitude-based) by removing weights with the
    smallest magnitudes, under the assumption that smaller weights contribute less to the output.

    sparsity_level: percentage between 0 and 100, represents the amount of weights to be pruned.
    """
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad and param.dim() >= 2:

            # Flatten the parameter to easily access all values
            flat_weights = param.data.view(-1)

            # Calculate the threshold below which weights will be zeroed
            k = int(flat_weights.numel() * (sparsity_level / 100))
            if k == 0:
                continue  # Skip sparsification if the reduction percentage is very small

            # Get the absolute values of weights and find the threshold value
            threshold = flat_weights.abs().topk(k, largest=False).values.max()

            # Apply sparsity by setting weights below the threshold to zero
            mask = param.data.abs() >= threshold
            param.data *= mask  # Zero out the small-magnitude weights


def sparsify_random_based(model, sparsity_level,):
    """
    Applies static sparsification to a given `model` by pruning its weights.
    Uses a threshold-based sparsification (magnitude-based) by removing weights with the
    smallest magnitudes, under the assumption that smaller weights contribute less to the output.

    sparsity_level: percentage between 0 and 100, represents the amount reduced
    """

    for name, param in model.named_parameters():
        if "weight" in name:
            param_data = param.data.view(-1)
            mask = (torch.rand_like(param_data) > sparsity_level / 100).float()
            param_data *= mask


def assess_sparsity_structure(model, sparsed=False, zero_tol=1e-8):

    sparsity_data = []  # List to store sparsity data for each layer
    layer_weights = defaultdict(list)

    # Go through each parameter in the model
    for name, param in model.named_parameters():
        if "weight" in name:  # Filter to include only weight parameters
            layer = name.split(".")[3]
            layer_weights[layer].append(param.cpu().detach().numpy().flatten())

    # Calculate sparsity and plot for each layer
    layer_sparsity_data = {}  # Store sparsity data for each layer
    for layer_name, weights_list in layer_weights.items():
        # Concatenate weights in the layer and calculate sparsity
        all_weights = torch.tensor(np.concatenate(weights_list))
        total_elements = all_weights.numel()
        non_zero_elements = torch.sum(all_weights.abs() > zero_tol).item()
        sparsity_fraction = non_zero_elements / total_elements  # Fraction of weights > 0
        layer_sparsity_data[layer_name] = sparsity_fraction


    # Plot fraction of non-zero weights by layer
    layer_names = list(layer_sparsity_data.keys())
    non_zero_fractions = list(layer_sparsity_data.values())

    plt.figure(figsize=(12, 6))
    plt.bar(layer_names, non_zero_fractions)
    plt.xticks(rotation=90)
    plt.ylabel("Fraction of Non-Zero Weights")
    plt.xlabel("Layer Index")
    plt.title("Non-Zero Weight Fractions by Layer")
    plt.savefig(f"{'after' if sparsed else 'before'}_all_layers_{layer_name}_weight_distribution.png")
    plt.close()

def assess_overall_weight_distribution(model, sparsed=False):
    """
    Aggregates and plots the weight distribution across all layers in the model.
    """
    all_weights = []  # List to store all weights

    # Collect weights from all layers
    for name, param in model.named_parameters():
        if "weight" in name:  # Filter to include only weight parameters
            all_weights.append(param.cpu().detach().numpy().flatten())

    # Flatten the list of arrays into a single array for plotting
    all_weights = np.concatenate(all_weights)


    # Plot the overall weight distribution
    plt.figure(figsize=(10, 5))
    plt.hist(all_weights, bins=50, range=(-0.75, 0.75))
    plt.xlabel("Weights")
    plt.ylabel("Count")
    plt.title("Overall Weight Distribution Across All Layers")
    plt.savefig(f"{'after' if sparsed else 'before'}_overall_weight_distribution_{name}.png")  # Save plot for each layer
    plt.close()
