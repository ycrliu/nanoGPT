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
        In other words, (100 - sparsity_level)% is the amount that remains post-pruning.
    """
    # Group weights by layer
    layer_weights = defaultdict(list)
    for name, param in model.named_parameters():
        if "weight" in name:  # Filter to include only weight parameters
            if len(name.split(".")) < 4: continue
            layer = name.split(".")[3]
            layer_weights[layer].append(param)
    
    # Calculate the threshold and apply sparsity for each layer
    for layer, params in layer_weights.items():
        # Collect all weights across the entire layer to compute a global threshold
        all_weights = torch.cat([param.data.view(-1).abs() for param in params])

        # Determine the threshold for pruning based on the desired sparsity level
        k = int(all_weights.numel() * sparsity_level / 100)

        if k > 0:
            # `kthvalue` gets the kth smallest value, so threshold prunes `sparsity_level` amount of weights
            threshold = torch.kthvalue(all_weights, k).values
            
            for param in params:
                # Check if the entire parameter tensor falls below the threshold
                mask = param.data.abs() > threshold  # 1 where tensor's magnitude is above threshold
                param.data *= mask
                param.data[param.data.abs() <= threshold] = 0


def sparsify_threshold_based_global(model, sparsity_level, threshold=1e-5):
    for _, param in model.named_parameters():
        if param.dim() > 1:
            abs_weights = param.abs()

            # Calculate threshold for desired sparsity
            k = int(param.numel() * sparsity_level / 100)
            threshold = torch.kthvalue(abs_weights.view(-1), k).values

            # Zero out weights below threshold
            mask = abs_weights > threshold
            param.data *= mask


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


def assess_sparsity_structure(model, sparsed=False, zero_tol=1e-3, file_name_append=""):

    layer_weights = defaultdict(list)

    # Go through each parameter in the model
    for name, param in model.named_parameters():
        if "weight" in name:  # Filter to include only weight parameters
            if len(name.split(".")) < 4: continue
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
    plt.savefig(f"{file_name_append}_{'after' if sparsed else 'before'}_all_layers_{layer_name}_weight_distribution.png")
    plt.close()



def assess_overall_weight_distribution(model, tolerance=1e-3, sparsed=False, file_name_append=""):
    """
    Aggregates and plots the weight distribution across all layers in the model.
    """
    all_weights = []  # List to store all weights

    # Collect weights from all layers
    for name, param in model.named_parameters():
        if "weight" in name:  # Filter to include only weight parameters
            weights = param.cpu().detach().numpy().flatten()
            # Apply thresholding: set weights below the tolerance to zero
            weights[np.abs(weights) < tolerance] = 0
            all_weights.append(weights)

    # Flatten the list of arrays into a single array for plotting
    all_weights = np.concatenate(all_weights)

    # Plot the overall weight distribution
    plt.figure(figsize=(10, 5))
    plt.hist(all_weights, bins=50, range=(-0.75, 0.75))
    plt.xlabel("Weights")
    plt.ylabel("Count")
    plt.title("Overall Weight Distribution Across All Layers (Threshold Applied)")
    plt.savefig(f"{file_name_append}_{'after' if sparsed else 'before'}_overall_weight_distribution_{name}.png")
    plt.close()


