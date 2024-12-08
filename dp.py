
import torch
import matplotlib.pyplot as plt
from model import GPTConfig, GPT
import numpy as np
from collections import defaultdict


def add_noise_to_gradients(param, noise_std=1e-3):
    """
    Add Gaussian noise to gradients for differential privacy.
    """
    if param.grad is not None:
        noise = torch.normal(mean=0, std=noise_std, size=param.grad.size()).to(param.device)
        param.grad += noise
