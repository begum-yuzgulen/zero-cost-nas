import torch
import numpy as np

from . import measure
@measure('rademacher', bn=True)
def get_rademacher(net, inputs, targets, split_data=1, loss_fn=None):
    """
    Compute the Rademacher complexity of a neural network model.
    
    Args:
        net (torch.nn.Module): The neural network model.
        inputs (torch.Tensor): The input samples.
        targets (torch.Tensor): The corresponding labels.
        split_data (int): Number of splits to divide the data for Rademacher complexity estimation.
        loss_fn (torch.nn.Module, optional): The loss function to compute empirical risk. Defaults to None.
    
    Returns:
        float: The estimated Rademacher complexity.
    """
    device = next(net.parameters()).device
    batch_size = inputs.size(0)
    
    # Generate random Rademacher variables
    rademacher_vars = (torch.randint(0, 2, size=(batch_size, split_data), device=device) * 2 - 1).float()
    
    if loss_fn is None:
        loss_fn = torch.nn.functional.cross_entropy
    
    complexity_sum = 0.0
    for i in range(split_data):
        start_idx = int(i * (batch_size / split_data))
        end_idx = int((i + 1) * (batch_size / split_data))
        data = inputs[start_idx:end_idx].to(device)
        targets_batch = targets[start_idx:end_idx].to(device)

        # Compute model predictions
        outputs = net(data)

        # Compute empirical risk
        loss = loss_fn(outputs, targets_batch)

        # Compute Rademacher complexity term
        complexity_sum += loss * rademacher_vars[:, i].mean()

    # Average over the number of splits
    rademacher_complexity = complexity_sum / split_data

    return rademacher_complexity.item()