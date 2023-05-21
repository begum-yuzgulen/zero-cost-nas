import torch
import numpy as np

from . import measure
@measure('lipschitz_simple', bn=True)
def get_lipschitz_constant(net, inputs, targets, split_data=1, loss_fn=None):
    # Set the network to evaluation mode
    net.eval()

    # Number of data points
    num_data = inputs.size(0)

    # Split the data into batches
    batch_size = num_data // split_data
    inputs_batches = inputs.split(batch_size)
    targets_batches = targets.split(batch_size)

    # Compute gradients of the network parameters for each batch
    gradients = []
    for inputs_batch, targets_batch in zip(inputs_batches, targets_batches):
        inputs_batch = inputs_batch.clone().detach().requires_grad_(True)
        targets_batch = targets_batch.clone().detach()

        # Forward pass
        outputs = net(inputs_batch)
        loss = loss_fn(outputs, targets_batch) if loss_fn else torch.tensor(0.0)

        # Compute gradients
        gradients_batch = torch.autograd.grad(loss, inputs_batch)[0]
        gradients.append(gradients_batch)

    # Compute the Lipschitz constant
    lipschitz_constant = 0.0
    for gradients_batch in gradients:
        norm = torch.norm(gradients_batch.view(gradients_batch.size(0), -1), dim=1)
        lipschitz_constant = max(lipschitz_constant, torch.max(norm))

    if isinstance(lipschitz_constant, torch.Tensor):
        return lipschitz_constant.item()
    else:
        return lipschitz_constant