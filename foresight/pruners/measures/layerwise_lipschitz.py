import torch
import torch.nn as nn
import numpy as np

from . import measure

@measure('layerwise_lipschitz', bn=True)
def compute_lipschitz_constant(net, inputs, targets, split_data=1, loss_fn=None):
    lipschitz_constants = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weight = layer.weight.data

            # Compute Lipschitz constant for Conv2d layer
            if isinstance(layer, nn.Conv2d):
                out_channels, in_channels, kernel_size1, kernel_size2 = weight.shape
                weight = weight.view(out_channels, -1)

            # Compute Lipschitz constant for Linear layer
            elif isinstance(layer, nn.Linear):
                weight = weight.t()

            singular_values = torch.svd(weight, compute_uv=False)[1]
            lipschitz_constants.append(singular_values.max())

    return lipschitz_constants