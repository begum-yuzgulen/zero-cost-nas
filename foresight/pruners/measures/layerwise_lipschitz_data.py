import torch
import torch.nn as nn
import numpy as np

from . import measure

@measure('layerwise_lipschitz_data', bn=True)
def compute_lipschitz_constant(net, inputs, targets, split_data=1, loss_fn=None):
    lipschitz_constants = []

    N = inputs.shape[0]
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.zero_grad()
            layer.weight.requires_grad = True

            outputs = net.forward(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            weight_grad = layer.weight.grad.detach()
            weight_grad_norm = torch.norm(weight_grad.view(weight_grad.size(0), -1), dim=1)
            lipschitz_constants.append(weight_grad_norm.max())

            layer.weight.requires_grad = False

    return lipschitz_constants