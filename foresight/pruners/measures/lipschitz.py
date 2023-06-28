# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import numpy as np
import time

from . import measure
from ..p_utils import get_layer_metric_array

def compute_jacobian(net, x):
    x.requires_grad_(True)
    output = net(x)

    # Step 4: Flatten the output tensor
    output_flat = torch.flatten(output)

    # Step 5: Initialize list for gradients
    gradients = []

    # Step 6: Compute gradients of each output element with respect to input tensor
    for i in range(len(output_flat)):
        gradient = torch.autograd.grad(output_flat[i], x, retain_graph=True)[0]
        gradients.append(gradient)

    # Step 7: Stack gradients into a matrix
    jacobian = torch.stack(gradients)

    return jacobian

def get_batch_jacobian(net, x, target, device, split_data):
    x.requires_grad_(True)

    N = x.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        y = net(x[st:en])
        y.backward(torch.ones_like(y))

    jacob = x.grad.detach()
    x.requires_grad_(False)
    return jacob, target.detach()

@measure('lipschitz_spectral', bn=True)
def get_lipschitz_constant(net, inputs, targets, split_data=1, loss_fn=None):
    inputs.requires_grad_(True)
    device = inputs.device
    jacobs, _ = get_batch_jacobian(net, inputs, targets, device, split_data)
    u, s, vh = np.linalg.svd(jacobs)
    lipschitz = s.max()

    inputs.requires_grad_(False)
    end_time = time.time()

    return lipschitz


@measure('lipschitz_supremum', bn=True)
def get_lipschitz_constant(net, inputs, targets, split_data=1, loss_fn=None):
    inputs.requires_grad_(True)
    device = inputs.device
    jacobs, _ = get_batch_jacobian(net, inputs, targets, device, split_data)
    lipschitz = np.abs(jacobs).max()

    inputs.requires_grad_(False)

    return lipschitz


@measure('frobenius', bn=True)
def get_frobenius(net, inputs, targets, split_data=1, loss_fn=None):
    inputs.requires_grad_(True)
    device = inputs.device
    jacobs, _ = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    frobenius = np.sqrt((jacobs**2).sum())

    inputs.requires_grad_(False)
    return frobenius

@measure('nuclear', bn=True)
def get_nuclear_norm(net, inputs, targets, split_data=1, loss_fn=None):
    inputs.requires_grad_(True)
    device = inputs.device
    jacobs, _ = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1)
    u, s, vh = np.linalg.svd(jacobs)

    nuclear_norm = np.sum(s)

    inputs.requires_grad_(False)
    return nuclear_norm