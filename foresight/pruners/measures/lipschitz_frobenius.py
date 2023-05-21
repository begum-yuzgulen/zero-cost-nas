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

from . import measure


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

@measure('lipschitz_frobenius', bn=True)
def get_lipschitz_constant(net, inputs, targets, split_data=1, loss_fn=None):
    inputs.requires_grad_(True)
    device = inputs.device
    jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    u, s, vh = np.linalg.svd(jacobs)
    lipschitz = s.max()

    frobenius = np.sqrt((jacobs**2).sum())

    inputs.requires_grad_(False)

    return lipschitz * frobenius

@measure('lipschitz_frobenius_sum', bn=True)
def get_lipschitz_constant(net, inputs, targets, split_data=1, loss_fn=None):
    inputs.requires_grad_(True)
    device = inputs.device
    jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    u, s, vh = np.linalg.svd(jacobs)
    lipschitz = s.max()

    frobenius = np.sqrt((jacobs**2).sum())

    inputs.requires_grad_(False)

    return lipschitz + frobenius

@measure('lipschitz_frobenius_divide', bn=True)
def get_lipschitz_constant(net, inputs, targets, split_data=1, loss_fn=None):
    inputs.requires_grad_(True)
    device = inputs.device
    jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    u, s, vh = np.linalg.svd(jacobs)
    lipschitz = s.max()

    frobenius = np.sqrt((jacobs**2).sum())

    inputs.requires_grad_(False)

    return lipschitz / frobenius
