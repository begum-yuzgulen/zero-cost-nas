import torch
import numpy as np

from . import measure

import torch
import numpy as np

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

def spectral_norm(net):
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                weight = module.weight
                u = weight.new_empty((weight.size(0), 1)).normal_()
                v = weight.new_empty((1, weight.size(1))).normal_()
                sigma = torch.norm(weight.view(weight.size(0), -1), dim=1)
                u = torch.nn.functional.normalize(u, dim=0)
                v = torch.nn.functional.normalize(v, dim=1)
                for _ in range(30):
                    v = torch.nn.functional.normalize(torch.matmul(u.t(), weight.view(weight.size(0), -1)), dim=1).t()
                    u = torch.nn.functional.normalize(torch.matmul(weight.view(weight.size(0), -1), v), dim=0)
                sigma = torch.sum(u * torch.matmul(weight.view(weight.size(0), -1), v))
                module.weight.data /= sigma.item()

@measure('lipschitz_spectralnorm', bn=True)
def get_lipschitz_constant(net, inputs, targets, split_data=1, loss_fn=None):
    inputs.requires_grad_(True)
    device = inputs.device
    jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    # Spectral Normalization
    spectral_norm(net)

    u, s, vh = np.linalg.svd(jacobs)
    lipschitz = s.max()

    inputs.requires_grad_(False)

    return lipschitz


@measure('lipschitz_eigenvalue', bn=True)
def get_lipschitz_constant_max_eigenvalue(net, inputs, targets, split_data=1, loss_fn=None):
    inputs.requires_grad_(True)
    device = inputs.device
    jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    cov = np.cov(jacobs, rowvar=False)
    eigenvalues = np.linalg.eigvals(cov)
    lipschitz = np.sqrt(eigenvalues.max()).real

    inputs.requires_grad_(False)

    return lipschitz
