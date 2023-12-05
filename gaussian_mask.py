import torch
import math
import torch.nn as nn
import numpy as np
import os
import sys

#import Gaussian_RF as gauss



def get_gaussian_mask(attn_p, mask_prev=None, heatmap_s=[192,256], sigma=None, device='cuda'):
    '''
    Args: 
        attn_p: (float x, float y), range -1~1
        mask_prev: (b, )
        heatmap_s: (int h, int w)
        sigma: tensor, (b, 1)
    Returns:
        gaussian_kernel: (b, 1, heatmap_s[0], heatmap_s[1])
    '''
    batch_s = attn_p.size()[0]
    if mask_prev is None:
        mask_prev = torch.ones((batch_s, heatmap_s[0], heatmap_s[1]), device=device)
    if sigma is None:
        sigma = torch.ones((batch_s, 1), device=device) * 25.0

    region_cur = get_gaussian_kernel(attn_p, sigma, kernel_size=heatmap_s, norm='max', device=device)
    
    mask_cur = nn.functional.relu(mask_prev - region_cur)
    return mask_cur


def get_gaussian_kernel(attn_p, sigmas, kernel_size=(256,256), norm='max', device='cuda'):
    '''
        Args:
            attn_p: (b, 2), tensor, fixation center ranged in -1~1
            kernel_size: (int y, int x), size of kernel returned.
            sigmas: (b, 1), tensor float, sigma of Gaussian kernel.
        Return:
            gaussian_kernel: (b, 1, h, w)
    '''
    sigmas = torch.abs(sigmas) + 1e-6
    batch_s = attn_p.size(0)

    x_coord = torch.arange(kernel_size[1], device=device)
    x_grid = x_coord.repeat(kernel_size[0]).view(kernel_size[0], kernel_size[1])

    y_coord = torch.arange(kernel_size[0], device=device)
    y_grid = y_coord.repeat(kernel_size[1]).view(kernel_size[1], kernel_size[0])
    y_grid = y_grid.t()

    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    xy_grid = xy_grid.unsqueeze(0).repeat(batch_s, 1, 1, 1) # (b, h, w, 2)

    mean = (attn_p+1.0)/2.0 * torch.tensor([kernel_size[1], kernel_size[0]], device=device).unsqueeze(0) #modified 20200712
    variance = sigmas**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*np.pi*variance.unsqueeze(1))) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean.unsqueeze(1).unsqueeze(1))**2., dim=-1) /\
                                (2*variance.unsqueeze(1))
                        )

    if norm == 'max':
        # Always max value should be 1.
        m = torch.max(gaussian_kernel.view(batch_s, -1), 1)[0]
        gaussian_kernel = gaussian_kernel / m.unsqueeze(1).unsqueeze(1)
    elif norm == 'sum':
        # Make sure sum of values in gaussian kernel equals 1.
        m = torch.sum(gaussian_kernel.view(batch_s, -1), 1)
        gaussian_kernel = gaussian_kernel / m.unsqueeze(1).unsqueeze(1)

    return gaussian_kernel.unsqueeze(1)
