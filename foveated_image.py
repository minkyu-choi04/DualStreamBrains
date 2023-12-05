import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import gaussian_mask as G


def rgb_to_gray_even(images):
    # Set equal weights for the three channels when summing them up to get a grayscale image
    weights = torch.tensor([1/3., 1/3., 1/3.]).view(1,3,1,1).to(images.device)
    grayscale_images = torch.sum(images * weights, dim=1, keepdim=True)

    # Repeat grayscale image in three channels
    grayscale_images = grayscale_images.repeat(1, 3, 1, 1)

    return grayscale_images


def image_gaussian_blurr(imgs, sigma, kernel_size, device='cuda'):
    '''
    2023.06.15. Minkyu
    Perform Gaussian blurr. 
    Blurring is applied to each channel so that it keeps natural colors. 
    Args:
      imgs: tensor, natural images. size: (batch, 3, h, w)
      sigma: int, sigma of gaussian kernel
      kernel_size: int, kernel size of gaussian kernel. 
    Return:
      img_g: tensor, blurred natural images. size: (batch, 3, h, w)
    '''
    batch_size, c, h, w = imgs.size()
    pd = int((kernel_size-1)/2)
    img_pad = F.pad(imgs, (pd,pd,pd,pd), mode='replicate')
    img_pad_batch = img_pad.view(3*batch_size, 1, h+2*pd, w+2*pd)

    sigma = torch.ones((batch_size, 1), device=device) * sigma
    kernel = G.get_gaussian_kernel(torch.zeros((1, 2), device=device), 
                                   kernel_size=(kernel_size, kernel_size), 
                                   sigmas=sigma, channels=3, norm='sum', device=device)
    
    img_g = F.conv2d(img_pad_batch, kernel)
    img_g = img_g.view(batch_size, 3, h, w)
    return img_g


def create_gaussian(height, width, x, y, std=1.0):
    """Create a 2D Gaussian distribution."""
    y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height))
    y_grid, x_grid = y_grid.to('cuda'), x_grid.to('cuda')
    x_grid = x_grid - x
    y_grid = y_grid + y   # flip y-axis direction
    d = torch.sqrt(x_grid**2 + y_grid**2)
    sigma = std
    gaussian = torch.exp(-((d)**2 / (2.0 * sigma**2)))
    return gaussian

def color_to_gray(image):
    """Convert an image to grayscale."""
    gray = image.mean(dim=0, keepdim=True)
    return gray

def get_color_foveation(images, fixs_xy, std):
    """Transform an image based on a 2D Gaussian distribution."""
    batch, _, height, width = images.size()
    transformed_images = []
    masks = []

    for i in range(batch):
        # create 2D Gaussian distribution
        gaussian = create_gaussian(height, width, fixs_xy[i, 0], -1*fixs_xy[i, 1], std)

        # normalize the Gaussian distribution
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

        # Convert to 3-channel mask
        mask = gaussian.repeat((3, 1, 1))
        masks.append(mask)

        # create grayscale image
        gray_image = color_to_gray(images[i])

        # create the transformed image
        transformed_image = mask * images[i] + (1 - mask) * gray_image

        transformed_images.append(transformed_image)

    return torch.stack(transformed_images), torch.stack(masks)

class Get_foveated_images(nn.Module):
    ''' See <Scanpath estimation based on foveated image saliency>
            By Yixiu Wang, Bin Wang, Xiaofeng Wu, Liming Zhang'''
    def __init__(self, sigmas, mix_r, kss=11, device='cuda'):
        super().__init__()
        '''2020.11.11
        Make foveated images. 
        Args:
            imgs: (b, channels, h, w), tensor image
            fixs: (b, 2), (float x, float y) ranged -1~1, tensor
            kss: int, it defines the sizes of Gaussian kernels, must be odd numbers
            sigmas: (n_stages, 1), int, sigmas of each stages
        Returns:
            img_accum: (b, channels, h, w), tensor, foveated image
        '''
        self.n_stages = len(sigmas)
        self.kss = kss
        self.sigmas = sigmas
        self.mix_r = mix_r
        self.device = device

    def forward(self, imgs, fixs):
        batch_s, channel, h, w = imgs.size()

        img_gauss = []
        img_gauss.append(imgs)
        pd = int((self.kss-1)/2)
        img_pad = F.pad(imgs, (pd,pd,pd,pd), mode='replicate')
        # img_pad: (b, 3, h, w)
        img_pad_batch = img_pad.view(3*batch_s, 1, h+2*pd, w+2*pd)
        # img_pad_batch: (3*b, 1, h, w)
        # This reshaping is required because each channel of images must be applied with 
        #   Gaussian kernels individually. Therefore, by reshaping it to move channels to batch
        #   dimension, conv2d will operate on each channel indivisually. 

        self.gks = []
        for stg in range(self.n_stages):
            #print(self.n_stages, stg, sigmas[stg])
            self.gks.append(G.get_gaussian_kernel(torch.zeros((1, 2), device='cuda'), 
                                                    kernel_size=(self.kss,self.kss), 
                                                    sigmas=self.sigmas[stg]*torch.ones((batch_s, 1), device=self.device), 
                                                    norm='sum', 
                                                    device='cuda'))
            # list of (1, 1, h, w)

        ### Gaussian Blurred Images
        for stg in range(self.n_stages):
            img_g = F.conv2d(img_pad_batch, self.gks[stg])
            img_gauss.append(img_g.view(batch_s, 3, h, w))
            # list of (b, 3, h, w)
        # At this point, multi-level Gaussian kernered image is obtained. 
        # img_gauss: (n_stages+1, tensor(b, 3, h, w))

        ### Blend weights
        weights = []
        for stg in range(self.n_stages):
            weights.append(G.get_gaussian_kernel(fixs, 
                                                kernel_size=(h,w), 
                                                sigmas=self.mix_r[stg]*torch.ones((batch_s, 1), device=self.device), 
                                                norm='max', 
                                                device='cuda'))
            # list of (b, 1, h, w)
        w_diff = []
        w_diff.append(weights[0])
        for stg in range(self.n_stages):
            if stg != self.n_stages-1:
                w_diff.append(weights[stg+1] - weights[stg])
            else:
                w_diff.append(1 - weights[stg])
                # list of (b, 1, h, w)
        # w_diff: (n_stages+1, tensor(b, 1, h, w))

        ### Blend Images into one
        # img_gauss: (n_stages+1, tensor(b, 3, h, w))
        # w_diff:    (n_stages+1, tensor(b, 1, h, w))
        for stg in range(self.n_stages+1):
            if stg ==0:
                img_accum = img_gauss[stg] * w_diff[stg]
            else:
                img_accum = img_accum + img_gauss[stg] * w_diff[stg]

        return img_accum
