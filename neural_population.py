import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import numpy as np
from typing import Tuple


def convert_xy2ret(xy: torch.Tensor, a: float, b: float, m: int, n: int) -> torch.Tensor:
    """
    Convert x or y to retinal coordinates.

    Args:
        xy (torch.Tensor): Grid coordinates of either x or y (shape: (b, 1)).
        a (float): Parameter a as per the referenced paper.
        b (float): Parameter b as per the referenced paper.
        m (int): Resolution of the full image.
        n (int): Resolution of the sampled image.

    Returns:
        torch.Tensor: Transformed coordinates corresponding to retinal coordinates.
    """
    xy_ret = 2 / a * torch.log(1 - np.sqrt(np.pi) / b * (1 - np.exp(a / 2)) * xy)
    return xy_ret

def convert_ret2xy(xy_ret: torch.Tensor, a: float, b: float, m: int, n: int) -> torch.Tensor:
    """
    Convert retinal coordinates to x or y coordinates.

    Args:
        xy_ret (torch.Tensor): Retinal grid coordinates (shape: (b, 1)).
        a (float): Parameter a as per the referenced paper.
        b (float): Parameter b as per the referenced paper.
        m (int): Resolution of the full image.
        n (int): Resolution of the sampled image.

    Returns:
        torch.Tensor: Transformed coordinates corresponding to x or y.
    """
    xy = b / np.sqrt(np.pi) * ((1 - torch.exp(a / 2 * xy_ret)) / (1 - np.exp(a / 2)))
    return xy

def make_regular_grid(
    range_x: Tuple[float, float] = (-1, 1),
    range_y: Tuple[float, float] = (-1, 1),
    res_xy: Tuple[int, int] = (64, 64),
    device: str = 'cuda'
    ) -> torch.Tensor:
    """
    Make a regular grid.

    Args:
        range_x (Tuple[float, float]): Range of x values for the grid.
        range_y (Tuple[float, float]): Range of y values for the grid.
        res_xy (Tuple[int, int]): Resolution of the grid.
        device (str): The device to create the grid on.

    Returns:
        torch.Tensor: Regularly spaced grid.
    """
    xxrange = torch.linspace(range_x[0], range_x[1], res_xy[0], device=device)
    yyrange = torch.linspace(range_y[0], range_y[1], res_xy[1], device=device)
    ys, xs = torch.meshgrid(yyrange, xxrange)
    grid_reg = torch.stack([xs, ys], dim=-1).unsqueeze(0)
    return grid_reg


def convert_xy2rt(xy: torch.Tensor) -> torch.Tensor:
    """
    Convert xy coordinates to polar coordinates (r, theta).

    Args:
        xy (torch.Tensor): Tensor containing xy coordinates with shape (b, 2).

    Returns:
        torch.Tensor: Tensor containing polar coordinates with shape (b, 2).
    """
    rs = torch.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
    ts = torch.atan2(xy[:, 1], xy[:, 0])
    rt = torch.stack([rs, ts], dim=1)
    return rt

def convert_rt2xy(rt: torch.Tensor) -> torch.Tensor:
    """
    Convert polar coordinates (r, theta) to xy coordinates.

    Args:
        rt (torch.Tensor): Tensor containing polar coordinates with shape (b, 2).

    Returns:
        torch.Tensor: Tensor containing xy coordinates with shape (b, 2).
    """
    rs = rt[:, 0]
    ts = rt[:, 1]
    xs = rs * torch.cos(ts)
    ys = rs * torch.sin(ts)
    xy = torch.stack([xs, ys], dim=1)
    return xy

def make_xy2ret_grid_r(fixs_xy: torch.Tensor, m: int, n: int, density_ratio: float) -> torch.Tensor:
    """
    Make an xy to retinal coordinates grid.

    Args:
        fixs_xy (torch.Tensor): Tensor containing fixation points with shape (b, 2).
        m (int): Size of the input image.
        n (int): Size of the output image.
        density_ratio (float): Relative density of the field of view to the periphery.

    Returns:
        torch.Tensor: The grid in xy coordinates.
    """
    output_size = (n, n)
    batch_s = fixs_xy.size(0)

    rp_max = n / m
    r_max = 1
    a = np.log(density_ratio) / rp_max
    b = np.sqrt(np.pi) * r_max * (1 - np.exp(a/2)) / (1 - np.exp(a / 2 * rp_max))

    fixs_xy_length = 1 - torch.abs(fixs_xy)

    fixs_xp = torch.sign(fixs_xy[:, 0]) * (n/m - convert_xy2ret(fixs_xy_length[:, 0], a, b, m, n))
    fixs_yp = torch.sign(fixs_xy[:, 1]) * (n/m - convert_xy2ret(fixs_xy_length[:, 1], a, b, m, n))
    fixs_xyp = torch.cat((fixs_xp.unsqueeze(1), fixs_yp.unsqueeze(1)), dim=1)

    grid_lp_xyp_reg = make_regular_grid(range_x=(-1, 1), range_y=(-1, 1), res_xy=output_size[::-1], device=fixs_xy.device)
    grid_lp_xyp_reg = grid_lp_xyp_reg * (n / m)
    grid_lp_xyp_reg = grid_lp_xyp_reg.repeat(batch_s, 1, 1, 1)
    grid_lp_xyp_reg = grid_lp_xyp_reg - fixs_xyp.unsqueeze(1).unsqueeze(1)

    grid_rtp = convert_xy2rt(grid_lp_xyp_reg.view(-1, 2))
    grid_r = convert_ret2xy(grid_rtp.view(-1, 2)[:, 0], a, b, m, n)
    grid_rt = torch.cat((grid_r.unsqueeze(1), grid_rtp.view(-1, 2)[:, 1].unsqueeze(1)), dim=1)
    grid_xy = convert_rt2xy(grid_rt).view(batch_s, *output_size, 2)
    grid_xy = grid_xy + fixs_xy.unsqueeze(1).unsqueeze(1)

    return grid_xy.view(grid_lp_xyp_reg.size())

def make_ret2xy_grid_r(fixs_xy: torch.Tensor, m: int, n: int, density_ratio: float) -> torch.Tensor:
    """
    Generates a grid for converting image from retinal space to cartesian space.

    Args:
        fixs_xy (torch.Tensor): Tensor of fixation points with shape (b, 2).
        m (int): Size of the input image.
        n (int): Size of the output image.
        density_ratio (float): Relative density of the field of view to the periphery.

    Returns:
        torch.Tensor: The grid in retinal space.
    """
    output_size = (n, n)
    batch_s = fixs_xy.size(0)

    rp_max = n / m
    r_max = 1

    a = np.log(density_ratio) / rp_max
    b = np.sqrt(np.pi) * r_max * (1 - np.exp(a/2)) / (1 - np.exp(a / 2 * rp_max))

    fixs_xy_length = 1 - torch.abs(fixs_xy)
    fixs_xp = torch.sign(fixs_xy[:, 0]) * (n/m - convert_xy2ret(fixs_xy_length[:, 0], a, b, m, n))
    fixs_yp = torch.sign(fixs_xy[:, 1]) * (n/m - convert_xy2ret(fixs_xy_length[:, 1], a, b, m, n))
    fixs_xyp = torch.cat((fixs_xp.unsqueeze(1), fixs_yp.unsqueeze(1)), dim=1)

    grid_lp_xy_reg = make_regular_grid(res_xy=output_size[::-1], device=fixs_xy.device)
    grid_lp_xy_reg = grid_lp_xy_reg.repeat(batch_s, 1, 1, 1)
    grid_lp_xy_reg = grid_lp_xy_reg - fixs_xy.unsqueeze(1).unsqueeze(1)

    grid_rt = convert_xy2rt(grid_lp_xy_reg.view(-1, 2))
    grid_rp = convert_xy2ret(grid_rt[:, 0], a, b, m, n)
    grid_rtp = torch.cat((grid_rp.unsqueeze(1), grid_rt[:, 1].unsqueeze(1)), dim=1)
    grid_xyp = convert_rt2xy(grid_rtp).view(batch_s, *output_size, 2)
    grid_xyp = grid_xyp + fixs_xyp.unsqueeze(1).unsqueeze(1)
    grid_xyp = grid_xyp / (n / m)

    return grid_xyp.view(grid_lp_xy_reg.size())


def mark_point(imgs: torch.Tensor, fixs: torch.Tensor, ds: int = 7, is_red: bool = True) -> torch.Tensor:
    """
    Marks a point on a given batch of images.

    Args:
        imgs (torch.Tensor): A batch of images with shape (b, 3, h, w) and any range.
        fixs (torch.Tensor): A batch of fixation points with shape (b, 2) and values in the range -1 to 1.
        ds (int): Size of the marked square around the fixation point.
        is_red (bool): If True, marks with red color; otherwise, marks with blue.

    Returns:
        torch.Tensor: The batch of images with marked points.
    """
    # Convert fixation points to the image scale (0 to h-1 and w-1)
    fixs = ((fixs + 1) / 2.0) * torch.tensor([imgs.shape[3], imgs.shape[2]], dtype=torch.int32, device=imgs.device)
    fixs = fixs.to(torch.int)

    # Mark the points in the images
    for b in range(imgs.shape[0]):
        # Define color channels (red or blue)
        color_channel = 0 if is_red else 2

        # Ensure the marked square does not go out of image bounds
        y_min = max(fixs[b, 1] - ds, 0)
        y_max = min(fixs[b, 1] + ds, imgs.shape[2])
        x_min = max(fixs[b, 0] - ds, 0)
        x_max = min(fixs[b, 0] + ds, imgs.shape[3])

        # Clear the square area in all channels
        imgs[b, :, y_min:y_max, x_min:x_max] = 0.0

        # Set the specified color channel to a high value (assuming max value is 2.0 for the image)
        imgs[b, color_channel, y_min:y_max, x_min:x_max] = 2.0

    return imgs


def convert_coords_ret2xy_r(fixs_xy: torch.Tensor, coords: torch.Tensor, m: int, n: int, density_ratio: float) -> torch.Tensor:
    """
    Converts coordinates in x'y' to xy space.

    Args:
        fixs_xy (torch.Tensor): Fixation points in xy space with shape (b, 2) and range -1 to 1.
        coords (torch.Tensor): Coordinates in x'y' space with shape (b, 2) and range -1 to 1.
        m (int): Parameter for conversion.
        n (int): Parameter for conversion.
        density_ratio (float): Density ratio for conversion.

    Returns:
        torch.Tensor: Coordinates in xy space with shape (b, 2) and range -1 to 1.
    """
    output_size = (n, n)
    batch_s = fixs_xy.size(0)

    # Conversion constants
    rp_max = n / m
    r_max = 1

    a = np.log(density_ratio) / rp_max
    b = np.sqrt(np.pi) * r_max * (1 - np.exp(a/2)) / (1 - np.exp(a / 2 * rp_max))
    xyp_max = n / m

    # Consider Fixation: Convert fixation from xy to xyp
    fixs_xy_length = 1 - torch.abs(fixs_xy)
    fixs_xp = torch.sign(fixs_xy[:, 0]) * (n/m - convert_xy2ret(fixs_xy_length[:, 0], a, b, m, n))
    fixs_yp = torch.sign(fixs_xy[:, 1]) * (n/m - convert_xy2ret(fixs_xy_length[:, 1], a, b, m, n))
    fixs_xyp = torch.cat((fixs_xp.unsqueeze(1), fixs_yp.unsqueeze(1)), 1)

    # Scale and center grid to fixation point
    grid_lp_xyp_reg = coords * xyp_max - fixs_xyp

    # Convert x'y' to rtp and then to rt
    grid_rtp = convert_xy2rt(grid_lp_xyp_reg)
    grid_r = convert_ret2xy(grid_rtp.view(-1, 2)[:, 0], a, b, m, n)
    grid_rt = torch.cat((grid_r.unsqueeze(1), grid_rtp.view(-1, 2)[:, 1].unsqueeze(1)), 1)

    # Convert rt to xy and restore xy grid fixation
    grid_xy = convert_rt2xy(grid_rt) + fixs_xy

    return grid_xy

