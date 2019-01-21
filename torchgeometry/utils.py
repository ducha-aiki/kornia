import torch
import torch.nn as nn
import torchgeometry as tgm

import numpy as np


__all__ = [
    "tensor_to_image",
    "image_to_tensor",
    "draw_rectangle",
    "create_pinhole",
    "create_meshgrid",
]


def create_meshgrid(height, width, normalized_coordinates=True):
    """Generates a coordinate grid for an image.

    This is normalized to be in the range [-1,1] to be consistent with the
    pytorch function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): wether to used normalized coordinates in
        the range [-1, 1] in order to be consistent with the PyTorch function
        grid_sample.

    Return:
        Tensor: returns a 1xHxWx2 grid.
    """
    # generate coordinates
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxwx2


def image_to_tensor(image):
    """Converts a numpy image to a torch.Tensor image.

    Args:
        image (numpy.ndarray): image of the form (H, W, C).

    Returns:
        torch.Tensor: tensor of the form (C, H, W).

    """
    if not type(image) == np.ndarray:
        raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
            type(image)))

    if len(image.shape) > 3 or len(image.shape) < 2:
        raise ValueError("Input size must be a two or three dimensional array")

    tensor = torch.from_numpy(image)

    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=-1)

    return tensor.permute(2, 0, 1).squeeze_()  # CxHxW


def tensor_to_image(tensor):
    """Converts a torch.Tensor image to a numpy image. In case the tensor is in
       the GPU, it will be copied back to CPU.

    Args:
        tensor (Tensor): image of the form (C, H, W).

    Returns:
        numpy.ndarray: image of the form (H, W, C).

    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) > 3 or len(tensor.shape) < 2:
        raise ValueError(
            "Input size must be a two or three dimensional tensor")

    input_shape = tensor.shape
    if len(input_shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=0)

    tensor = tensor.permute(1, 2, 0)

    if len(input_shape) == 2:
        tensor = torch.squeeze(tensor, dim=-1)

    return tensor.contiguous().cpu().detach().numpy()


def create_pinhole(intrinsic, extrinsic, height, width):
    pinhole = torch.zeros(12)
    pinhole[0] = intrinsic[0, 0]  # fx
    pinhole[1] = intrinsic[1, 1]  # fy
    pinhole[2] = intrinsic[0, 2]  # cx
    pinhole[3] = intrinsic[1, 2]  # cy
    pinhole[4] = height
    pinhole[5] = width
    pinhole[6:9] = tgm.rotation_matrix_to_angle_axis(
        torch.tensor(extrinsic))
    pinhole[9:12] = torch.tensor(extrinsic[:, 3])
    return pinhole.view(1, -1)
