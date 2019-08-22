from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters import spatial_gradient, gaussian_blur2d


def harris_response(input: torch.Tensor,
                    k: torch.Tensor = 0.04,
                    grads_mode: str = 'sobel') -> torch.Tensor:
    r"""Computes the Harris cornerness function. Function does not do
    any normalization or nms.

    The response map is computed according the following formulation:

    .. math::
        R = max(0, det(M) - k \cdot trace(M)^2)

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}

    and :math:`k` is an empirically determined constant
    :math:`k ∈ [ 0.04 , 0.06 ]`

    Args:
        k (torch.Tensor): the Harris detector free parameter.
        grads_mode (string): can be 'sobel' for standalone use or 'diff' for use on Gaussian pyramid

    Input:
        torch.Tensor: 4d tensor

    Return:
        torch.Tensor: the response map per channel.

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]])  # 1x1x7x7
        >>> # compute the response map
        >>> output = harris_response(input, 0.04)
        tensor([[[[0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012],
          [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
          [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
          [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
          [0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012]]]])
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))
    gradients: torch.Tensor = spatial_gradient(input, grads_mode)
    dx: torch.Tensor = gradients[:, :, 0]
    dy: torch.Tensor = gradients[:, :, 1]

    # compute the structure tensor M elements
    def g(x):
        return gaussian_blur2d(x, (3, 3), (1., 1.))

    dx2: torch.Tensor = g(dx ** 2)
    dy2: torch.Tensor = g(dy ** 2)
    dxy: torch.Tensor = g(dx * dy)
    det_m: torch.Tensor = dx2 * dy2 - dxy * dxy
    trace_m: torch.Tensor = dx2 + dy2
    # compute the response map
    scores: torch.Tensor = torch.relu(det_m - k * (trace_m ** 2))
    return scores


def gftt_response(input: torch.Tensor,
                  grads_mode: str = 'sobel') -> torch.Tensor:
    r"""Computes the Shi-Tomasi cornerness function. Function does not do
    any normalization or nms.

    The response map is computed according the following formulation:

    .. math::
        R = min(eig(M))

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}


    Args:
        grads_mode (string): can be 'sobel' for standalone use or 'diff' for use on Gaussian pyramid

    Input:
        torch.Tensor: 4d tensor

    Return:
        torch.Tensor: the response map per channel.

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]])  # 1x1x7x7
        >>> # compute the response map
        >>> output = gftt_response(input)
        tensor([[[[0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155],
          [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
          [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
          [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
          [0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155]]]])
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))
    gradients: torch.Tensor = spatial_gradient(input, grads_mode)
    dx: torch.Tensor = gradients[:, :, 0]
    dy: torch.Tensor = gradients[:, :, 1]

    # compute the structure tensor M elements
    def g(x):
        return gaussian_blur2d(x, (3, 3), (1., 1.))

    dx2: torch.Tensor = g(dx ** 2)
    dy2: torch.Tensor = g(dy ** 2)
    dxy: torch.Tensor = g(dx * dy)

    det_m: torch.Tensor = dx2 * dy2 - dxy * dxy
    trace_m: torch.Tensor = dx2 + dy2

    e1: torch.Tensor = 0.5 * (trace_m + torch.sqrt((trace_m ** 2 - 4 * det_m).abs()))
    e2: torch.Tensor = 0.5 * (trace_m - torch.sqrt((trace_m ** 2 - 4 * det_m).abs()))

    scores: torch.Tensor = torch.min(e1, e2)
    return scores


def hessian_response(input: torch.Tensor,
                  grads_mode: str = 'sobel') -> torch.Tensor:
    r"""Computes the absolute of determinant of the Hessian matrix. Function does not do
     any normalization or nms.

     The response map is computed according the following formulation:

    .. math::
        R = det(H)
    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I_{xx} & I_{xy} \\
            I_{xy} & I_yy \\
        \end{bmatrix}


     Args:
         grads_mode (string): can be 'sobel' for standalone use or 'diff' for use on Gaussian pyramid

     Input:
         torch.Tensor: 4d tensor

     Return:
         torch.Tensor: the response map per channel.

     Shape:
       - Input: :math:`(B, C, H, W)`
       - Output: :math:`(B, C, H, W)`

     Examples:
         >>> input = torch.tensor([[[
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
         ]]])  # 1x1x7x7
         >>> # compute the response map
         >>> output = hessian_response(input)
         tensor([[[[0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155],
           [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
           [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
           [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
           [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
           [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
           [0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155]]]])
     """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))
    gradients: torch.Tensor = spatial_gradient(input, grads_mode, 2)
    dxx: torch.Tensor = gradients[:, :, 0]
    dxy: torch.Tensor = gradients[:, :, 1]
    dyy: torch.Tensor = gradients[:, :, 2]

    scores: torch.Tensor = torch.abs(dxx * dyy - dxy ** 2)

    return scores