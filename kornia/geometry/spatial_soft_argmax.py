import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry import dsnt as dsnt
from kornia.utils import create_meshgrid
from kornia.geometry import normalize_pixel_coordinates
from typing import Tuple, Union


def _get_window_grid_kernel(h: int, w: int) -> torch.Tensor:
    '''Helper function, which generates a kernel to
    with window coordinates, residual to window center
    Args:
         h (int): kernel height
         w (int): kernel width
    Returns:
        conv_kernel (torch.Tensor) [2x1xhxw]
    '''
    window_grid2d = create_meshgrid(h, w, False)
    window_grid2d = normalize_pixel_coordinates(window_grid2d, h, w)
    conv_kernel = window_grid2d.permute(3, 0, 1, 2)
    return conv_kernel


def _get_center_kernel(h: int, w: int) -> torch.Tensor:
    '''Helper function, which generates a kernel to
    return center coordinates, when applied with F.conv2d to 2d coordinates grid
    Args:
         h (int): kernel height
         w (int): kernel width
    Returns:
        conv_kernel (torch.Tensor) [2x2xhxw]
    '''
    center_kernel = torch.zeros(2, 2, h, w)
    center_kernel[0, 0, h // 2, w // 2] = 1.0
    center_kernel[1, 1, h // 2, w // 2] = 1.0
    return center_kernel


class ConvSoftArgmax2d(nn.Module):
    def __init__(self,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1),
                 temperature: torch.Tensor = torch.tensor(1.0),
                 normalized_coordinates: bool = True,
                 eps: float = 1e-8,
                 output_nmsed_value: bool = True) -> None:
        super(ConvSoftArgmax2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_nmsed_value = output_nmsed_value
        return

    def forward(self, x: torch.Tensor):  # type: ignore
        return conv_soft_argmax2d(x,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.temperature,
                                  self.normalized_coordinates,
                                  self.eps,
                                  self.output_nmsed_value)


def conv_soft_argmax2d(input: torch.Tensor,
                       kernel_size: Tuple[int, int] = (3, 3),
                       stride: Tuple[int, int] = (1, 1),
                       padding: Tuple[int, int] = (1, 1),
                       temperature: torch.Tensor = torch.tensor(1.0),
                       normalized_coordinates: bool = True,
                       eps: float = 1e-8,
                       output_nmsed_value: bool = True) -> Union[torch.Tensor,
                                                                 Tuple[torch.Tensor, torch.Tensor]]:
    """
    Function that computes the convolutional spatial Soft-Argmax 2D over the windows
    of a given input heatmap. Function has two outputs: argmax coordinates and the softmaxpooled heatmap values
    themselves.

    On each window, the function computed is:

    .. math::
        ij(X) = \frac{\sum_{(i,j) * exp(x / T)  \in X}} {\sum_{exp(x / T)  \in X}}
        val(X) = \frac{\sum_{x * exp(x / T)  \in X}} {\sum_{exp(x / T)  \in X}}

    - where T is temperature.

    Args:
        kernel_size (Tuple([int,int])): the size of the window
        stride  (Tuple([int,int])): the stride of the window.
        padding (Tuple([int,int])): input zero padding
        temperature (torch.Tensor): factor to apply to input. Default is 1.
        normalized_coordinates (bool): whether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.
        eps (float): small value to avoid zero division. Default is 1e-8.
        output_nmsed_value(bool): if True, val is outputed, if False, only ij


    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: math:`(N, C, 2, H_{out}, W_{out})`, :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
              (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
              (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> input = torch.randn(20, 16, 50, 32)
        >>> nms_coords, nms_val = conv_soft_argmax2d(input, (3,3), (2,2), (1,1))

    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))
    b, c, h, w = input.shape
    input = input.view(b * c, 1, h, w)

    center_kernel = _get_center_kernel(kernel_size[0], kernel_size[1])
    window_kernel = _get_window_grid_kernel(kernel_size[0], kernel_size[1])

    x_exp = (input / torch.clamp(temperature, eps)).exp()

    # F.avg_pool2d(.., divisor_override = 1.0) - proper way for sum pool in PyTorch 1.2.
    # Not available yet in version 1.0, so let's do manually
    pool_coef: float = float(kernel_size[0] * kernel_size[1])

    # softmax denominator
    den = pool_coef * F.avg_pool2d(x_exp,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding) + 1e-12

    resp_maxima = pool_coef * F.avg_pool2d(x_exp * input,
                                           kernel_size,
                                           stride=stride,
                                           padding=padding) / den

    # We need to output also coordinates
    # Pooled window center coordinates
    grid_global: torch.Tensor = create_meshgrid(h, w, False).permute(0, 3, 1, 2)
    grid_global = grid_global.to(input.device).to(input.dtype)
    grid_global_pooled = F.conv2d(grid_global,
                                  center_kernel,
                                  stride=stride,
                                  padding=padding)

    # Coordinates of maxima residual to window center
    # prepare kernel
    coords_max: torch.Tensor = F.conv2d(x_exp,
                                        window_kernel,
                                        stride=stride,
                                        padding=padding)

    coords_max = coords_max / den.expand_as(coords_max)
    coords_max = coords_max + grid_global_pooled.expand_as(coords_max)
    # [:,:, 0, ...] is x
    # [:,:, 1, ...] is y

    if normalized_coordinates:
        coords_max = normalize_pixel_coordinates(coords_max.permute(0, 2, 3, 1), h, w)
        coords_max = coords_max.permute(0, 3, 1, 2)
    if output_nmsed_value:
        return coords_max, resp_maxima
    return coords_max


def spatial_soft_argmax2d(
        input: torch.Tensor,
        temperature: torch.Tensor = torch.tensor(1.0),
        normalized_coordinates: bool = True,
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        temperature (torch.Tensor): factor to apply to input. Default is 1.
        normalized_coordinates (bool): whether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.
        eps (float): small value to avoid zero division. Default is 1e-8.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 10., 0.],
            [0., 0., 0.]]]])
        >>> coords = kornia.spatial_soft_argmax2d(input, False)
        tensor([[[1.0000, 1.0000]]])
    """
    input_soft: torch.Tensor = dsnt.spatial_softmax_2d(input, temperature)
    output: torch.Tensor = dsnt.spatial_softargmax_2d(input_soft,
                                                      normalized_coordinates)
    return output


class SpatialSoftArgmax2d(nn.Module):
    r"""Function that computes the Spatial Soft-Argmax 2D of a given heatmap.

    See :class:`~kornia.contrib.spatial_soft_argmax2d` for details.
    """

    def __init__(self,
                 temperature: torch.Tensor = torch.tensor(1.0),
                 normalized_coordinates: bool = True,
                 eps: float = 1e-8) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.temperature: torch.Tensor = temperature
        self.normalized_coordinates: bool = normalized_coordinates
        self.eps: float = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return spatial_soft_argmax2d(input, self.temperature,
                                     self.normalized_coordinates, self.eps)
