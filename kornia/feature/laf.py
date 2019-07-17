from typing import Tuple

import kornia
import torch
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def angle_to_rotation_matrix(angle:torch.Tensor, do_deg2rad:bool = False) -> torch.Tensor:
    """
    Creates a rotation matrix out of angles
    Args:
        angle: (torch.Tensor): tensor of angles, any shape.
        do_deg2rad: (bool): if we should convert to radians first

    Returns:
        torch.Tensor: tensor of Nx2x2 rotation matrices.

    Shape:
        - Input: :math:`(N)`
        - Output: :math:`(N, 2, 2)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = kornia.angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    if do_deg2rad:
        ang = kornia.deg2rad(angle)
    else:
        ang =  angle
    n_dims = len(ang.size())
    cos_a = torch.cos(ang).unsqueeze(-1).unsqueeze(-1)
    sin_a = torch.sin(ang).unsqueeze(-1).unsqueeze(-1)
    A1_ang = torch.cat([cos_a, sin_a], dim = n_dims+1)
    A2_ang = torch.cat([-sin_a, cos_a], dim = n_dims+1)
    return  torch.cat([A1_ang, A2_ang], dim = n_dims)

def get_laf_scale(A:torch.Tensor) -> torch.Tensor:
    """
    Returns a scale of the LAFs
    Args:
        LAF: (torch.Tensor): tensor [BxNx2x3] or [BxNx2x2].


    Returns:
        torch.Tensor: tensor  BxNx1x1 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Output: :math: `(B, N, 1, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = kornia.get_laf_scale(input)  # BxNx1x1
    """
    n_dims = len(A.size())
    eps = 1e-10;
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [BxNx2x3]. "
            "Got {}".format(A.size())
        )    
    return torch.sqrt(torch.abs(A[:,:,0:1,0:1]*A[:,:,1:2,1:2]
                              - A[:,:,1:2,0:1]*A[:,:,0:1,1:2] + eps))

 

def make_upright(A:torch.Tensor) -> torch.Tensor:
    """
    Rectifies the affine matrix, so that it becomes upright
    Args:
        A: (torch.Tensor): tensor of LAFs.

    Returns:
        torch.Tensor: tensor of same shape.

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = kornia.make_upright(input)  #  BxNx2x3
    """
    n_dims = len(A.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [BxNx2x3]. "
            "Got {}".format(A.size())
        )    
    det = get_laf_scale(A)
    eps = 1e-10;
    b2a2 = torch.sqrt(A[:,:,0:1,1:2]**2 + A[:,:,0:1,0:1]**2)
    A1_ell = torch.cat([(b2a2 / det).contiguous(), torch.zeros_like(det)], dim = 3)
    A2_ell = torch.cat([((A[:,:,1:2,1:2]*A[:,:,0:1,1:2]+A[:,:,1:2,0:1]*A[:,:,0:1,0:1])/(b2a2*det)),
                        (det / b2a2).contiguous()], dim = 3)
    return torch.cat([torch.cat([A1_ell, A2_ell], dim = 2), A[:,:,:,2:3]], dim=3)
    



def invSqrt2x2Sym(a:torch.Tensor, b:torch.Tensor, c:torch.Tensor) -> Tuple:
    """
    Aux. function. Caclulates inverse square root of batch of batch
    of symmetric 2x2 matrices,  given a11, a12 and a22 elements
    """
    eps = 1e-12
    mask = (b != 0).float()
    r1 = mask * (c - a) / (2. * b + eps)
    t1 = torch.sign(r1) / (torch.abs(r1) + torch.sqrt(1. + r1*r1));
    r = 1.0 / torch.sqrt( 1. + t1*t1)
    t = t1*r;
    r = r * mask + 1.0 * (1.0 - mask);
    t = t * mask;

    x = 1. / torch.sqrt( r*r*a - 2.0*r*t*b + t*t*c)
    z = 1. / torch.sqrt( t*t*a + 2.0*r*t*b + r*r*c)

    d = torch.sqrt( x * z)

    x = x / d
    z = z / d

    new_a = r*r*x + t*t*z
    new_b = -r*t*x + t*r*z
    new_c = t*t*x + r*r *z

    return new_a, new_b, new_c,


def ell2LAF(ells:torch.Tensor) -> torch.Tensor:
    """
    Converts ellipse regions to LAF format
    Args:
        A: (torch.Tensor): tensor of ellipses in Oxford format [x y a b c].

    Returns:
        LAF: (torch.Tensor) tensor of ellipses in LAF format.

    Shape:
        - Input: :math:`(B, N, 5)`
        - Output:  :math:`(B, N, 2, 3)`
    Example:
        >>> input = torch.ones(1, 10, 5)  # BxNx5
        >>> output = kornia.ell2LAF(input)  #  BxNx2x3
    """
    n_dims = len(ells.size())
    if (n_dims != 3):
        raise TypeError(
            "ellipse shape should be must be [BxNx5]. "
            "Got {}".format(ells.size()))
    B,N,dim = ells.size()
    if (dim != 5):
        raise TypeError(
            "ellipse shape should be must be [BxNx5]. "
            "Got {}".format(ells.size()))
    
    LAF = torch.zeros(B,N,2,3).to(ells.device)
    LAF[:,:,0,2] = ells[:,:,0]
    LAF[:,:,1,2] = ells[:,:,1]
    a = ells[:,:,2]
    b = ells[:,:,3]
    c = ells[:,:,4]
    sc = torch.sqrt(torch.sqrt(a*c - b*b + 1e-12))
    ia,ib,ic = invSqrt2x2Sym(a,b,c)  #because sqrtm returns ::-1, ::-1 matrix, don`t know why 
    A = torch.cat([torch.cat([(ia/sc).view(B,N,1,1), (ib/sc).view(B,N,1,1)], dim = 3),
                   torch.cat([(ib/sc).view(B,N,1,1), (ic/sc).view(B,N,1,1)], dim = 3)], dim = 2)
    sc = get_laf_scale(A)
    LAF[:,:,0:2,0:2] = make_upright(A / sc.repeat(1,1,2,2)) * sc.repeat(1,1,2,2)
    return LAF

def LAF2pts(LAF:torch.Tensor, n_pts:int = 50)-> torch.Tensor:
    """
    Converts LAFs to boundary points of the regions + center.
    Used for local features visualization, see visualize_LAF function
    Args:
        LAF: (torch.Tensor).
        n_pts: number of points to output

    Returns:
        pts: (torch.Tensor) tensor of boundary points

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, n_pts, 2)`
    """
    n_dims = len(LAF.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [Nx2x3] or [BxNx2x3]. "
            "Got {}".format(LAF.size())
        )
    B,N, _,_ = LAF.size()
    pts = torch.cat([torch.sin(torch.linspace(0, 2*math.pi, n_pts)).unsqueeze(-1),
                    torch.cos(torch.linspace(0, 2*math.pi, n_pts)).unsqueeze(-1),
                    torch.ones(n_pts,1)
                    ], dim=1)
    #Add origin to draw also the orientation
    pts = torch.cat([torch.tensor([0, 0, 1.]).view(1,3), pts], dim = 0).unsqueeze(0).expand(B*N,51,3)
    
    HLAF = torch.cat([LAF.view(-1,2,3),torch.tensor([0, 0, 1.]).view(1,1,3).expand(B*N,1,3)], dim =1)
    pts_h = torch.bmm(HLAF, pts.permute(0,2,1)).permute(0,2,1) 
    return kornia.convert_points_from_homogeneous(pts_h.view(B,N,n_pts+1, 3))

def visualize_LAF(img:torch.Tensor, LAF:torch.Tensor, 
                  img_idx:int = 0,
                  color:str = 'r'):
    """
    Draws affine regions (LAF)
    """
    pts = LAF2pts(LAF[img_idx:img_idx+1])[0]
    pts_np = pts.data.permute(1,0,2).cpu().numpy()
    plt.figure()
    plt.imshow(kornia.utils.tensor_to_image(img[img_idx]))
    plt.plot( pts_np[:,:,0], pts_np[:,:,1], color)
    plt.show()
    return 




def denormalize_LAF(LAF:torch.Tensor, images:torch.Tensor) -> torch.Tensor:
    """
    De-normalizes LAFs from scale to image scale
    B,N,H,W = images.size()
    MIN_SIZE = min(H,W)
    [a11 a21 x]
    [a21 a22 y] 
    becomes 
    [a11*MIN_SIZE a21*MIN_SIZE x*W]
    [a21*MIN_SIZE a22*MIN_SIZE y*H] 
    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in

    Returns:
        LAF: (torch.Tensor).

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    n_dims = len(LAF.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [Nx2x3] or [BxNx2x3]. "
            "Got {}".format(LAF.size())
        )
    n, ch, h, w = images.size()
    w = float(w)
    h = float(h)
    min_size = min(h,w)
    coef = torch.ones(1,1,2,3).float()  * min_size
    coef[0,0,0,2] = w
    coef[0,0,1,2] = h
    coef.to(LAF.device)
    return coef.expand_as(LAF) * LAF

def normalize_LAF(LAF:torch.Tensor, images:torch.Tensor) -> torch.Tensor:
    """
    Normalizes LAFs to [0,1] scale.
    B,N,H,W = images.size()
    MIN_SIZE = min(H,W)
    [a11 a21 x]
    [a21 a22 y] 
    becomes 
    [a11/MIN_SIZE a21/MIN_SIZE x/W]
    [a21/MIN_SIZE a22/MIN_SIZE y/H] 
    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in

    Returns:
        LAF: (torch.Tensor).

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    n_dims = len(LAF.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [BxNx2x3]. "
            "Got {}".format(LAF.size()))
    n, ch, h, w = images.size()
    w = float(w)
    h = float(h)
    min_size = min(h,w)
    coef = torch.ones(1,1,2,3).float()  / min_size
    coef[0,0,0,2] = 1.0 / w
    coef[0,0,1,2] = 1.0 / h
    coef.to(LAF.device)
    return coef.expand_as(LAF) * LAF

def generate_patch_grid_from_normalized_LAF(img:torch.Tensor,
                                            LAF:torch.Tensor,
                                            PS:int=32) -> torch.Tensor:
    """
    Helper function for affine grid generation
    """
    n_dims = len(LAF.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [BxNx2x3]. "
            "Got {}".format(LAF.size()))
    B,N,_,_ = LAF.size()
    num, ch, h, w = img.size()
    # norm, then renorm is needed for allowing detection on one resulution
    # and extraction at arbitrary other
    LAF_renorm = denormalize_LAF(LAF,img)
    
    grid = F.affine_grid(LAF_renorm.view(B*N,2,3), torch.Size((B*N,1,PS,PS)))
    grid[:,:,:,0] = 2.0 * grid[:,:,:,0] / float(w)  - 1.0
    grid[:,:,:,1] = 2.0 * grid[:,:,:,1] / float(h)  - 1.0     
    return grid


def extract_patches_simple(img:torch.Tensor,
                           LAF:torch.Tensor,
                           PS:int=32) -> torch.Tensor:
    """
    Extract patches defined by LAFs from image tensor.
    No smoothing applied, huge aliasing (better use extract_patches_from_pyramid)
    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in
        PS: (int) patch size, default = 32

    Returns:
        LAF: (torch.Tensor),  :math:`(B, N, CH, PS,PS)`
    """
    num, ch, h, w = img.size()
    B,N,_,_ = LAF.size()
    out = torch.zeros(B,N,ch,PS, PS)
    # for loop temporarily, to be refactored
    for i in range(B):
        grid = generate_patch_grid_from_normalized_LAF(img[i:i+1], LAF[i:i+1], PS)
        out[i,:,:,:,:] =  F.grid_sample(img.expand(grid.size(0), ch, h, w),  grid, padding_mode="border") 
    return out

def extract_patches_from_pyramid(img:torch.Tensor,
                           LAF:torch.Tensor,
                           PS:int=32) -> torch.Tensor:
    """
    Extract patches defined by LAFs from image tensor.
    Patches are extracted from appropriate pyramid level
    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in
        PS: (int) patch size, default = 32

    Returns:
        LAF: (torch.Tensor),  :math:`(B, N, CH, PS,PS)`
    """
    B,N,_,_ = LAF.size()
    num, ch, h, w = img.size()
    scale =  2.0 * get_laf_scale(denormalize_LAF(LAF,img)) / float(PS)
    pyr_idx = (scale.log2()+0.5).relu().long()
    cur_img = img
    out = torch.zeros(B,N,ch,PS, PS)
    cur_pyr_level = int(0)
    while min(cur_img.size(2),cur_img.size(3)) > PS:
        num, ch, h, w = cur_img.size()
        # for loop temporarily, to be refactored
        for i in range(B):
            scale_mask = (pyr_idx[i] == cur_pyr_level).squeeze()
            grid = generate_patch_grid_from_normalized_LAF(cur_img[i:i+1], LAF[i:i+1,scale_mask], PS)
            out[i,scale_mask,:,:,:] =  F.grid_sample(cur_img.expand(grid.size(0), ch, h, w),  grid, padding_mode="border")         
        cur_img = kornia.pyrdown(cur_img)
        cur_pyr_level+=1
    return out