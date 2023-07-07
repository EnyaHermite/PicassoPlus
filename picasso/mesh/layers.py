import torch
import torch.nn as nn
from typing import List
from torch_scatter import scatter_mean, scatter_max
from .pi_modules import *
import numpy as np
from picasso.math_utils import torch_sph_harm_reorder


def _filter_variable_(shape, stddev=1.0, use_xavier=True):
    weight = nn.Parameter(torch.empty(shape))
    if use_xavier:
        return nn.init.xavier_normal_(weight)
    else:
        return nn.init.trunc_normal_(weight,mean=0,std=stddev)


class F2FConv3d(nn.Module):
    """ convolution on the facet textures
    Args:
       in_channels:  number of input  channels
       out_channels: number of output channels
       use_xavier: bool, use xavier_initializer if true
       stddev: float, stddev for truncated_normal init
    Returns:
       Variable tensor
    """
    def __init__(self, in_channels, out_channels, spharm_L, use_xavier=True, stddev=1e-3,
                 activation_fn=nn.ReLU(), with_bn=True, bn_momentum=0.1):
        super(F2FConv3d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L = spharm_L
        self.num_basis = (spharm_L+1)**2
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.with_bn = with_bn
        self.activation_fn = activation_fn
        if with_bn:
            self.BatchNorm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=bn_momentum)
        self.reset_parameters()

    def reset_parameters(self):
        # depthwise kernel shape
        f2f_kernel_shape = [self.out_channels, self.in_channels, self.num_basis]
        self.weights = _filter_variable_(f2f_kernel_shape, stddev=self.stddev, use_xavier=self.use_xavier)

        # biases term
        self.biases = nn.Parameter(torch.zeros(1,self.out_channels,requires_grad=True))

        assert(self.weights.requires_grad and self.biases.requires_grad)

    def forward(self, input_texture, bary_coeff, num_texture):
        assert(self.in_channels==input_texture.shape[-1])
        assert(self.num_basis==bary_coeff.shape[-1])
        # print(num_texture[:20], num_texture.shape)
        num_texture = torch.cumsum(num_texture, dim=0, dtype=torch.int)  # cumulative sum required
        outputs = conv3d.facet2facet(input_texture, self.weights, bary_coeff, num_texture)
        outputs = outputs + self.biases
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs)
        return outputs


class V2FConv3d(nn.Module):
    """ feature propagation from vertex to facet
    Args:
        in_channels:  number of input  channels
        out_channels: number of output channels
        depth_multiplier: depth multiplier in the separable convolution
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
    Returns:
        Variable tensor
    """
    def __init__(self, in_channels, out_channels, spharm_L, depth_multiplier=1,
                 use_xavier=True, stddev=1e-3, activation_fn=nn.ReLU(),
                 with_bn=True, bn_momentum=0.1):
        super(V2FConv3d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L = spharm_L
        self.num_basis = (self.L+1)**2
        self.multiplier = depth_multiplier
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.with_bn = with_bn
        self.activation_fn = activation_fn
        if self.with_bn:
            self.BatchNorm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=bn_momentum)
        self.reset_parameters()

    def reset_parameters(self):
        # depthwise kernel shape
        v2f_kernel_shape = [self.num_basis, self.in_channels*self.multiplier]
        self.spatial_weights = _filter_variable_(v2f_kernel_shape, stddev=self.stddev, use_xavier=self.use_xavier)

        # pointwise kernel shape
        kernel_shape = [self.in_channels*self.multiplier, self.out_channels]
        self.depth_weights = _filter_variable_(kernel_shape, stddev=self.stddev, use_xavier=self.use_xavier)

        # biases term
        self.biases = nn.Parameter(torch.zeros(1,self.out_channels,requires_grad=True))

        assert(self.spatial_weights.requires_grad and self.depth_weights.requires_grad and self.biases.requires_grad)

    def forward(self, inputs, face):
        spharm_values = SphHarmCoeff(self.L)(torch.eye(3).to(inputs.get_device()))
        spatial_filter = torch.matmul(spharm_values, self.spatial_weights)
        spatial_filter = spatial_filter.reshape([3, self.in_channels, self.multiplier])
        outputs = conv3d.vertex2facet(inputs, spatial_filter, face)
        outputs = torch.matmul(outputs, self.depth_weights)
        outputs = outputs + self.biases
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs)
        return outputs


class F2VConv3d(nn.Module):
    """ convolution to learn vertex features from adjacent facets
     Args:
         in_channels:  number of input  channels
         out_channels: number of output channels
         num_basis:  number of components in the mixture model
         depth_multiplier: depth multiplier in the separable convolution
         use_xavier: bool, use xavier_initializer if true
         stddev: float, stddev for truncated_normal init
     Returns:
         Variable tensor
     """
    def __init__(self, in_channels, out_channels, spharm_L, depth_multiplier=1,
                 use_xavier=True, stddev=1e-3, activation_fn=nn.ReLU(),
                 with_bn=True, bn_momentum=0.1):
        super(F2VConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L = spharm_L
        self.num_basis = (spharm_L+1)**2
        self.multiplier = depth_multiplier
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.with_bn = with_bn
        self.activation_fn = activation_fn
        if with_bn:
            self.BatchNorm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=bn_momentum)
        self.reset_parameters()

    def reset_parameters(self):
        # depthwise kernel shape
        f2v_kernel_shape = [self.num_basis, self.in_channels, self.multiplier]
        self.spatial_weights = _filter_variable_(f2v_kernel_shape, stddev=self.stddev, use_xavier=self.use_xavier)

        # pointwise kernel shape
        kernel_shape = [self.in_channels*self.multiplier, self.out_channels]
        self.depth_weights = _filter_variable_(kernel_shape, stddev=self.stddev, use_xavier=self.use_xavier)

        # biases term
        self.biases = nn.Parameter(torch.zeros(1,self.out_channels,requires_grad=True))

        assert(self.spatial_weights.requires_grad and self.depth_weights.requires_grad and self.biases.requires_grad)

    def forward(self, inputs, face, nf_count, vt_map, filt_coeff):
        outputs = conv3d.facet2vertex(inputs, self.spatial_weights,
                                      filt_coeff, face, nf_count, vt_map)
        outputs = torch.matmul(outputs, self.depth_weights)
        outputs = outputs + self.biases
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs)
        return outputs


class V2VConv3d(nn.Module):
    # out_channels and depth_multiplier are vectors of 2 values.
    def __init__(self, in_channels, dual_out_channels, spharm_L,
                 dual_depth_multiplier:List[int], use_xavier=True, stddev=1e-3,
                 activation_fn=nn.ReLU(), with_bn=True, bn_momentum=0.1):
        super(V2VConv3d, self).__init__()
        self.V2F_conv3d = V2FConv3d(in_channels, dual_out_channels[0], spharm_L,
                                    dual_depth_multiplier[0], use_xavier,
                                    stddev, activation_fn, with_bn, bn_momentum)

        self.F2V_conv3d = F2VConv3d(dual_out_channels[0], dual_out_channels[1],
                                    spharm_L, dual_depth_multiplier[1], use_xavier,
                                    stddev, activation_fn, with_bn, bn_momentum)

    def forward(self, inputs, face, nf_count, vt_map, filt_coeff):
        outputs = self.V2F_conv3d(inputs, face)
        outputs = self.F2V_conv3d(outputs, face, nf_count, vt_map, filt_coeff)
        return outputs


class PerItemConv3d(nn.Module):
    """ elementwise convolutionwith non-linear operation.
        e.g. per-pixel, per-point, per-voxel, per-facet, etc.
    Args:
        in_channels:  number of input  channels
        out_channels: number of output channels
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
    Returns:
      Variable tensor
    """
    def __init__(self, in_channels, out_channels, use_xavier=True, stddev=1e-3,
                 activation_fn=nn.ReLU(), with_bn=True, bn_momentum=0.1):
        super(PerItemConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.with_bn = with_bn
        self.activation_fn = activation_fn
        if with_bn:
            self.BatchNorm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=bn_momentum)
        self.reset_parameters()

    def reset_parameters(self):
        # pointwise kernel shape
        kernel_shape = [self.in_channels, self.out_channels]
        self.weights = _filter_variable_(kernel_shape, stddev=self.stddev, use_xavier=self.use_xavier)

        # biases term
        self.biases = nn.Parameter(torch.zeros(1, self.out_channels, requires_grad=True))

        assert(self.weights.requires_grad and self.biases.requires_grad)

    def forward(self, inputs):
        outputs = torch.matmul(inputs, self.weights)
        outputs = outputs + self.biases
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs)
        return outputs


class GlobalPool3d(nn.Module):
    """ 3D Global Mesh pooling.
    """
    def __init__(self, method='avg'):
        super(GlobalPool3d, self).__init__()
        self.method = method

    def forward(self, inputs, nv_in):
        batch_size = nv_in.size()[0]
        sample_idx = torch.repeat_interleave(torch.arange(batch_size).to(nv_in.get_device()), nv_in)

        if self.method=='avg':
            outputs = scatter_mean(inputs, sample_idx, dim=0)
        elif self.method=='max':
            outputs = scatter_max(inputs, sample_idx, dim=0)
        else:
            raise ValueError("Unknow pooling method %s."%self.method)

        return outputs


class Pool3d(nn.Module):
    """ 3D Mesh pooling.
    Args:
        pooling method, default to 'max'
    Returns:
        Variable tensor
    """
    def __init__(self, method='max'):
        super(Pool3d, self).__init__()
        self.method = method

    def forward(self, inputs, vt_replace, vt_map, vt_out):
        if self.method == 'max':
            outputs = pool3d.maxPool(inputs, vt_replace, vt_map, vt_out)
        elif self.method == 'avg':
            outputs = pool3d.avgPool(inputs, vt_replace, vt_map, vt_out)
        else:
            raise ValueError("Unknow pooling method %s."%self.method)

        return outputs


class Unpool3d(nn.Module):
    """ 3D Mesh unpooling
    Returns:
        Variable tensor
    """
    def __init__(self):
        super(Unpool3d, self).__init__()

    @staticmethod
    def forward(inputs, vt_replace, vt_map):
        outputs = unpool3d.interpolate(inputs, vt_replace, vt_map)
        return outputs


class BuildF2VCoeff(nn.Module):
    ''' Compute fuzzy coefficients for the facet2vertex convolution
    Returns:
        fuzzy filter coefficients
    '''
    def __init__(self, kernel_size, tau=0.1, dim=3, stddev=1.0):
        super(BuildF2VCoeff, self).__init__()
        self.kernel_size = kernel_size
        self.tau = tau # similar to the temperature parameter in ContrastLoss
        self.dim = dim
        self.stddev = stddev
        self.reset_parameters()

    def reset_parameters(self):
        shape = [self.kernel_size, self.dim]
        self.centers = _filter_variable_(shape, stddev=self.stddev, use_xavier=False)
        assert(self.centers.requires_grad)

    def forward(self, face_normals):
        # Note: face normal features are already unit vectors
        # normalize clustering center of normals to unit vectors
        l2norm = torch.sqrt(torch.sum(self.centers**2, dim=-1, keepdim=True))
        norm_center = self.centers/l2norm  # unit length normal vectors
        norm_center = norm_center.to(face_normals.get_device())

        cos_theta = torch.matmul(face_normals, norm_center.t())  # cos(theta): range [-1,1]
        zeta = cos_theta/self.tau
        coeff = torch.exp(zeta)
        coeff = coeff/torch.sum(coeff, dim=-1, keepdim=True) # softmax compute
        return coeff


class SphHarmCoeff(nn.Module):
    ''' Compute fuzzy coefficients for the facet2vertex convolution.
    We use spherical harmonics to achieve this.
    Returns:
        fuzzy filter coefficients
    '''
    def __init__(self, l):
        super(SphHarmCoeff, self).__init__()
        self.l = l

    def proj_to_sphere(self, xyz):
        # due to the uncertainty of normal directions,
        # we only use the top hemisphere in the fuzzy modelling
        l2norm = torch.sqrt(torch.sum(xyz**2, dim=-1, keepdim=True))
        xyz = xyz/(l2norm+1e-10)
        return xyz

    def __call__(self, cart_coords):
        cart_coords = self.proj_to_sphere(cart_coords)
        phi = torch.atan2(cart_coords[:,1], cart_coords[:,0])   # azimuth
        theta = torch.atan2(cart_coords[:,2], torch.linalg.norm(cart_coords[:,:2], dim=-1))    # polar
        phi[phi<0] += 2*np.pi
        theta = np.pi/2 - theta

        coeff = torch_sph_harm_reorder(self.l, phi, theta)
        return coeff





