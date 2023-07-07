import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from .pi_modules import *
import numpy as np
from picasso.math_utils import torch_sph_harm_reorder
from scipy.special import sph_harm


def _filter_variable_(shape, stddev=1.0, use_xavier=True):
    weight = nn.Parameter(torch.empty(shape))
    if use_xavier:
        return nn.init.xavier_normal_(weight)
    else:
        return nn.init.trunc_normal_(weight,mean=0,std=stddev)


class PCloudConv3d(nn.Module):
    """ 3D separable convolution with non-linear operation.
    Args:
        in_channels:  number of input  channels
        out_channels: number of output channels
        depth_multiplier: depth multiplier in the separable convolution
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
    Returns:
      Variable tensor
    """

    def __init__(self, in_channels, out_channels, num_basis, depth_multiplier=1,
                 use_xavier=True, stddev=1e-3, activation_fn=nn.ReLU(),
                 with_bn=True, bn_momentum=0.1):
        super(PCloudConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_basis = num_basis
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
        spatial_kernel_shape = [self.num_basis, self.in_channels, self.multiplier]
        self.spatial_weights = _filter_variable_(spatial_kernel_shape, use_xavier=self.use_xavier)

        # pointwise kernel shape
        kernel_shape = [self.in_channels*self.multiplier, self.out_channels]
        self.depth_weights = _filter_variable_(kernel_shape, use_xavier=self.use_xavier)

        # biases term
        self.biases = nn.Parameter(torch.zeros(1, self.out_channels, requires_grad=True))

        assert (self.spatial_weights.requires_grad and self.depth_weights.requires_grad and self.biases.requires_grad)

    def forward(self, inputs, nn_count, nn_index, filt_index, filt_coeff=None):
        if filt_coeff is None:
            outputs = conv3d.conv3d(inputs, self.spatial_weights,
                                    nn_count, nn_index, filt_index)
        else:
            outputs = conv3d.fuzzyconv3d(inputs, self.spatial_weights, nn_count,
                                         nn_index, filt_index, filt_coeff)
        outputs = torch.matmul(outputs, self.depth_weights)
        outputs = outputs + self.biases
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs)

        return outputs


class PerItemConv3d(nn.Module):
    """ elementwise convolution with non-linear operation.
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
                 activation_fn=torch.nn.ReLU(), with_bn=True, bn_momentum=0.1):
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
        self.weights = _filter_variable_(kernel_shape, use_xavier=self.use_xavier)

        # biases term
        self.biases = nn.Parameter(torch.zeros(1,self.out_channels,requires_grad=True))

        assert(self.weights.requires_grad and self.biases.requires_grad)

    def forward(self, inputs):
        outputs = torch.matmul(inputs, self.weights)
        outputs = outputs + self.biases
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs)
        return outputs


class Pool3d(nn.Module):
    """ 3D point cloud pooling.
    Args:
        pooling method, default to 'max'
    Returns:
        Variable tensor of shape [concat_Mp,C]
    """
    def __init__(self, method='max'):
        super(Pool3d, self).__init__()
        self.method = method

    def __call__(self, inputs, nn_count, nn_index):
        if self.method == 'max':
            outputs = pool3d.maxPool(inputs, nn_count, nn_index)
        elif self.method == 'avg':
            outputs = pool3d.avgPool(inputs, nn_count, nn_index)
        else:
            raise ValueError("Unknow pooling method %s."%self.method)

        return outputs


class Unpool3d(nn.Module):
    """ 3D unpooling
    Returns:
        outputs: float32 tensor of shape [concat_Np,C], unpooled features
    """
    def __init__(self):
        super(Unpool3d, self).__init__()

    def __call__(self, inputs, weight, nn_index):
        outputs = unpool3d.interpolate(inputs, weight, nn_index)
        return outputs


class BuildSpharmCoeff(nn.Module):
    ''' Compute fuzzy coefficients for the facet2vertex convolution.
    We use spherical harmonics to achieve this.
    Returns:
        fuzzy filter coefficients
    '''
    def __init__(self, l):
        super(BuildSpharmCoeff, self).__init__()
        self.l = l

    @staticmethod
    def proj_to_sphere(xyz):
        l2norm = torch.sqrt(torch.sum(xyz**2, dim=-1, keepdim=True))
        xyz = xyz/(l2norm+1e-10)
        return xyz

    def __call__(self, xyz_data, xyz_query, nn_idx):
        nn_idx = nn_idx.to(torch.long)
        delta = xyz_data[nn_idx[:,1]] - xyz_query[nn_idx[:,0]]
        delta = self.proj_to_sphere(delta)

        phi = torch.atan2(delta[:,1], delta[:,0])
        theta = torch.atan2(delta[:,2], torch.linalg.norm(delta[:,:2], dim=-1))
        phi[phi<0] += 2*np.pi
        theta = np.pi/2 - theta

        coeff = torch_sph_harm_reorder(self.l, phi, theta)
        return coeff

