import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import numpy as np
from math import log, floor, ceil
from torch_scatter import scatter_mean

from .pi_modules.buildkernel import SPH3Dkernel_, fuzzySPH3Dkernel_
from .pi_modules.nnquery import range_search_, cube_search_, knn3_search_
from .pi_modules.sample import fps_
from picasso.point.layers import BuildSpharmCoeff


def range_search(xyz_data, xyz_query, nv_data, nv_query, radius, max_nn):
    cnt_info, nn_cnt, nn_idx, nn_dst = range_search_(xyz_data, xyz_query, nv_data, nv_query,
                                                     radius=radius, nnsample=max_nn)
    return nn_cnt, nn_idx, nn_dst


def cube_search(xyz_data, xyz_query, nv_data, nv_query, length, nnsample=None, gridsize=4):
    cnt_info, nn_cnt, nn_idx, nn_dst = cube_search_(xyz_data, xyz_query, nv_data, nv_query,
                                                    length=length, nnsample=nnsample, gridsize=gridsize)
    return nn_cnt, nn_idx, nn_dst


def knn3_search(xyz_data, xyz_query, nv_data, nv_query):
    nn_idx, nn_dst = knn3_search_(xyz_data, xyz_query, nv_data, nv_query)
    return nn_idx, nn_dst


def compute_unpool_weight(nn_dist):
    # default inverse distance based weights
    weight = 1/(nn_dist+np.finfo(float).eps)
    weight = weight/torch.sum(weight, dim=-1, keepdim=True)[0]
    return weight


def fps(xyz_in, nv_xyz, num_sample):
    sample_index = fps_(xyz_in, nv_xyz, num_sample) # single dimension
    return sample_index


def build_range_graph(xyz_data, xyz_query, nv_data, nv_query,
                      radius, nn_uplimit=None, kernel_shape=None, fuzzy=True):
    if fuzzy:
        nn_cnt, nn_idx, nn_dst = range_search(xyz_data, xyz_query, nv_data, nv_query,
                                              radius=radius, max_nn=nn_uplimit)
        filt_idx, filt_coeff = fuzzySPH3Dkernel_(xyz_data, xyz_query, nn_idx, nn_dst, radius,
                                                 kernel=kernel_shape)
        return nn_cnt, nn_idx, nn_dst, filt_idx, filt_coeff
    else:
        nn_cnt, nn_idx, nn_dst = range_search(xyz_data, xyz_query, nv_data, nv_query,
                                              radius=radius, max_nn=nn_uplimit)
        filt_idx = SPH3Dkernel_(xyz_data, xyz_query, nn_idx, nn_dst,
                                radius, kernel=kernel_shape)
        filt_coeff = None
    return nn_cnt, nn_idx, nn_dst, filt_idx, filt_coeff


def build_cube_graph(xyz_data, xyz_query, nv_data, nv_query,
                     radius, nn_uplimit=None, grid_size=None):
    cnt_info, nn_cnt, nn_idx = cube_search(xyz_data, xyz_query, nv_data, nv_query, length=2*radius,
                                           nnsample=nn_uplimit, gridsize=grid_size)
    filt_idx = nn_idx[:, -1]
    nn_idx = nn_idx[:, :-1]
    nn_dst, filt_coeff = None, None
    return nn_cnt, nn_idx, nn_dst, filt_idx, filt_coeff


def build_sphere_harmonics(xyz_data, xyz_query, nv_data, nv_query,
                           radius, L=2, nn_uplimit=None):
    kernel_size = (L+1)**2+1
    nn_cnt, nn_idx, nn_dst = range_search(xyz_data, xyz_query, nv_data, nv_query,
                                          radius=radius, max_nn=nn_uplimit)
    gamma = nn_dst[:,None]/radius
    gamma = torch.clip(gamma, 0, 1)
    filt_coeff = BuildSpharmCoeff(L)(xyz_data, xyz_query, nn_idx)
    filt_coeff = torch.concat([1-gamma, gamma*filt_coeff], dim=-1)
    filt_idx = torch.arange(kernel_size, dtype=torch.int).to(filt_coeff.get_device())
    filt_idx = torch.tile(filt_idx, [filt_coeff.shape[0],1])
    return nn_cnt, nn_idx, nn_dst, filt_idx, filt_coeff



