import torch
import torch.nn as nn
import picasso.point.utils as pointUtil
from picasso.point.layers import PCloudConv3d
from picasso.mesh.layers import V2VConv3d, F2VConv3d, F2FConv3d, PerItemConv3d


class DualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, spharm_L, radius,
                 max_iter=2, max_nn=None, with_bn=True, bn_momentum=0.1):
        super(DualBlock, self).__init__()
        self.radius = radius
        self.max_nn = max_nn
        self.maxIter = max_iter
        self.L = spharm_L

        factor = 4
        self.DualConvGroup = []
        _in_channels_ = in_channels
        for n in range(0,self.maxIter):
            MeshConv1 = V2VConv3d(_in_channels_, [factor*growth_rate,factor*growth_rate],
                                  self.L, dual_depth_multiplier=[1,1],
                                  with_bn=with_bn, bn_momentum=bn_momentum)
            MeshConv2 = V2VConv3d(factor*growth_rate, [factor*growth_rate,2*growth_rate],
                                  self.L, dual_depth_multiplier=[1,1],
                                  with_bn=with_bn, bn_momentum=bn_momentum)
            Dense = PerItemConv3d(_in_channels_, factor*growth_rate,
                                  with_bn=with_bn, bn_momentum=bn_momentum)
            PCloudConv1 = PCloudConv3d(factor*growth_rate, 2*growth_rate,
                                       (self.L+1)**2+1, with_bn=with_bn, bn_momentum=bn_momentum)
            self.DualConvGroup.append(MeshConv1)
            self.DualConvGroup.append(MeshConv2)
            self.DualConvGroup.append(Dense)
            self.DualConvGroup.append(PCloudConv1)
            _in_channels_ += (2*growth_rate*2)

        self.DualConvGroup = nn.ModuleList(self.DualConvGroup)
        self.Transit = PerItemConv3d(_in_channels_, out_channels, with_bn=with_bn, bn_momentum=bn_momentum)

    def forward(self, inputs, vertex, face, full_nf_count, full_vt_map, filt_coeff, nv_in):
        # build graph for point-based convolution
        xyz = vertex[:,:3]
        graph_tuple = self._build_point_graph_(xyz, nv_in, nn_uplimit=self.max_nn)

        # perform the rest dualConv
        for n in range(0,self.maxIter):
            Mnet = self.DualConvGroup[4*n](inputs, face, full_nf_count, full_vt_map, filt_coeff) # V2V
            Mnet = self.DualConvGroup[4*n+1](Mnet, face, full_nf_count, full_vt_map, filt_coeff) # V2V
            Pnet = self.DualConvGroup[4*n+2](inputs)
            Pnet = self.DualConvGroup[4*n+3](Pnet, *graph_tuple)  # Point
            inputs = torch.cat([inputs,Mnet,Pnet],dim=-1)

        outputs = self.Transit(inputs)
        return outputs

    def _build_point_graph_(self, xyz, nv_in, nn_uplimit=None):
        nn_cnt, nn_idx, nn_dst, \
        filt_idx, filt_coeff = pointUtil.build_sphere_harmonics(xyz_data=xyz, xyz_query=xyz,
                                                                nv_data=nv_in, nv_query=nv_in,
                                                                radius=self.radius, L=self.L,
                                                                nn_uplimit=nn_uplimit)
        return nn_cnt, nn_idx, filt_idx, filt_coeff


class EncoderMeshBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, spharm_L,
                 max_iter=2, with_bn=True, bn_momentum=0.1):
        super(EncoderMeshBlock, self).__init__()
        self.maxIter = max_iter

        factor = 4
        self.MeshConvGroup = []
        _in_channels_ = in_channels
        for n in range(0,self.maxIter):
            MeshConv1 = V2VConv3d(_in_channels_, [factor*growth_rate,factor*growth_rate],
                                  spharm_L, dual_depth_multiplier=[1,1],
                                  with_bn=with_bn, bn_momentum=bn_momentum)
            MeshConv2 = V2VConv3d(factor*growth_rate, [factor*growth_rate,growth_rate],
                                  spharm_L, dual_depth_multiplier=[1,1],
                                  with_bn=with_bn, bn_momentum=bn_momentum)
            self.MeshConvGroup.append(MeshConv1)
            self.MeshConvGroup.append(MeshConv2)
            _in_channels_ += growth_rate

        self.MeshConvGroup = nn.ModuleList(self.MeshConvGroup)
        self.Transit = PerItemConv3d(_in_channels_, out_channels, with_bn=with_bn, bn_momentum=bn_momentum)

    def forward(self, inputs, vertex, face, full_nf_count, full_vt_map, filt_coeff, nv_in):
        for n in range(0, self.maxIter):
            net = self.MeshConvGroup[2*n](inputs, face, full_nf_count, full_vt_map, filt_coeff)
            net = self.MeshConvGroup[2*n+1](net, face, full_nf_count, full_vt_map, filt_coeff)
            inputs = torch.cat([inputs, net], dim=-1)

        outputs = self.Transit(inputs)
        return outputs


class FirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spharm_L, with_bn=True, bn_momentum=0.1):
        super(FirstBlock, self).__init__()
        self.MLP0 = PerItemConv3d(in_channels, out_channels, with_bn=with_bn,
                                  bn_momentum=bn_momentum)
        self.F2V0 = F2VConv3d(out_channels, out_channels, spharm_L,
                              depth_multiplier=2, with_bn=with_bn,
                              bn_momentum=bn_momentum)

    def forward(self, inputs, face, full_nf_count, full_vt_map, filt_coeff):
        net = self.MLP0(inputs)
        net = self.F2V0(net, face, full_nf_count, full_vt_map, filt_coeff)
        return net


class FirstBlockTexture(nn.Module):
    def __init__(self, in_channels, out_channels, spharm_L, with_bn=True, bn_momentum=0.1):
        super(FirstBlockTexture, self).__init__()
        GeometryInChannels, TextureInChannels = in_channels
        self.F2F0 = F2FConv3d(TextureInChannels, out_channels, spharm_L,
                              with_bn=with_bn, bn_momentum=bn_momentum)
        self.MLP0 = PerItemConv3d(GeometryInChannels, out_channels, with_bn=with_bn,
                                  bn_momentum=bn_momentum)
        self.F2V0 = F2VConv3d(out_channels, out_channels, spharm_L,
                              depth_multiplier=2, with_bn=with_bn, bn_momentum=bn_momentum)

    def __call__(self, facet_geometrics, facet_textures, bary_coeff, num_texture,
                 face, full_nf_count, full_vt_map, filt_coeff):
        Tnet = self.F2F0(facet_textures, bary_coeff, num_texture)
        Gnet = self.MLP0(facet_geometrics)
        net = Tnet + Gnet
        net = self.F2V0(net, face, full_nf_count, full_vt_map, filt_coeff)
        return net
