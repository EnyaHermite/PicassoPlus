import numpy as np
import torch
import torch.nn as nn
import picasso.mesh.utils as meshUtil
from picasso.mesh.layers import PerItemConv3d, SphHarmCoeff
from picasso.mesh.layers import Pool3d as mesh_pool
from picasso.mesh.layers import GlobalPool3d as global_mesh_pool
from picasso.mesh.layers import Unpool3d as mesh_unpool
from .pi2_blocks import DualBlock, EncoderMeshBlock, FirstBlock


class PicassoNetII(nn.Module):
    def __init__(self, num_class, stride=None, spharm_L=3, use_height=False):
        super(PicassoNetII,self).__init__()

        if stride is None:
            self.stride = [1.5, 1.5, 1.5, 1.5]
        else:
            assert(len(stride)==4)
            self.stride = stride
        self.num_class = num_class

        self.L = spharm_L
        self.useArea, self.wgtBnd = (True, 1.) # recommended hyper-parameters for mesh decimation
        self.bn_momentum = 0.01
        self.num_blocks = 5
        self.Niters = [2, 2, 4, 4, 4]
        self.radius = [0.2, 0.4, 0.8]
        self.EncodeChannels = [32, 64, 96, 128, 192, 256]
        self.useHeight = use_height
        if self.useHeight:
            geometry_in_channels = 12
        else:
            geometry_in_channels = 9

        self.min_nv = 80  # set the minimum vertex number for the coarsest layer
        self.stride = np.float32(self.stride)
        self.nv_limit = np.int32(np.cumprod(self.stride[::-1])[::-1]*self.min_nv)

        self.build_mesh_hierarchy = meshUtil.MeshHierarchy(self.stride, self.nv_limit,
                                                           self.useArea, self.wgtBnd)
        self.cal_sph_harm = SphHarmCoeff(self.L)

        # FirstBlock
        self.Conv0 = FirstBlock(geometry_in_channels, self.EncodeChannels[0], self.L,
                                bn_momentum=self.bn_momentum)

        # setting of convolution blocks in the encoder
        growth_rate = 32
        Enc1 = EncoderMeshBlock(self.EncodeChannels[0], self.EncodeChannels[1], growth_rate,
                                self.L, max_iter=self.Niters[0],
                                bn_momentum=self.bn_momentum)
        Enc2 = EncoderMeshBlock(self.EncodeChannels[1], self.EncodeChannels[2], growth_rate,
                                self.L, max_iter=self.Niters[1],
                                bn_momentum=self.bn_momentum)
        Enc3 = DualBlock(self.EncodeChannels[2], self.EncodeChannels[3], growth_rate,
                         self.L, radius=self.radius[0], max_iter=self.Niters[2],
                         bn_momentum=self.bn_momentum)
        Enc4 = DualBlock(self.EncodeChannels[3], self.EncodeChannels[4], growth_rate,
                         self.L, radius=self.radius[1], max_iter=self.Niters[3],
                         bn_momentum=self.bn_momentum)
        Enc5 = DualBlock(self.EncodeChannels[4], self.EncodeChannels[5], growth_rate,
                         self.L, radius=self.radius[2], max_iter=self.Niters[4],
                         bn_momentum=self.bn_momentum)
        self.Block = nn.ModuleList([Enc1, Enc2, Enc3, Enc4, Enc5])

        # mesh pooling, unpooling configuration
        self.Mesh_Pool = mesh_pool(method='max')
        self.Mesh_Unpool = mesh_unpool()
        self.globalPool = global_mesh_pool(method='avg')

        # classifier
        self.FC1 = PerItemConv3d(self.EncodeChannels[-1], 128)
        self.DP = nn.Dropout(0.2)
        self.FC2 = PerItemConv3d(128, self.num_class, with_bn=False, activation_fn=None)

    def extract_facet_geometric_features(self, vertex, face, normals):
        face = face.to(torch.long)
        V1 = vertex[face[:,0],:]
        V2 = vertex[face[:,1],:]
        V3 = vertex[face[:,2],:]

        Height_features = torch.stack([V1[:,2], V2[:,2], V3[:,2]], dim=1)

        Delta12 = V2[:,:3] - V1[:,:3]
        Delta23 = V3[:,:3] - V2[:,:3]
        Delta31 = V1[:,:3] - V3[:,:3]

        # length of edges
        L12 = torch.norm(Delta12, dim=-1, keepdim=True)
        L23 = torch.norm(Delta23, dim=-1, keepdim=True)
        L31 = torch.norm(Delta31, dim=-1, keepdim=True)
        Length_features = torch.cat([L12, L23, L31], dim=-1)

        # angles between edges
        Theta1 = torch.sum(Delta12*(-Delta31), dim=-1, keepdim=True)/(L12*L31)
        Theta2 = torch.sum((-Delta12)*Delta23, dim=-1, keepdim=True)/(L12*L23)
        Theta3 = torch.sum((-Delta23)*Delta31, dim=-1, keepdim=True)/(L23*L31)
        Theta_features = torch.cat([Theta1, Theta2, Theta3], dim=-1)

        if self.useHeight:
            geometry_features = torch.cat([Length_features, Theta_features,
                                           normals, Height_features], dim=-1)
        else:
            geometry_features = torch.cat([Length_features, Theta_features,
                                           normals], dim=-1)
        return geometry_features

    def get_oriented_normals(self, geometry_in, vertex, face):
        face_normals = geometry_in[:, :3]
        return face_normals

    def __call__(self, vertex_in, face_in, nv_in, mf_in):
        vertex_input = vertex_in
        vertex_in = vertex_in[:,:3]

        # build mesh hierarchy
        mesh_hierarchy = self.build_mesh_hierarchy(vertex_in, face_in, nv_in, mf_in)

        # import open3d as o3d
        # for i in range(self.num_blocks):
        #     vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[i][:4]
        #     mesh = o3d.geometry.TriangleMesh()
        #     mesh.vertices = o3d.utility.Vector3dVector(vertex_in[:,:3].cpu().detach().numpy())
        #     mesh.triangles = o3d.utility.Vector3iVector(face_in.cpu().detach().numpy())
        #     lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        #     # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        #     o3d.visualization.draw_geometries([lineset]) # mesh_show_back_face=True)

        # ============================================Initial Conv==============================================
        vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[0][:4]
        full_vt_map = torch.arange(vertex_in.shape[0],dtype=torch.int).to(vertex_in.get_device())
        full_nf_count = meshUtil.count_vertex_adjface(face_in, full_vt_map, vertex_in)
        face_normals = self.get_oriented_normals(geometry_in, vertex_in, face_in)
        facet_geometrics = self.extract_facet_geometric_features(vertex_input, face_in, face_normals)
        filt_coeff = self.cal_sph_harm(cart_coords=face_normals)
        feats = self.Conv0(facet_geometrics, face_in, full_nf_count, full_vt_map, filt_coeff)
        # ===============================================The End================================================

        # ============================================Encoder Flow==============================================
        for k in range(self.num_blocks):
            # block computation:
            vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[k][:4]
            full_vt_map = torch.arange(vertex_in.shape[0], dtype=torch.int).to(vertex_in.get_device())
            full_nf_count = meshUtil.count_vertex_adjface(face_in, full_vt_map, vertex_in)
            face_normals = self.get_oriented_normals(geometry_in, vertex_in, face_in)
            filt_coeff = self.cal_sph_harm(cart_coords=face_normals)
            feats = self.Block[k](feats, vertex_in, face_in, full_nf_count, full_vt_map, filt_coeff, nv_in)

            if k<(self.num_blocks-1):
                vertex_out, _, _, _, _, vt_replace, vt_map = mesh_hierarchy[k+1]
                feats = self.Mesh_Pool(feats, vt_replace, vt_map, vertex_out)  # mesh pool
        # ===============================================The End================================================
        avg_pooled = self.globalPool(feats, nv_in)

        feats = self.FC1(avg_pooled)
        DP_feats = self.DP(feats)
        pred_logits = self.FC2(DP_feats)
        return pred_logits