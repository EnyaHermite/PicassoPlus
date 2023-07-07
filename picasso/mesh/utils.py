import torch
import numpy as np
import open3d as o3d
from .pi_modules.decimate import mesh_decimation_, compute_triangle_geometry_, \
                                 count_vertex_adjface_, combine_clusters_


def mesh_decimation(vertexIn, faceIn, geometryIn, nvIn, mfIn, nvSample, useArea=True, wgtBnd=1.):
    vertexOut, faceOut, geometryOut, nvOut, mfOut, \
    repOut, mapOut = mesh_decimation_(vertexIn, faceIn, geometryIn, nvIn, mfIn,
                                      nvSample, useArea=useArea, wgtBnd=wgtBnd)
    return vertexOut, faceOut, geometryOut, nvOut, mfOut, repOut, mapOut


# triangle geometry include [unit length normals, intercept of the triangle plane, face area]
def compute_triangle_geometry(vertex, face):
    geometry = compute_triangle_geometry_(vertex, face)
    return geometry


# Count the number of adjacent faces for output vertices, i.e. vertices with vtMap[i]>=0
def count_vertex_adjface(face, vtMap, vertexOut):
    nf_count = count_vertex_adjface_(face, vtMap, vertexOut)
    return nf_count


def combine_clusters(rep_highRes, map_highRes, rep_lowRes, map_lowRes):
    rep_combined, map_combined = combine_clusters_(rep_highRes, map_highRes, rep_lowRes, map_lowRes)
    return rep_combined, map_combined


class MeshHierarchy:
    def __init__(self, *params):
        self.num_params = len(params)
        if self.num_params==3:
            self.nv_samples, self.useArea, self.wgtBnd = params
            self.Iters = len(self.nv_samples)
        elif self.num_params==4:
            self.stride, self.min_nvOut, self.useArea, self.wgtBnd = params
            self.Iters = len(self.stride)

    def __call__(self, vertex_in, face_in, nv_in, mf_in):
        mesh_hierarchy = []
        geometry_in = compute_triangle_geometry(vertex_in, face_in)
        mesh_hierarchy.append((vertex_in, face_in, geometry_in, nv_in, mf_in, None, None))

        for l in range(self.Iters):
            if self.num_params==3:
                nv_out = self.nv_samples[l]
            elif self.num_params==4:
                nv_out = (nv_in.to(torch.float)/float(self.stride[l])).to(torch.int)
                nv_out = torch.clip(nv_out, min=self.min_nvOut[l])
            nv_out = nv_out.to(nv_in.get_device())
            dec_mesh = mesh_decimation(vertex_in, face_in, geometry_in, nv_in, mf_in,
                                       nv_out, self.useArea, self.wgtBnd)
            mesh_hierarchy.append(dec_mesh)
            vertex_in, face_in, geometry_in, nv_in, mf_in = dec_mesh[:5]
        return mesh_hierarchy


def view_geometry(geometry):
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()