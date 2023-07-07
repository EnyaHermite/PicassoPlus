import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d


class CustomMeshDataset(Dataset):
    def __init__(self, annotations_files, mesh_files, transform=None, normalize=None):
        super(CustomMeshDataset, self).__init__()
        self.mesh_labels = annotations_files
        self.mesh_files = mesh_files
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        label_path = self.mesh_labels[idx]
        raw_mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertex = torch.tensor(raw_mesh.vertices, dtype=torch.float)
        face = torch.tensor(raw_mesh.triangles, dtype=torch.int32)
        label = np.loadtxt(label_path, dtype=np.int32)
        assert (face.min()==0)

        # get torch tensors of (vertex, face, label)
        vertex = torch.tensor(vertex).to(torch.float)
        face = torch.tensor(face).to(torch.long)
        label = torch.tensor(label).view(-1)

        if self.normalize:
            vertex = self.normalize(vertex)

        if self.transform:
            vertex, face, label = self.transform(vertex, face, label)

        face = face.to(torch.int)
        nv = torch.tensor(vertex.shape[0]).to(torch.int)
        mf = torch.tensor(face.shape[0]).to(torch.int)
        return vertex, face, nv, mf, label

    def collate_fn(self, batch_data):
        batch_vertex = [item[0] for item in batch_data]
        batch_nv = [item[2] for item in batch_data]
        batch_mf = [item[3] for item in batch_data]
        batch_label = [item[-1] for item in batch_data]

        batch_vertex = torch.concat(batch_vertex, dim=0)
        batch_label = torch.concat(batch_label, dim=0)
        batch_mf = torch.tensor(batch_mf)

        vid_offsets = torch.cumsum(torch.tensor([0] + batch_nv), dim=0)
        batch_face = [item[1] + vid_offsets[i] for i, item in enumerate(batch_data)]
        batch_face = torch.concat(batch_face, dim=0)
        batch_nv = torch.tensor(batch_nv)

        return batch_vertex, batch_face, batch_nv, batch_mf, batch_label











