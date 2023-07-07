import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class CustomMeshDataset(Dataset):
    '''
    This dataset suits meshes rendered offline. All information is stored
    in the .json file
    '''
    def __init__(self, annotations_files, mesh_files, transform=None, normalize=None):
        super(CustomMeshDataset, self).__init__()
        self.mesh_labels = annotations_files
        self.mesh_files = mesh_files
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        # print(idx)
        mesh_path = self.mesh_files[idx]
        label_path = self.mesh_labels[idx]
        label = np.loadtxt(label_path, dtype=np.int32)
        hf = h5py.File(mesh_path, 'r')
        vertex = np.asarray(hf.get('vertex'))
        face = np.asarray(hf.get('face'))
        texture = np.asarray(hf.get('face_texture'))
        bcoeff = np.asarray(hf.get('bary_coeff'))
        kt = np.asarray(hf.get('num_texture'))
        hf.close()

        vertex = torch.tensor(vertex).to(torch.float)
        face = torch.tensor(face).to(torch.long)
        texture = torch.tensor(texture).to(torch.float)
        bcoeff = torch.tensor(bcoeff).to(torch.float)
        kt = torch.tensor(kt).to(torch.int)
        label = torch.tensor(label).view(-1)

        if self.normalize:
            vertex = self.normalize(vertex)

        if self.transform:
            vertex, face, label, \
            texture, bcoeff, kt = self.transform(vertex, face, label,
                                                 texture, bcoeff, kt)

        face = face.to(torch.int)
        nv = torch.tensor(vertex.shape[0]).to(torch.int)
        mf = torch.tensor(face.shape[0]).to(torch.int)
        return vertex, face, nv, mf, texture, bcoeff, kt, label



class CustomCollate:
    def __init__(self, batch_size=1, max_nv=1000000):
        self.batch = []
        self.batch_size = batch_size
        self.max_nv = max_nv

    def __call__(self, data):
        self.batch += data
        trunc_id = self._trunc_batch_()
        if trunc_id==(len(self.batch)-1):
            batch_data = self._collate_fn_(self.batch[:trunc_id])
            self.batch = self.batch[trunc_id:]
            return batch_data

        if len(self.batch)==self.batch_size:
            batch_data = self._collate_fn_(self.batch)
            self.batch = []
            return batch_data

    def _trunc_batch_(self):
        batch_nv = [item[2] for item in self.batch]
        batch_nv = torch.tensor(batch_nv)
        cumsum_nv = torch.cumsum(batch_nv, dim=0)
        valid_indices = torch.where(cumsum_nv <= self.max_nv)[0]
        if valid_indices.shape[0]>0:
            trunc_batch_size = valid_indices[-1] + 1
        else:
            trunc_batch_size = 1
        return trunc_batch_size

    def _collate_fn_(self, batch_data):
        batch_texture = [item[4] for item in batch_data]
        batch_bcoeff = [item[5] for item in batch_data]
        batch_kt = [item[6] for item in batch_data]
        batch_label = [item[7] for item in batch_data]

        batch_texture = torch.concat(batch_texture, dim=0)
        batch_bcoeff = torch.concat(batch_bcoeff, dim=0)
        batch_kt = torch.concat(batch_kt, dim=0)
        batch_label = torch.concat(batch_label, dim=0)

        batch_vertex = [item[0] for item in batch_data]
        batch_nv = [item[2] for item in batch_data]
        batch_mf = [item[3] for item in batch_data]

        batch_vertex = torch.concat(batch_vertex, dim=0)
        batch_mf = torch.tensor(batch_mf)

        vid_offsets = torch.cumsum(torch.tensor([0] + batch_nv), dim=0)
        batch_face = [item[1] + vid_offsets[i] for i, item in enumerate(batch_data)]
        batch_face = torch.concat(batch_face, dim=0)
        batch_nv = torch.tensor(batch_nv)

        return batch_vertex, batch_face, batch_nv, batch_mf, \
            batch_texture, batch_bcoeff, batch_kt, batch_label








