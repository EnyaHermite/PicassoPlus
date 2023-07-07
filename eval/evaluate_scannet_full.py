import warnings
warnings.filterwarnings("ignore")
import argparse, os
import numpy as np
import torch
from picasso.networks.scene_seg import PicassoNetII
from eval_scene_dense import evaluate
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to tfrecord data')
parser.add_argument('--model_path', required=True, help='data path to checkpoint')
parser.add_argument('--degree', type=int, default=3, help='degree of spherical harmonics')
opt = parser.parse_args()


def align_fn(vertex):
    assert (vertex.shape[-1] == 3)
    xyz_min = torch.min(vertex, dim=0, keepdim=True)[0]
    xyz_max = torch.max(vertex, dim=0, keepdim=True)[0]
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_center[0][-1] = xyz_min[0][-1]
    vertex -= xyz_center
    return vertex


class TransformTexture:

    def __init__(self, align_fn=None, knn=5, raw_mesh_dir=None):
        self._align_fn_ = align_fn
        self.knn = knn
        self.raw_mesh_dir = raw_mesh_dir

    def __call__(self, mesh_path, label_path):
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

        # load the dense point cloud
        scene_names = os.path.basename(mesh_path).replace('.h5', '')
        raw_mesh = o3d.io.read_triangle_mesh(f"{raw_mesh_dir}/{scene_names}/{scene_names}.ply")
        dense_label = np.loadtxt(f"{raw_mesh_dir}/{scene_names}/{scene_names}_scan20_labels.txt")
        dense_pts = torch.tensor(raw_mesh.vertices).to(torch.float)
        dense_label = torch.tensor(dense_label)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vertex)
        nn_dst, nn_idx = nbrs.kneighbors(dense_pts)
        nn_idx = torch.tensor(nn_idx.reshape(-1)).to(torch.long)

        if self._align_fn_:
            vertex = self._align_fn_(vertex)

        face = face.to(torch.int)
        nv = torch.tensor([vertex.shape[0]]).to(torch.int)
        mf = torch.tensor([face.shape[0]]).to(torch.int)
        return vertex, face, nv, mf, texture, bcoeff, kt, dense_label, nn_idx


if __name__=='__main__':
    device = torch.device("cuda")

    # load the datapath of all training and testing files
    Meshes, Labels = {}, {}
    test_files = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'val_files.txt'))]
    Meshes['val'] = [f"{opt.data_dir}/val/{filename}.h5" for filename in test_files]
    Labels['val'] = [f"{opt.data_dir}/val/{filename}.txt" for filename in test_files]
    print("#test samples=%d." %len(Meshes['val']))

    # create model, loss function, optimizer, learning rate scheduler, and writer for tensorboard
    NUM_CLASSES = 20
    classnames = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                  'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator',
                  'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    model = PicassoNetII(num_class=NUM_CLASSES, stride=[4,3,3,2,2], spharm_L=opt.degree, use_height=True).to(device)

    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_fn = torch.nn.CrossEntropyLoss()

    raw_mesh_dir = './data/ScanNet/scans_trainval'

    # initialize the train and test transformations
    transform = TransformTexture(align_fn=align_fn, raw_mesh_dir=raw_mesh_dir)
    evaluate(model, loss_fn, Meshes['val'], Labels['val'], transform, classnames, device)
