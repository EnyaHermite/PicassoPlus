import warnings
warnings.filterwarnings("ignore")
import argparse, os
import numpy as np
import torch, h5py
from picasso.networks.scene_seg import PicassoNetII
from glob import glob
from sklearn.neighbors import NearestNeighbors
from eval_scene_dense import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to data')
parser.add_argument('--model_path', required=True, help='data path to trained model')
parser.add_argument('--test_fold', type=int, default=5, help='index of test Area')
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
    def __init__(self, voxel_size=None, alpha=0, beta=1, align_fn=None):
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.beta = beta
        self._align_fn_ = align_fn

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

        # load the dense/raw point cloud
        raw_cloud_dir = "./data/S3DIS-Aligned-Raw/Area_%d"%opt.test_fold
        scene_names = os.path.basename(mesh_path).replace('.h5','')
        dense_cloud = np.loadtxt(f"{raw_cloud_dir}/{scene_names}/cloud.txt", delimiter=',')
        dense_label = np.loadtxt(f"{raw_cloud_dir}/{scene_names}/labels.txt")
        dense_pts = torch.tensor(dense_cloud[:,:3]).to(torch.float)
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

    # load the datapath of testing files
    Meshes, Labels = {"test": []}, {"test": []}
    Meshes['test'] = glob(f"{opt.data_dir}/Area_{opt.test_fold}/*.h5")
    Labels['test'] = [filename.replace('.h5', '.txt') for filename in Meshes['test']]
    print("#test samples=%d." % len(Meshes['test']))

    # initialize the data transformations
    voxel_grid, alpha, beta = (3, 5, 3)
    transform = TransformTexture(voxel_size=voxel_grid, alpha=alpha, beta=beta, align_fn=align_fn)

    # network configuration
    NUM_CLASSES = 13
    classnames = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
                  'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    model = PicassoNetII(num_class=NUM_CLASSES, stride=[4,3,3,2,2], spharm_L=opt.degree, use_height=True).to(device)

    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_fn = torch.nn.CrossEntropyLoss()

    evaluate(model, loss_fn, Meshes['test'], Labels['test'], transform, classnames, device)

