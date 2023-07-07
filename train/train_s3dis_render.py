import warnings
warnings.filterwarnings("ignore")
import argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
from glob import glob
from datetime import datetime
from torchvision import transforms
from picasso.augmentor import Augment
from torch.utils.tensorboard import SummaryWriter
from picasso.networks.scene_seg import PicassoNetII
from picasso.mesh.dataset_scene import CustomMeshDataset, CustomCollate
from torch.utils.data import DataLoader
from fit import MyFit


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to hdf5 data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--ckpt_path', default=None, help='data path to checkpoint')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--max_nv', type=int, default=1400000, help='maximum vertices allowed in a batch')
parser.add_argument('--degree', type=int, default=3, help='degree of spherical harmonics')
parser.add_argument('--test_fold', type=int, default=5, help='index of test Area')
opt = parser.parse_args()


def align_fn(vertex):
    assert (vertex.shape[-1] == 3)
    xyz_min = torch.min(vertex, dim=0, keepdim=True)[0]
    xyz_max = torch.max(vertex, dim=0, keepdim=True)[0]
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_center[0][-1] = xyz_min[0][-1]
    vertex -= xyz_center
    return vertex

class augment_fn:
    def __init__(self, prob=0.5):
        self.prob = prob
        CustomColorJitter = transforms.ColorJitter(brightness=(0.75, 1.25), contrast=(0.5, 1.5))
        self.color_transform = torch.nn.Sequential(CustomColorJitter, transforms.RandomGrayscale(0.1))

    def __call__(self, vertex, face, label, texture, bcoeff, kt):
        assert(vertex.shape[-1]==3)

        # geometry augmentation
        vertex = Augment.rotate_point_cloud(vertex, upaxis=3, prob=self.prob)
        vertex = Augment.rotate_perturbation_point_cloud(vertex, prob=self.prob)
        vertex = Augment.flip_point_cloud(vertex, prob=0.5)
        vertex = Augment.random_scale_point_cloud(vertex, prob=self.prob)
        vertex = Augment.shift_point_cloud(vertex, prob=self.prob)

        # texture augmentation
        texture = Augment.shift_color(texture, prob=self.prob)
        texture = Augment.jitter_color(texture, prob=self.prob)
        texture = texture.permute([1, 0])[..., None]
        texture = self.color_transform(texture)
        texture = torch.squeeze(texture)
        texture = texture.permute([1,0])

        # random drop out faces
        vertex, face, label, \
            face_mask = Augment.random_drop_vertex(vertex, face, label,
                                                   drop_rate=0.15, prob=0.5,
                                                   return_face_mask=True)
        valid_indices = torch.repeat_interleave(face_mask, kt, dim=0)
        texture = texture[valid_indices]
        bcoeff = bcoeff[valid_indices]
        kt = kt[face_mask]

        assert(vertex.shape[0]==label.shape[0])
        assert(face.shape[0]==kt.shape[0])
        assert(texture.shape[0]==bcoeff.shape[0])
        return vertex, face, label, texture, bcoeff, kt


if __name__ == '__main__':
    # hyperparameter settings
    MAX_EPOCHS = 300
    device = torch.device("cuda")

    # load the datapath of all training and testing files
    Meshes, Labels = {"train": [], "test": []}, {"train": [], "test": []}
    area_ids = [*range(1, 7)]
    area_ids.remove(opt.test_fold)
    for id in area_ids:
        Meshes['train'] += glob(f"{opt.data_dir}/Area_{id}/*.h5")
    Labels['train'] = [filename.replace('.h5', '.txt') for filename in Meshes['train']]

    Meshes['test'] = glob(f"{opt.data_dir}/Area_{opt.test_fold}/*.h5")
    Labels['test'] = [filename.replace('.h5', '.txt') for filename in Meshes['test']]

    # build training set dataloader
    repeat = np.ceil((600*opt.batch_size)/len(Meshes['train'])).astype('int32')
    trainSet = CustomMeshDataset(Labels['train']*repeat, Meshes['train']*repeat,
                                 transform=augment_fn(0.5), normalize=align_fn)
    trainLoader = DataLoader(trainSet, shuffle=True, batch_size=1, num_workers=6,
                             collate_fn=CustomCollate(batch_size=opt.batch_size, max_nv=opt.max_nv))
    # build validation set dataloader
    testSet = CustomMeshDataset(Labels['test'], Meshes['test'], transform=None, normalize=align_fn)
    testLoader = DataLoader(testSet, shuffle=False, batch_size=1, num_workers=0,
                            collate_fn=CustomCollate(batch_size=1, max_nv=opt.max_nv))

    # create model, loss function, optimizer, learning rate scheduler, and writer for tensorboard
    NUM_CLASSES = 13
    classnames = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
                  'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    model = PicassoNetII(num_class=NUM_CLASSES, stride=[4, 3, 3, 2, 2], spharm_L=opt.degree, use_height=True).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    if opt.ckpt_path is not None:
        checkpoint = torch.load(opt.ckpt_path)
        write_folder = os.path.dirname(opt.ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ckpt_epoch = checkpoint['epoch'] + 1
        print(ckpt_epoch)
        print("model loaded.")
    else:
        ckpt_epoch = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        write_folder = 'runs_scenes/s3dis_render_{}'.format(timestamp)
    print("write_folder:", write_folder)
    writer = SummaryWriter(write_folder)
    os.system('cp %s %s' % (__file__, write_folder))  # bkp of train procedure
    os.system('cp picasso/mesh/layers.py %s' % (write_folder))
    os.system('cp picasso/networks/scene_seg.py %s' % (write_folder))
    fout = open(os.path.join(write_folder, 'log_train.txt'), 'a')
    fout.write(str(opt) + '\n')

    fit = MyFit(model=model, optimizer=optimizer, scheduler=scheduler,
                writer=writer, loss=loss_fn, device=device, fout=fout)
    fit(ckpt_epoch, MAX_EPOCHS, trainLoader, testLoader, write_folder,
        report_iou=True, class_names=classnames)
