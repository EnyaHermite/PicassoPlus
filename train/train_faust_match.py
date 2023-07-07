import warnings
warnings.filterwarnings("ignore")
import argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from datetime import datetime
from picasso.augmentor import Augment
from picasso.networks.shape_seg import PicassoNetII
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from picasso.mesh.dataset_shape import CustomMeshDataset
from glob import glob
from fit import MyFit

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to mesh data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--ckpt_path', default=None, help='data path to checkpoint')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 6]')
parser.add_argument('--degree', type=int, default=3, help='degree of spherical harmonics')
opt = parser.parse_args()


def normalize_fn(vertex):
    assert(vertex.shape[-1]==3)
    xyz_min = torch.min(vertex, dim=0)[0]
    xyz_max = torch.max(vertex, dim=0)[0]
    xyz_center = (xyz_min + xyz_max)/2
    vertex -= xyz_center
    scale = torch.max(xyz_max - xyz_min)/2
    vertex /= scale
    return vertex


class augment_fn:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, vertex, face, label):
        assert(vertex.shape[1]==3)
        vertex = Augment.flip_point_cloud(vertex, prob=self.prob)
        vertex = Augment.random_scale_point_cloud(vertex, scale_low=0.5,
                                                  scale_high=1.5, prob=self.prob)
        vertex = Augment.rotate_point_cloud(vertex, upaxis=1, prob=self.prob)
        vertex = Augment.rotate_point_cloud(vertex, upaxis=2, prob=self.prob)
        vertex = Augment.rotate_point_cloud(vertex, upaxis=3, prob=self.prob)
        vertex = Augment.jitter_point_cloud(vertex, sigma=0.01, prob=self.prob)

        # random drop out faces
        vertex, face, label = Augment.random_drop_vertex(vertex, face, label,
                                                         drop_rate=0.15, prob=0.5)
        return vertex, face, label


if __name__=='__main__':
    # hyper-parameter settings
    MAX_EPOCHS = 300
    device = torch.device("cuda")

    # file_lists
    Meshes, Labels = {'train':[], 'test':[]}, {'train':[], 'test':[]}
    mesh_files = glob(f'{opt.data_dir}/*.ply')
    mesh_files.sort()
    label_file = f'{os.path.dirname(opt.data_dir)}/match_labels.txt'
    Meshes['train'] = mesh_files[:80]
    Meshes['test'] = mesh_files[80:]
    Labels['train'] = [label_file]*len(Meshes['train'])
    Labels['test'] = [label_file]*len(Meshes['test'])

    # build training set dataloader
    repeat = opt.batch_size*500//len(Meshes['train']) + 1
    trainSet = CustomMeshDataset(Labels['train'] * repeat, Meshes['train'] * repeat,
                                 transform=augment_fn(0.5), normalize=normalize_fn)
    trainLoader = DataLoader(trainSet, batch_size=opt.batch_size, shuffle=True, num_workers=4,
                             collate_fn=trainSet.collate_fn)
    # build validation set dataloader
    testSet = CustomMeshDataset(Labels['test'], Meshes['test'],
                                transform=None, normalize=normalize_fn)
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False,
                            collate_fn=testSet.collate_fn)
    # create model, loss function, optimizer, learning rate scheduler, and writer for tensorboard
    NUM_CLASSES = 6890  # this is the number of candidate correspondences in the reference
    classnames = [str(i) for i in range(NUM_CLASSES)]
    model = PicassoNetII(num_class=NUM_CLASSES, stride=[3, 2, 2, 2], pred_facet=False, spharm_L=opt.degree,
                         use_height=False).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
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
        write_folder = 'runs_shapes/{}_CAD_{}'.format(os.path.basename(opt.data_dir), timestamp)
    print("write_folder:", write_folder)
    writer = SummaryWriter(write_folder)
    os.system('cp %s %s' % (__file__, write_folder))  # bkp of train procedure
    os.system('cp picasso/networks/shape_seg.py %s' % (write_folder))
    fout = open(os.path.join(write_folder, 'log_train.txt'), 'a')
    fout.write(str(opt) + '\n')

    fit = MyFit(model=model, optimizer=optimizer, scheduler=scheduler,
                writer=writer, loss=loss_fn, device=device, fout=fout)
    fit(ckpt_epoch, MAX_EPOCHS, trainLoader, testLoader, write_folder,
        report_iou=False, class_names=classnames)

