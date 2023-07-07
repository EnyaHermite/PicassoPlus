import numpy as np
import os, sys
import glob
from scipy import stats
import meshio as mio


dataDir = '../data/human_seg'
folders = glob.glob(dataDir+'/*')
if not os.path.exists(os.path.join(dataDir, 'face_label')):
    os.makedirs(os.path.join(dataDir, 'face_label'))

def edge2face_label(objPath):
    for currPath in objPath:
        modelName = os.path.basename(currPath)
        label_path = os.path.join(dataDir, 'seg', modelName.replace('.obj', '.eseg'))
        seg_labels = np.loadtxt(label_path, dtype=np.int32)

        # read in the .obj mesh file with meshio, open3d is not working properly
        Mesh = mio.read(currPath)
        xyz = np.asarray(Mesh.points, dtype=np.float32)
        face = np.asarray(Mesh.cells_dict['triangle'], dtype=np.int32)

        begin_face_id = np.amin(face)
        if begin_face_id > 0:
            face -= begin_face_id
        assert (np.amin(face) == 0)

        # =================================location normalization=================================
        xyz_min = np.amin(xyz, axis=0, keepdims=True)
        xyz_max = np.amax(xyz, axis=0, keepdims=True)
        xyz_center = (xyz_min + xyz_max) / 2
        xyz -= xyz_center  # align to room bottom center
        box = np.squeeze((xyz_max - xyz_min) / 2)
        # ========================================================================================

        # ============================Transform edge label to face labels=========================
        # ++++++++++++++++++++++We use majority edge label voting on each facet+++++++++++++++++++
        v1, v2, v3 = face[:, 0], face[:, 1], face[:, 2]
        edge = np.stack([v1, v2, v2, v3, v1, v3], axis=1)
        edge = np.reshape(edge, newshape=[-1, 2])
        edge_pool = np.sort(edge, axis=-1)
        edge, uni_indices, \
            inv_indices = np.unique(edge_pool, axis=0, return_index=True, return_inverse=True)
        sortIdx = np.argsort(np.argsort(uni_indices))  # two argsort works
        edge_labels = seg_labels[sortIdx]
        edge_pool_labels = np.reshape(edge_labels[inv_indices], [-1, 3])
        face_labels = stats.mode(edge_pool_labels, axis=1)[0]
        face_labels = np.squeeze(face_labels)
        # ========================================================================================
        assert (face_labels.shape[0] == face.shape[0])
        assert (np.amin(face_labels) >= 1 and np.amax(face_labels) <= 8)

        face_labels = face_labels - np.min(face_labels)
        np.savetxt(dataDir+f"/face_label/{modelName[:-4]}.txt", face_labels, fmt='%d')

    return


if __name__=='__main__':
    train_files = glob.glob(os.path.join(dataDir, 'train/*.obj'))
    test_files = glob.glob(os.path.join(dataDir, 'test/*.obj'))
    objPath = train_files + test_files

    print(len(train_files), len(test_files))

    train_files = [os.path.basename(file)[:-4] for file in train_files]
    test_files = [os.path.basename(file)[:-4] for file in test_files]
    np.savetxt(dataDir + f"/train_files.txt", train_files, fmt='%s')
    np.savetxt(dataDir + f"/test_files.txt", test_files, fmt='%s')

    edge2face_label(objPath)



