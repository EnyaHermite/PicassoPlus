import torch
import numpy as np
from torchvision.transforms import ColorJitter


class Augment:

    def __init__(self):
        pass

    @classmethod
    def rotate_point_cloud(cls, xyz, max_angle=2*np.pi, upaxis=3, prob=0.95):
        """ Randomly rotate the point clouds to augment the dataset
            rotation is vertical
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        if torch.rand([])<prob:
            angle = torch.rand([1])*max_angle
            if upaxis==1:
                R = cls.rot_x(angle)
            elif upaxis==2:
                R = cls.rot_y(angle)
            elif upaxis==3:
                R = cls.rot_z(angle)
            else:
                raise ValueError('unkown spatial dimension')
            xyz = torch.matmul(xyz, R)
        return xyz

    @classmethod
    def rotate_perturbation_point_cloud(cls, xyz, angle_sigma=0.06, prob=0.95):
        """ Randomly perturb the point clouds by small rotations
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        if torch.rand([])<prob:
            angle = torch.randn([1,3])*angle_sigma
            Rx = cls.rot_x(angle[:,0])
            Ry = cls.rot_y(angle[:,1])
            Rz = cls.rot_z(angle[:,2])
            R = torch.matmul(Rz, torch.matmul(Ry, Rx))
            xyz = torch.matmul(xyz, R)
        return xyz

    @classmethod
    def rotate_point_cloud_by_angle(cls, xyz, angle):
        """ Rotate the point cloud along up direction with certain angle.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        Rz = cls.rot_z(angle)
        rotated_xyz = torch.matmul(xyz, Rz)
        return rotated_xyz

    @staticmethod
    def jitter_point_cloud(xyz, sigma=0.001, prob=0.95):
        """ Randomly jitter point heights.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        if torch.rand([])<prob:
            noise = torch.randn(xyz.shape)*sigma
            noise = torch.clip(noise, min=-3*sigma, max=3*sigma)
            xyz += noise
        return xyz

    @staticmethod
    def random_drop_vertex(vertex, face, label, drop_rate=0.1,
                           prob=0.5, return_face_mask=False):
        assert(vertex.shape[0]!=face.shape[0])
        if label.shape[0]==1:  # global classification task
            vertex_label, face_label = None, None
        elif label.shape[0]==vertex.shape[0]:
            vertex_label, face_label = label, None
        elif label.shape[0]==face.shape[0]:
            vertex_label, face_label = None, label

        face_mask = torch.ones([face.shape[0]], dtype=torch.bool)
        if torch.rand([])<prob:
            num_vertices = vertex.shape[0]
            vertex_mask = torch.rand(num_vertices)>drop_rate # True for keep, False for drop

            face = face.to(torch.long)
            face_mask = vertex_mask[face[:,0]] & vertex_mask[face[:,1]] & vertex_mask[face[:,2]]
            face = face[face_mask,:]

            uni_ids, idx = torch.unique(face.view(-1),sorted=True,return_inverse=True)
            face = idx.view(-1, 3)
            vertex = vertex[uni_ids,:]

            if vertex_label is not None:
                label = vertex_label[uni_ids]
            elif face_label is not None:
                label = face_label[face_mask]

        if return_face_mask:
            return vertex, face, label, face_mask
        else:
            return vertex, face, label

    @staticmethod
    def shift_point_cloud(xyz, sigma=0.1, prob=0.95):
        """ Randomly shift point cloud.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, shifted point clouds
        """
        if torch.rand([])<prob:
            shifts = torch.randn([3])*sigma
            xyz += shifts
        return xyz

    @staticmethod
    def random_scale_point_cloud(xyz, scale_low=0.8, scale_high=1.2, prob=0.95):
        """ Randomly scale the point cloud.
            Input:
                Nx3 array, original point clouds
            Return:
                Nx3 array, scaled point clouds
        """
        if torch.rand([])<prob:
            scales = torch.rand([3])*(scale_high-scale_low) + scale_low
            xyz *= scales
        return xyz

    @staticmethod
    def flip_point_cloud(xyz, prob=0.95):
        """ Randomly flip the point cloud.
            Input:
                Nx3 array, original point clouds
            Return:
                Nx3 array, flipped point clouds
        """
        if torch.rand([])<prob:
            x = xyz[:,0]
            y = xyz[:,1]
            z = xyz[:,2]
            if torch.rand([])<0.5:
                x = - x  # flip x-dimension(horizontal)
            if torch.rand([])<0.5:
                y = - y  # flip y-dimension(vertical)
            xyz = torch.stack([x, y, z], dim=1)
        return xyz

    @staticmethod
    def shift_color(color, shift_range=0.1, prob=0.95):
        """ Randomly shift color.
            Input:
              Nx3 array, original point colorss
            Return:
              Nx3 array, shifted  point colors
        """
        if torch.rand([])<prob:
            shifts = torch.rand([3])*shift_range
            color += shifts
            color = torch.clip(color, min=0., max=1.)
        return color

    @staticmethod
    def jitter_color(color, sigma=0.05, prob=0.95):
        """ Randomly jitter colors.
            Input:
              Nx3 array, original point colors
            Return:
              Nx3 array, jittered point colors
        """
        if torch.rand([])<prob:
            noise = torch.randn(color.shape)*sigma
            color += noise
            color = torch.clip(color, min=0., max=1.)
        return color

    @staticmethod
    def color_brightness(texture, brightness: tuple[float, float]=(0.5, 1.5), prob=0.5):
        if torch.rand([]) < prob:
            color_transform = ColorJitter(brightness=brightness)
            texture = texture.permute([1, 0])[..., None]
            texture = color_transform(texture)
            texture = torch.squeeze(texture)
            texture = texture.permute([1, 0])
        return texture

    @staticmethod
    def color_contrast(texture, contrast: tuple[float, float]=(0.5, 1.5), prob=0.5):
        if torch.rand([]) < prob:
           color_transform = ColorJitter(contrast=contrast)
           texture = texture.permute([1, 0])[..., None]
           texture = color_transform(texture)
           texture = torch.squeeze(texture)
           texture = texture.permute([1, 0])
        return texture

    @staticmethod
    def color_saturation(texture, saturation: tuple[float, float]=(0.5, 1.5), prob=0.5):
        if torch.rand([]) < prob:
            color_transform = ColorJitter(saturation=saturation)
            texture = texture.permute([1, 0])[..., None]
            texture = color_transform(texture)
            texture = torch.squeeze(texture)
            texture = texture.permute([1, 0])
        return texture

    @staticmethod
    def color_hue(texture, hue: tuple[float, float]=(0.5, 1.5), prob=0.5):
        if torch.rand([]) < prob:
            color_transform = ColorJitter(hue=hue)
            texture = texture.permute([1, 0])[..., None]
            texture = color_transform(texture)
            texture = torch.squeeze(texture)
            texture = texture.permute([1, 0])
        return texture

    # @staticmethod
    # def random_drop_color(color, prob=0.1):
    #     """ Randomly drop colors.
    #         Input:
    #           Nx3 array, original point clouds
    #         Return:
    #           Nx3 array, color dropped point clouds
    #     """
    #     if torch.rand([]) < prob:
    #         color *= 0
    #     return color

    @staticmethod
    def clip_color(color, thresh=0.5, prob=0.5):
        """ Clip colors.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, color clipped point clouds
        """
        if torch.rand([])<prob:
            if torch.rand([])<0.5:
                color = torch.clip(color, min=0., max=thresh)
            else:
                color = torch.clip(color, min=thresh, max=1.)
        return color

    @staticmethod
    def random_drop_color(color, drop_rate=0.3, prob=0.5):
        """ Randomly drop colors.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, color dropped point clouds
        """
        if torch.rand([])<prob:
            color_mask = (torch.rand(color.shape)>drop_rate).to(torch.float)
            color *= color_mask
        return color

    @staticmethod
    def rot_x(angle):
        cosval = torch.cos(angle)
        sinval = torch.sin(angle)
        val0 = torch.zeros_like(cosval)
        val1 = torch.ones_like(cosval)
        R = torch.stack([val1, val0, val0,
                         val0, cosval, -sinval,
                         val0, sinval, cosval], dim=1)
        R = torch.reshape(R, (3, 3))
        return R

    @staticmethod
    def rot_y(angle):
        cosval = torch.cos(angle)
        sinval = torch.sin(angle)
        val0 = torch.zeros_like(cosval)
        val1 = torch.ones_like(cosval)
        R = torch.stack([cosval, val0, sinval,
                         val0, val1, val0,
                        -sinval, val0, cosval], dim=1)
        R = R.view(3, 3)
        return R

    @staticmethod
    def rot_z(angle):
        cosval = torch.cos(angle)
        sinval = torch.sin(angle)
        val0 = torch.zeros_like(cosval)
        val1 = torch.ones_like(cosval)
        R = torch.stack([cosval, -sinval, val0,
                         sinval, cosval, val0,
                         val0, val0, val1], dim=1)
        R = R.view(3, 3)
        return R