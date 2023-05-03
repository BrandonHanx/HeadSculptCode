import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import get_rays, safe_normalize

DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)

def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis, overhead, front, back):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - back / 2)] = 1
    res[(phis >= np.pi - back / 2) & (phis < np.pi + back / 2)] = 2
    res[(phis >= np.pi + back / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    # res[thetas <= overhead] = 4
    # res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[20, 110], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=70, angle_back=70, jitter=False, uniform_sphere_rate=0.5, sample_p=[1, 1, 1, 1]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    angle_back = np.deg2rad(angle_back)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    
    def get_phi(x):
        if x == 0:
            phi = torch.rand(1, device=device) * angle_front - angle_front / 2
        elif x == 1:
            phi = torch.rand(1, device=device) * angle_back + np.pi - angle_back / 2
        elif x == 2:
            phi = torch.rand(1, device=device) * (np.pi - angle_back / 2 - angle_front / 2) + angle_front / 2
        elif x == 3:
            phi = torch.rand(1, device=device) * (np.pi - angle_front / 2 - angle_back / 2) + np.pi + angle_back / 2
        return phi
    
    # A temp hack soultion here, need to fix
    if size == 4:
        sample_p = np.array(sample_p)
        sample_p = sample_p / np.sum(sample_p)
        chosen_angle = np.random.choice(4, 4, p=sample_p)
        phis = torch.cat([get_phi(x) for x in chosen_angle])
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        centers = torch.stack([
                radius * torch.sin(thetas) * torch.sin(phis),
                radius * torch.cos(thetas),
                radius * torch.sin(thetas) * torch.cos(phis),
            ], dim=-1) # [B, 3]
    else:
        if random.random() < uniform_sphere_rate:
            unit_centers = F.normalize(
                torch.stack([
                    (torch.rand(size, device=device) - 0.5) * 2.0,
                    torch.rand(size, device=device),
                    (torch.rand(size, device=device) - 0.5) * 2.0,
                ], dim=-1), p=2, dim=1
            )
            thetas = torch.acos(unit_centers[:,1])
            phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
            phis[phis < 0] += 2 * np.pi
            centers = unit_centers * radius.unsqueeze(-1)
        else:
            thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
            phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

            centers = torch.stack([
                radius * torch.sin(thetas) * torch.sin(phis),
                radius * torch.cos(thetas),
                radius * torch.sin(thetas) * torch.cos(phis),
            ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, angle_back)
    else:
        dirs = None
    
    return poses, dirs


def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=70, angle_back=70):

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    angle_back = np.deg2rad(angle_back)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, angle_back)
    else:
        dirs = None
    
    return poses, dirs    
    

class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']
        
        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.opt.min_near
        self.far = 1000 # infinite

        # [debug] visualize poses
        # poses, dirs = rand_poses(100, self.device, radius_range=self.opt.radius_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())


    def collate(self, index):

        B = len(index) # always 1

        if self.training:
            # random pose on the fly
            poses, dirs = rand_poses(B, self.device, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, angle_back=self.opt.angle_back, jitter=self.opt.jitter_pose, uniform_sphere_rate=self.opt.uniform_sphere_rate, sample_p=self.opt.sample_p)

            # random focal
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]
        else:
            # circle pose
            phi = (index[0] / self.size) * 360
            poses, dirs = circle_poses(self.device, radius=self.opt.radius_range[1] * 1.2, theta=80, phi=phi, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, angle_back=self.opt.angle_back)

            # fixed focal
            # fov = 60 
            fov = (self.opt.fovy_range[1] + self.opt.fovy_range[0]) / 2

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])
        # HACK: fixed 64 for copy model
        copy_focal = 64 / (2 * np.tan(np.deg2rad(fov) / 2))
        copy_intrinsics = np.array([copy_focal, copy_focal, 32, 32])

        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0], 
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        mvp = projection @ torch.inverse(poses) # [1, 4, 4]
        
        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)
        copy_rays = get_rays(poses, copy_intrinsics, 64, 64, -1)
        
        focal = 512 / (2 * np.tan(np.deg2rad(fov) / 2))
        perspective_matrix = torch.Tensor(
            [
                [focal, 0, 256, 0],
                [0, focal, 256, 0],
                [0, 0, 1, 0],
                [0, 0, self.near, 0],
            ]
        ).cuda().unsqueeze(0).expand(B, -1, -1)
        world2camera_matrix = torch.inverse(poses)  # Bx4x4

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'copy_rays_o': copy_rays['rays_o'],
            'copy_rays_d': copy_rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'perspective_matrix': perspective_matrix,
            'world2camera_matrix': world2camera_matrix,
        }

        return data


    def dataloader(self, batch_size=1):
        dataset = list(range(self.size))
        if self.opt.multi_gpu and self.training:
            sampler = DistributedSampler(dataset)
            # batch_size = batch_size * self.opt.world_size
            loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=0, drop_last=True)
            return loader
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        return loader