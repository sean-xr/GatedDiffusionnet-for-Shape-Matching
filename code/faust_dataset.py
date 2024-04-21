import os
from itertools import permutations, combinations
import random
from torch.utils.data import Dataset
import igl
import numpy as np
from scipy.spatial.transform import Rotation
import potpourri3d as pp3d
import torch


def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def random_rotate_points(point):
    """
    randomly rotate a (N, 3) shape point cloud around z axis
    pytorch version
    """
    rotation_angle = torch.rand(1) * 2 * torch.tensor([np.pi])
    rotation_matrix = torch.tensor([[torch.cos(rotation_angle), -torch.sin(rotation_angle), 0],
                                    [torch.sin(rotation_angle), torch.cos(rotation_angle), 0],
                                    [0, 0, 1]])
    rotated_point = torch.mm(point, rotation_matrix.t())
    return rotated_point


def normalize_positions(pos, faces=None, method='mean', scale_method='max_rad'):
    # center and unit-scale positions

    if method == 'mean':
        # center using the average point position
        pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    elif method == 'bbox':
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, dim=-2).values
        bbox_max = torch.max(pos, dim=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center.unsqueeze(-2)
    else:
        raise ValueError("unrecognized method")

    if scale_method == 'max_rad':
        scale = torch.max(norm(pos), dim=-1, keepdim=True).values.unsqueeze(-1)
        pos = pos / scale
    elif scale_method == 'area':
        if faces is None:
            raise ValueError("must pass faces for area normalization")
        coords = pos[faces]
        vec_A = coords[:, 1, :] - coords[:, 0, :]
        vec_B = coords[:, 2, :] - coords[:, 0, :]
        face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
        total_area = torch.sum(face_areas)
        scale = (1. / torch.sqrt(total_area))
        pos = pos * scale
    else:
        raise ValueError("unrecognized scale method")
    return pos


class FaustDataset(Dataset):
    """
    Providing pair-wise vertics and correspondence
    """

    def __init__(self, root_dir, name="MPI-FAUST", train=True, augm=False, use_cache=True):

        self.train = train  # bool
        self.root_dir = root_dir
        self.augm = augm
        self.cache_dir = os.path.join(root_dir, name, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        self.name = name

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.vts_list = []
        self.names_list = []

        # set combinations
        n_train = {'MPI-FAUST': 80, 'scape': 51}[self.name]

        if self.train:
            self.combinations = list(combinations(range(n_train), 2))
            # self.combinations = random.sample(list(combinations(range(n_train), 2)), k=self.n_samples)
        else:
            # self.n_samples = 32  # small set of test pairs
            # self.combinations = random.sample(list(combinations(range(n_train, n_train + 20), 2)), k=self.n_samples)
            self.combinations = list(combinations(range(n_train, n_train + 20), 2))

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.names_list = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []

        # load faust data
        mesh_dirpath = os.path.join(self.root_dir, name)

        for fname in sorted(os.listdir(mesh_dirpath)):
            if fname.endswith(".ply"):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                mesh_files.append(mesh_fullpath)
        print(f"loading {len(mesh_files)} meshes")

        # Load the actual files
        for iFile in range(len(mesh_files)):
            print("loading mesh " + str(mesh_files[iFile]))
            verts, faces = pp3d.read_mesh(mesh_files[iFile])

            # convert to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and scale
            verts = normalize_positions(verts, method='bbox')

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.names_list.append(os.path.basename(mesh_files[iFile]).split(".")[0])

        # save to cache
        if use_cache:
            torch.save((self.verts_list, self.faces_list, self.names_list), load_cache)

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = [
            self.verts_list[idx1],
            self.faces_list[idx1],
            self.names_list[idx1]
        ]

        shape2 = [
            self.verts_list[idx2],
            self.faces_list[idx2],
            self.names_list[idx2]
        ]

        verts1, verts2 = shape1[0], shape2[0]
        faces1, faces2 = shape1[1], shape2[1]
        name1, name2 = shape1[2], shape2[2]

        if self.augm:  # not tested yet
            verts1 = random_rotate_points(verts1)
            verts2 = random_rotate_points(verts2)

        return verts1, verts2
