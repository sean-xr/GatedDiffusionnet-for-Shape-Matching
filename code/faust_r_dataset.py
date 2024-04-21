import os
import sys
from itertools import permutations, combinations
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import igl


def data_augmentation(pointA, pointB):
    """
    Input:
        pointA:(M, D)
        pointB:(M, D)
    Return:
        pointA:(M, D)
        pointB:(M, D)
    """
    theta = np.random.uniform(0, np.pi * 2)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointA[:, [0, 2]] = pointA[:, [0, 2]].dot(rotation_matrix)  # random rotation
    pointB[:, [0, 2]] = pointB[:, [0, 2]].dot(rotation_matrix)  # random rotation
    return pointA, pointB


def pc_normalize(pointA, pointB):
    """
        Input:
            pointA:(M, D)
            pointB:(M, D)
        Return:
            pointA:(M, D)
            pointB:(M, D)
        """
    centroidA = np.mean(pointA, axis=0)
    pointA = pointA - centroidA
    centroidB = np.mean(pointB, axis=0)
    pointB = pointB - centroidB
    # m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    # pc = pc / m
    return pointA, pointB


def farthest_point_sample(pointA, pointB, correspondence, npoint):
    """
    Input:
        pointA:(N, D)
        pointB:(N, D)
        correspondence:(N, N)
        npoint: int
    Return:
        pointA:(M, D)
        pointB:(M, D)
        correspondence: (M, M), identity after re-indexing
    """
    N, D = pointA.shape
    xyz = pointA[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest  # centroids:storing the indices in pointA
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)  # scalar
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    pointA = pointA[centroids.astype(np.int32)]
    indicesB = np.where(correspondence[centroids.astype(np.int32)] == 1)[1]
    pointB = pointB[indicesB.astype(np.int32)]
    correspondence = np.identity(pointA.shape[0], dtype=np.int32)
    return pointA, pointB, correspondence


def construct_corres(vts1, vts2):
    """
    Compute the correspondence mwtrix out of two vts files
    Input: vts1: ndarray, shape: (n,)
           vts2: ndarray, shape: (n,)
    Output: Correspondence: ndarray, shape: (n, n)
    """
    correspondence_array = (vts1[:, None] == vts2).astype(int)
    return correspondence_array


def get_corresponding_points(subset_A, A, B, C):
    # Find the indices of the subset points in the original array A
    indices_A = np.where(np.isin(A, subset_A).all(axis=1))[0]

    # Use the correspondence matrix to find the corresponding indices in array B
    indices_B = np.where(C[indices_A] == 1)

    # Get the corresponding points from array B
    corresponding_points_B = B[indices_B]

    return corresponding_points_B


def resize_verts(verts, target_shape):
    # Assuming target_shape[0] is greater than the current number of points
    diff = target_shape[0] - verts.shape[0]
    if diff > 0:
        # Add zero-filled rows to make up the difference
        zeros = np.zeros((diff, verts.shape[1]))
        verts = np.vstack([verts, zeros])
    elif diff < 0:
        # Trim rows to match the target shape
        verts = verts[:target_shape[0], :]
    return verts


class FaustRDataset(Dataset):
    """
    Providing pair-wise vertics and correspondence
    """

    def __init__(self, root_dir, name="faust", train=True, n_samples=2000, n_points=4000, augm=False, uniform=False, use_cache=True):

        self.train = train  # bool
        self.root_dir = root_dir
        self.n_samples = n_samples
        self.n_points = n_points
        self.augm = augm
        self.uniform = uniform
        self.cache_dir = os.path.join(root_dir, name, "cache")
        self.name = name
        self.target_shape = (5000, 3)

        # store in memory
        self.verts_list = []
        self.vts_list = []
        self.names_list = []

        # set combinations
        n_train = {'FAUST_r': 80, 'scape': 51}[self.name]

        if self.train:
            #self.combinations = list(permutations(range(n_train), 2))
            self.combinations = random.sample(list(combinations(range(n_train), 2)), k=self.n_samples)
        else:
            self.n_samples = 200 #small set of test pairs
            self.combinations = random.sample(list(combinations(range(n_train, n_train + 20), 2)), k=self.n_samples)

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.npz")
            test_cache = os.path.join(self.cache_dir, "test.npz")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                loaded_cache = np.load(load_cache)
                self.verts_list = loaded_cache['verts']
                self.vts_list = loaded_cache['vts']
                self.names_list = loaded_cache['names']
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        vts_files = []

        # load faust data
        mesh_dirpath = os.path.join(self.root_dir, name, "off")
        vts_dirpath = os.path.join(self.root_dir, name, "corres")

        for fname in sorted(os.listdir(mesh_dirpath)):
            mesh_fullpath = os.path.join(mesh_dirpath, fname)
            vts_fullpath = os.path.join(vts_dirpath, fname[:-4] + ".vts")
            mesh_files.append(mesh_fullpath)
            vts_files.append(vts_fullpath)

        print("loading {} meshes".format(len(mesh_files)))

        # Load the actual files
        for iFile in range(len(mesh_files)):
            print("loading mesh " + str(mesh_files[iFile]))

            verts, _, _ = igl.read_off(mesh_files[iFile])
            # ensure that the 'verts'have the target shape
            if verts.shape != self.target_shape:
                verts = resize_verts(verts, self.target_shape)
            vts_file = np.loadtxt(vts_files[iFile]).astype(int) - 1  # convert to 0-based indexing

            self.verts_list.append(verts)
            self.vts_list.append(vts_file)
            self.names_list.append(os.path.basename(mesh_files[iFile]).split(".")[0])

        for ind, labels in enumerate(self.vts_list):
            self.vts_list[ind] = labels

        # save to cache
        if use_cache:
            np.savez(load_cache, verts=self.verts_list, vts=self.vts_list, names=self.names_list)

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = [
            self.verts_list[idx1],
            self.vts_list[idx1],
            self.names_list[idx1]
        ]

        shape2 = [
            self.verts_list[idx2],
            self.vts_list[idx2],
            self.names_list[idx2]
        ]

        verts1, verts2 = shape1[0], shape2[0]
        vts1, vts2 = shape1[1], shape2[1]

        # construct the relative gt point correspondences between shape1 and shape2
        corres_array = construct_corres(vts1, vts2)

        if self.uniform: #when sampling, the correspondence array will become identity
            verts1, verts2, corres_array = farthest_point_sample(verts1, verts2, corres_array, npoint=self.n_points)
        if self.augm:# not tested yet
            verts1, verts2 = data_augmentation(verts1, verts2)

        vert1, verts2 = pc_normalize(verts1, verts2)

        return verts1, verts2, corres_array
