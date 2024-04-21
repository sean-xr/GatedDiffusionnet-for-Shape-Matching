import os
import sys
from itertools import permutations, combinations
import numpy as np

import torch
from torch.utils.data import Dataset
import geometry
import potpourri3d as pp3d
import utils_diffusionnet
import scipy


class DiffusionnetFaustDataset(Dataset):
    def __init__(self, root_dir, name="MPI-FAUST", train=True, k_eig=128, use_cache=True, op_cache_dir=None):

        # NOTE: These datasets are setup such that each dataset object always loads the entire dataset regardless of
        # train/test mode. The correspondence pair combinations are then set such that the train dataset only returns
        # train pairs, and the test dataset only returns test pairs. Be aware of this if you try to adapt the code
        # for any other purpose!

        self.train = train  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, name, "cache")
        self.op_cache_dir = op_cache_dir
        self.name = name

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.names_list = []

        # set combinations
        n_train = {'MPI-FAUST': 80}[self.name]
        if self.train:
            self.combinations = list(permutations(range(n_train), 2))
        else:
            self.combinations = list(combinations(range(n_train, n_train + 20), 2))
        print("length of dataset:", len(self.combinations))

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.names_list
                ) = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []

        # load faust data
        mesh_dirpath = os.path.join(self.root_dir, name, "ply")
        for fname in sorted(os.listdir(mesh_dirpath)):
            mesh_fullpath = os.path.join(mesh_dirpath, fname)
            mesh_files.append(mesh_fullpath)

        print("loading {} meshes".format(len(mesh_files)))



        # Load the actual files
        for iFile in range(len(mesh_files)):
            print("loading mesh " + str(mesh_files[iFile]))

            verts, faces = pp3d.read_mesh(mesh_files[iFile])

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit-area scale
            verts = geometry.normalize_positions(verts, faces=faces, scale_method='area')

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.names_list.append(os.path.basename(mesh_files[iFile]).split(".")[0])

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        self.hks_list = [geometry.compute_hks_autoscale(self.evals_list[i], self.evecs_list[i], 16)
                         for i in range(len(self.L_list))]

        # save to cache
        if use_cache:
            utils_diffusionnet.ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.names_list,
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = [
            self.verts_list[idx1],
            self.faces_list[idx1],
            self.frames_list[idx1],
            self.massvec_list[idx1],
            self.L_list[idx1],
            self.evals_list[idx1],
            self.evecs_list[idx1],
            self.gradX_list[idx1],
            self.gradY_list[idx1],
            self.hks_list[idx1],
            self.names_list[idx1],
        ]

        shape2 = [
            self.verts_list[idx2],
            self.faces_list[idx2],
            self.frames_list[idx2],
            self.massvec_list[idx2],
            self.L_list[idx2],
            self.evals_list[idx2],
            self.evecs_list[idx2],
            self.gradX_list[idx2],
            self.gradY_list[idx2],
            self.hks_list[idx2],
            self.names_list[idx2],
        ]
        return shape1, shape2


class DiffusionnetFaustDataset2(Dataset):
    def __init__(self, root_dir, name="MPI-FAUST", train=True, k_eig=128, use_cache=True, op_cache_dir=None):

        # NOTE: These datasets are setup such that each dataset object always loads the entire dataset regardless of
        # train/test mode. The correspondence pair combinations are then set such that the train dataset only returns
        # train pairs, and the test dataset only returns test pairs. Be aware of this if you try to adapt the code
        # for any other purpose!

        self.train = train  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, name, "cache")
        self.op_cache_dir = op_cache_dir
        self.name = name
        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.names_list = []
        self.basis_list = []
        # set combinations
        n_train = {'MPI-FAUST': 80}[self.name]
        if self.train:
            self.combinations = list(combinations(range(n_train), 2))
        else:
            self.combinations = list(combinations(range(n_train, n_train + 20), 2))
        print("length of dataset:", len(self.combinations))

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.names_list,
                    self.basis_list
                ) = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        basis_files = []
        # load faust data
        mesh_dirpath = os.path.join(self.root_dir, name, "ply")
        basis_dirpath = os.path.join(self.root_dir, name, "out")
        if not os.path.exists(basis_dirpath):
          os.makedirs(basis_dirpath)
        for fname in sorted(os.listdir(mesh_dirpath)):
            mesh_fullpath = os.path.join(mesh_dirpath, fname)
            mesh_files.append(mesh_fullpath)
            number = fname[7:-4]
            basis_fullpath = os.path.join(basis_dirpath, f"outtr_reg_{number}_elasticbasis.npy")
            basis_files.append(basis_fullpath)
        print("loading {} meshes".format(len(mesh_files)))

        # Load the actual files
        for iFile in range(len(mesh_files)):
            print("loading mesh " + str(mesh_files[iFile]))

            verts, faces = pp3d.read_mesh(mesh_files[iFile])
            basis = np.load(basis_files[iFile])
            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            basis = torch.from_numpy(basis)
            # center and unit-area scale
            verts = geometry.normalize_positions(verts, faces=faces, scale_method='area')

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.names_list.append(os.path.basename(mesh_files[iFile]).split(".")[0])
            self.basis_list.append(basis)

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        self.hks_list = [geometry.compute_hks_autoscale(self.evals_list[i], self.evecs_list[i], 16)
                         for i in range(len(self.L_list))]

        # save to cache
        if use_cache:
            utils_diffusionnet.ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.names_list,
                    self.basis_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = [
            self.verts_list[idx1],
            self.faces_list[idx1],
            self.frames_list[idx1],
            self.massvec_list[idx1],
            self.L_list[idx1],
            self.evals_list[idx1],
            self.evecs_list[idx1],
            self.gradX_list[idx1],
            self.gradY_list[idx1],
            self.hks_list[idx1],
            self.basis_list[idx1],
            self.names_list[idx1]
        ]

        shape2 = [
            self.verts_list[idx2],
            self.faces_list[idx2],
            self.frames_list[idx2],
            self.massvec_list[idx2],
            self.L_list[idx2],
            self.evals_list[idx2],
            self.evecs_list[idx2],
            self.gradX_list[idx2],
            self.gradY_list[idx2],
            self.hks_list[idx2],
            self.basis_list[idx2],
            self.names_list[idx2]
        ]
        return shape1, shape2


class DiffusionnetFaustDataset3(Dataset):
    def __init__(self, root_dir, name="MPI-FAUST", train=True, k_eig=128, use_cache=True, op_cache_dir=None):

        # NOTE: These datasets are setup such that each dataset object always loads the entire dataset regardless of
        # train/test mode. The correspondence pair combinations are then set such that the train dataset only returns
        # train pairs, and the test dataset only returns test pairs. Be aware of this if you try to adapt the code
        # for any other purpose!

        self.train = train  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, name, "cache")
        self.op_cache_dir = op_cache_dir
        self.mass_cache_dir = os.path.join(root_dir, name, "cache_mass")
        if not os.path.exists(self.mass_cache_dir):
            os.makedirs(self.mass_cache_dir, exist_ok=True)
        self.name = name
        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.names_list = []
        self.basis_list = []
        # set combinations
        n_train = {'MPI-FAUST': 80}[self.name]
        if self.train:
            self.combinations = list(combinations(range(n_train), 2))
        else:
            self.combinations = list(combinations(range(n_train, n_train + 20), 2))
        print("length of dataset:", len(self.combinations))

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            mass_cahce_list = [file for file in os.listdir(self.mass_cache_dir) if file.endswith(".npz")]
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache) and len(mass_cahce_list) == 100:
                print("  --> loading dataset from cache")
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.names_list,
                    self.basis_list
                ) = torch.load(load_cache)
                return
            print("  --> dataset or mass not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        basis_files = []
        # load faust data
        mesh_dirpath = os.path.join(self.root_dir, name, "ply")
        basis_dirpath = os.path.join(self.root_dir, name, "out")
        if not os.path.exists(basis_dirpath):
            os.makedirs(basis_dirpath)
        for fname in sorted(os.listdir(mesh_dirpath)):
            mesh_fullpath = os.path.join(mesh_dirpath, fname)
            mesh_files.append(mesh_fullpath)
            number = fname[7:-4]
            basis_fullpath = os.path.join(basis_dirpath, f"outtr_reg_{number}_elasticbasis.npy")
            basis_files.append(basis_fullpath)
        print("loading {} meshes".format(len(mesh_files)))

        # Load the actual files
        for iFile in range(len(mesh_files)):
            print("loading mesh " + str(mesh_files[iFile]))
            filename = os.path.basename(mesh_files[iFile]).split(".")[0]
            verts, faces = pp3d.read_mesh(mesh_files[iFile])
            basis = np.load(basis_files[iFile])

            # compute mass, sqrtmass and save one-by-one
            print(f"computing and saving mass{filename}.npz")
            mass = utils_diffusionnet.compute_mass(verts, faces)
            sqrtmass = scipy.sparse.diags(np.sqrt(mass.diagonal()))
            mass_np = mass.toarray().astype(np.float32)
            sqrtmass_np = sqrtmass.toarray().astype(np.float32)
            mass_save_path = os.path.join(self.mass_cache_dir, f"{filename}.npz")
            np.savez(
                mass_save_path,
                mass=mass_np,
                sqrtmass=sqrtmass_np
            )

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            basis = torch.from_numpy(basis)
            # center and unit-area scale
            verts = geometry.normalize_positions(verts, faces=faces, scale_method='area')

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.names_list.append(filename)
            self.basis_list.append(basis)

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        self.hks_list = [geometry.compute_hks_autoscale(self.evals_list[i], self.evecs_list[i], 16)
                         for i in range(len(self.L_list))]

        # save to cache
        if use_cache:
            utils_diffusionnet.ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.names_list,
                    self.basis_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]
        # reading mass w.r.t. their names
        name1, name2 = self.names_list[idx1], self.names_list[idx2]
        mass_save_path1 = os.path.join(self.mass_cache_dir, f"{name1}.npz")
        mass_save_path2 = os.path.join(self.mass_cache_dir, f"{name2}.npz")
        mass_data1 = np.load(mass_save_path1)
        mass_data2 = np.load(mass_save_path2)
        mass1, sqrtmass1 = mass_data1["mass"], mass_data1["sqrtmass"]
        mass2, sqrtmass2 = mass_data2["mass"], mass_data2["sqrtmass"]

        mass1, sqrtmass1 = torch.from_numpy(mass1).to(torch.float32), torch.from_numpy(sqrtmass1).to(torch.float32)
        mass2, sqrtmass2 = torch.from_numpy(mass2).to(torch.float32), torch.from_numpy(sqrtmass2).to(torch.float32)

        shape1 = [
            self.verts_list[idx1],
            self.faces_list[idx1],
            self.frames_list[idx1],
            self.massvec_list[idx1],
            self.L_list[idx1],
            self.evals_list[idx1],
            self.evecs_list[idx1],
            self.gradX_list[idx1],
            self.gradY_list[idx1],
            self.hks_list[idx1],
            self.names_list[idx1],
            self.basis_list[idx1],
            mass1,
            sqrtmass1
        ]

        shape2 = [
            self.verts_list[idx2],
            self.faces_list[idx2],
            self.frames_list[idx2],
            self.massvec_list[idx2],
            self.L_list[idx2],
            self.evals_list[idx2],
            self.evecs_list[idx2],
            self.gradX_list[idx2],
            self.gradY_list[idx2],
            self.hks_list[idx2],
            self.names_list[idx2],
            self.basis_list[idx2],
            mass2,
            sqrtmass2
        ]
        return shape1, shape2

class DiffusionnetRegressionDataset(Dataset):
    def __init__(self, root_dir, name="MPI-FAUST", train=True, k_eig=128, use_cache=True, op_cache_dir=None):

        # NOTE: These datasets are setup such that each dataset object always loads the entire dataset regardless of
        # train/test mode. The correspondence pair combinations are then set such that the train dataset only returns
        # train pairs, and the test dataset only returns test pairs. Be aware of this if you try to adapt the code
        # for any other purpose!

        self.train = train  # bool
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, name, "cache")
        self.op_cache_dir = op_cache_dir
        self.name = name
        self.k_eig = k_eig
        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.names_list = []
        self.basis_list = []
        # set combinations
        n_train = {'MPI-FAUST': 80}[self.name]
        if self.train:
            self.idx_list = range(n_train)
        else:
            self.idx_list = range(n_train, n_train + 20)
        print(f"training mode: {self.train}, length of dataset:", len(self.idx_list))

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.names_list,
                    self.basis_list
                ) = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        basis_files = []
        # load faust data
        mesh_dirpath = os.path.join(self.root_dir, name, "ply")
        basis_dirpath = os.path.join(self.root_dir, name, "out")
        if not os.path.exists(basis_dirpath):
            os.makedirs(basis_dirpath)
        for fname in sorted(os.listdir(mesh_dirpath)):
            mesh_fullpath = os.path.join(mesh_dirpath, fname)
            mesh_files.append(mesh_fullpath)
            number = fname[7:-4]
            basis_fullpath = os.path.join(basis_dirpath, f"outtr_reg_{number}_elasticbasis.npy")
            basis_files.append(basis_fullpath)
        print("loading {} meshes".format(len(mesh_files)))

        # Load the actual files
        for iFile in range(len(mesh_files)):
            print("loading mesh " + str(mesh_files[iFile]))

            verts, faces = pp3d.read_mesh(mesh_files[iFile])
            basis = np.load(basis_files[iFile])
            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            basis = torch.from_numpy(basis)
            # center and unit-area scale
            verts = geometry.normalize_positions(verts, faces=faces, scale_method='area')

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.names_list.append(os.path.basename(mesh_files[iFile]).split(".")[0])
            self.basis_list.append(basis)

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        self.hks_list = [geometry.compute_hks_autoscale(self.evals_list[i], self.evecs_list[i], 16)
                         for i in range(len(self.L_list))]

        # save to cache
        if use_cache:
            utils_diffusionnet.ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.names_list,
                    self.basis_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):

        shape = [
            self.verts_list[idx],
            self.faces_list[idx],
            self.frames_list[idx],
            self.massvec_list[idx],
            self.L_list[idx],
            self.evals_list[idx],
            self.evecs_list[idx],
            self.gradX_list[idx],
            self.gradY_list[idx],
            self.hks_list[idx],
            self.names_list[idx],
            self.basis_list[idx]
        ]
        return shape


