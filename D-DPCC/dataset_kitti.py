import numpy as np
import open3d
import torch
import torch.utils.data as data
from os.path import join
import MinkowskiEngine as ME
from pathlib import Path

from typing import Callable, Dict, List, Tuple, Union


def my_torch_unique(x, dim=0):
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index


class RefDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        split,
        bit=10,
        maximum=20475,
        type="train",
        scaling_factor=1,
        time_step=1,
        format="npy",
    ):
        self.maximum = maximum
        self.type = type
        self.scaling_factor = scaling_factor
        self.format = format
        sequence_list = [
            "soldier",
            "redandblack",
            "loot",
            "longdress",
            "andrew",
            "basketballplayer",
            "dancer",
            "david",
            "exercise",
            "phil",
            "queen",
            "ricardo",
            "sarah",
            "model",
        ]
        self.sequence_list = sequence_list
        start = [536, 1450, 1000, 1051, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1]
        end = [835, 1749, 1299, 1350, 317, 600, 600, 215, 600, 244, 249, 215, 206, 600]
        num = [end[i] - start[i] for i in range(len(start))]
        self.lookup = []
        for i in split:
            sequence_dir = join(root_dir, sequence_list[i] + "_ori")
            # sequence_dir = join(root_dir, sequence_list[i])
            file_prefix = sequence_list[i] + "_vox" + str(bit) + "_"
            file_subfix = "." + self.format
            if type == "train":
                s = start[i]
                e = int((end[i] - start[i]) * 0.95 + start[i])
            elif type == "val":
                s = int((end[i] - start[i]) * 0.95 + start[i])
                e = end[i] - time_step + 1
            else:
                s = start[i]
                e = end[i]
            for s in range(s, e):
                s1 = str(s + time_step).zfill(4)
                s0 = str(s).zfill(4)
                file_name0 = file_prefix + s0 + file_subfix
                file_name1 = file_prefix + s1 + file_subfix
                file_dir = join(sequence_dir, file_name0)
                file_dir1 = join(sequence_dir, file_name1)
                self.lookup.append([file_dir, file_dir1])

    def __getitem__(self, item):
        file_dir, file_dir1 = self.lookup[item]
        if self.format == "npy":
            p, p1 = np.load(file_dir), np.load(file_dir1)
        elif self.format == "ply":
            p = np.asarray(open3d.io.read_point_cloud(file_dir).points)
            p1 = np.asarray(open3d.io.read_point_cloud(file_dir1).points)
        pc = torch.tensor(p[:, :3]).cuda()
        pc1 = torch.tensor(p1[:, :3]).cuda()

        if self.scaling_factor != 1:
            pc = torch.unique(torch.floor(pc / self.scaling_factor), dim=0)
            pc1 = torch.unique(torch.floor(pc1 / self.scaling_factor), dim=0)
        xyz, point = pc, torch.ones_like(pc[:, :1])
        xyz1, point1 = pc1, torch.ones_like(pc1[:, :1])

        return xyz, point, xyz1, point1

    def __len__(self):
        return len(self.lookup)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError("No data in the batch")

    # coords, feats, labels = list(zip(*list_data))
    xyz, point, xyz1, point1 = list(zip(*list_data))

    xyz_batch = ME.utils.batched_coordinates(xyz)
    point_batch = torch.vstack(point).float()
    xyz1_batch = ME.utils.batched_coordinates(xyz1)
    point1_batch = torch.vstack(point1).float()
    return xyz_batch, point_batch, xyz1_batch, point1_batch


class KittiDataset(torch.utils.data.Dataset):
    SEQUENCES = {
        # This is the original split
        "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
        "valid": ["08"],
        "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
    }

    def __init__(
        self,
        root_dir: str,
        sequences: Union[int, str, List[Union[int, str]]],
        coords_transform: Callable = None,
        r_transform: Callable = None,
    ):
        # {{{ path
        self.root_dir_path = Path(root_dir).resolve()
        if not self.root_dir_path.is_dir():
            raise RuntimeError("{} is not a dir!".format(root_dir))
        # }}}

        # dealing with sequences number
        self.sequences: List[str] = KittiDataset.generate_sequences(sequences)
        self.files_path: List[Path] = []
        for sequence in self.sequences:
            sequence_root_path = (
                self.root_dir_path / "sequences" / sequence / "velodyne"
            )
            bin_files_path = list(sequence_root_path.glob("*.bin"))
            self.files_path += bin_files_path

        self.coords_transform = coords_transform
        self.r_transform = r_transform

    @staticmethod
    def generate_sequences(s: Union[int, str, List[Union[int, str]]]) -> List[str]:
        """return a list that looks like ['03', '00', '10']"""
        if isinstance(s, int):
            # 5 -> '05'
            return ["{:02}".format(s)]
        if isinstance(s, str):
            if s in {"train", "valid", "test"}:
                return KittiDataset.SEQUENCES[s]
            # "00", "21"
            if s.isdigit():
                n = int(s)
                if 0 <= n <= 21:
                    return KittiDataset.generate_sequences(n)
        if isinstance(s, list):
            ret = []
            for a in s:
                ret += KittiDataset.generate_sequences(a)
            return ret
        raise ValueError("Cannot parse the input!")

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, index):
        bin_file = self.files_path[index]
        points, remissions = self.read_kitti_file(bin_file)

        if self.coords_transform:
            points = self.coords_transform(points)
        if self.r_transform:
            remissions = self.r_transform(remissions)

        return points, remissions

    @staticmethod
    def read_kitti_file(bin_file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        bin_file_path = Path(bin_file).resolve()
        if not bin_file_path.is_file():
            raise RuntimeError("{} is not a file!".format(bin_file_path))
        raw_numbers = np.fromfile(str(bin_file_path), dtype=np.float32)
        # m * [x, y, z, r]
        raw_numbers = raw_numbers.reshape((-1, 4))
        # shape: (n, 3)
        points = raw_numbers[:, :3]
        # shape: (n, 1)
        remissions = raw_numbers[:, 3].reshape((-1, 1))

        return points, remissions


class Dataset(KittiDataset):
    def __init__(
        self,
        root_dir: str,
        sequences: Union[int, str, List[Union[int, str]]],
        coords_transform: Callable = None,
        r_transform: Callable = None,
        level: int = 10,
    ):
        self.level = level
        return super().__init__(root_dir, sequences, coords_transform, r_transform)

    def __len__(self):
        return len(self.files_path - 1)

    def __getitem__(self, index):
        points_1, remissions_1 = super().__getitem__(index)
        points_2, remissions_2 = super().__getitem__(index + 1)
        # numpy array -> torch tensor
        points_1 = torch.from_numpy(points_1).float().cuda()
        points_2 = torch.from_numpy(points_2).float().cuda()
        # unique
        points_1 = torch.unique(points_1, dim=0)
        points_2 = torch.unique(points_2, dim=0)
        # feature
        feature_1 = torch.ones_like(points_1[:, [0]])
        feature_2 = torch.ones_like(points_2[:, [0]])
        return points_1, feature_1, points_2, feature_2

    @staticmethod
    def quantify(
        points: torch.tensor, level: int, return_info: bool = True
    ) -> torch.tensor | Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """quantify the coordinates of points"""
        bbox_min = torch.min(points, dim=0)[0]
        bbox_max = torch.max(points, dim=0)[0]
        bbox_size = torch.max(bbox_max - bbox_min)
        voxel_size = bbox_size / (2**level)
        # normalize to [0, 1]
        voxels = torch.floor((points - bbox_min) / voxel_size)
        voxels = torch.clamp(voxels, min=0, max=2**level - 1)
        # voxels = torch.unique(voxels, sorted=False, dim=0)
        if not return_info:
            return voxels
        else:
            return voxels, bbox_min, voxel_size

    @staticmethod
    def unquantify(
        voxels: torch.tensor, bbox_min: torch.tensor, voxel_size: torch.tensor
    ):
        points = voxels * voxel_size + voxel_size * 0.5 + bbox_min
        return points

    @staticmethod
    def unique(voxels: torch.tensor, feats: torch.tensor):
        return_voxels, _, _, indices = my_torch_unique(voxels, dim=0)
        return_feats = feats[:, indices]
        return return_voxels, return_feats
