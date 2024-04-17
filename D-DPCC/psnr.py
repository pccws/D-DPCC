import numpy as np
from scipy.spatial import cKDTree as KDTree

import open3d


def estimate_normals(points: np.ndarray, **kwds):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    if kwds:
        pcd.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(**kwds)
        )
    else:
        pcd.estimate_normals()
    return np.asarray(pcd.normals)


def psnr_d1(
    points_1: np.ndarray,
    points_2: np.ndarray,
    max_energy: float = None,
    workers: int = 4,
) -> float:
    # print("PSNR D1: points_1: {}, points_2: {}".format(points_1.shape, points_2.shape))
    tree_1 = KDTree(points_1, balanced_tree=False)
    tree_2 = KDTree(points_2, balanced_tree=False)

    distances_1, nb_index_1 = tree_2.query(points_1, workers=workers)
    distances_2, nb_index_2 = tree_1.query(points_2, workers=workers)

    mse_1 = np.mean(np.square(distances_1))
    mse_2 = np.mean(np.square(distances_2))

    # Calculate the Max Energy if is not given
    psnr_1, psnr_2 = 0, 0
    if not max_energy:
        max_energy_1 = np.amax(distances_1, axis=0)
        max_energy_2 = np.amax(distances_2, axis=0)
        psnr_1 = 10 * np.log10(3 * max_energy_1 * max_energy_1 / mse_1)
        psnr_2 = 10 * np.log10(3 * max_energy_2 * max_energy_2 / mse_2)
    else:
        psnr_1 = 10 * np.log10(3 * max_energy * max_energy / mse_1)
        psnr_2 = 10 * np.log10(3 * max_energy * max_energy / mse_2)
    return min(psnr_1, psnr_2)


def psnr_d2(
    points_1: np.ndarray,
    points_2: np.ndarray,
    normals_1: np.ndarray = None,
    normals_2: np.ndarray = None,
    max_energy: float = None,
    workers: int = 4,
) -> float:
    tree_1 = KDTree(points_1, balanced_tree=False)
    tree_2 = KDTree(points_2, balanced_tree=False)

    distances_1, nb_index_1 = tree_2.query(points_1, workers=workers)
    distances_2, nb_index_2 = tree_1.query(points_2, workers=workers)

    neighbors_1 = points_2[nb_index_1]
    neighbors_2 = points_1[nb_index_2]

    if not normals_1:
        normals_1 = estimate_normals(points_1)
    if not normals_2:
        normals_2 = estimate_normals(points_2)

    # row-wise inner product: np.sum(a*b, axis=1)
    mse_1 = np.mean(np.sum((points_1 - neighbors_1) * normals_1, axis=1) ** 2)
    mse_2 = np.mean(np.sum((points_2 - neighbors_2) * normals_2, axis=1) ** 2)

    # Calculate the Max Energy if is not given
    psnr_1, psnr_2 = 0, 0
    if not max_energy:
        max_energy_1 = np.amax(np.linalg.norm(points_1 - neighbors_1, axis=1))
        max_energy_2 = np.amax(np.linalg.norm(points_2 - neighbors_2, axis=1))
        psnr_1 = 10 * np.log10(3 * max_energy_1 * max_energy_1 / mse_1)
        psnr_2 = 10 * np.log10(3 * max_energy_2 * max_energy_2 / mse_2)
    else:
        psnr_1 = 10 * np.log10(3 * max_energy * max_energy / mse_1)
        psnr_2 = 10 * np.log10(3 * max_energy * max_energy / mse_2)
    return min(psnr_1, psnr_2)


def psnr_attr(points_1, attributes_1, points_2, attributes_2, max_energy, workers=6):
    def mse(x, y):
        return np.mean(np.square(x - y).flatten())

    # Find Nearest Neighbor
    tree1 = KDTree(points_1, balanced_tree=False)
    tree2 = KDTree(points_2, balanced_tree=False)

    _, idx2 = tree2.query(points_1, workers=workers)
    _, idx1 = tree1.query(points_2, workers=workers)

    # MSE
    mse_1 = mse(attributes_1, attributes_2[idx2])
    mse_2 = mse(attributes_2, attributes_1[idx1])

    final_mse = min(mse_1, mse_2)

    return 10 * np.log10(max_energy**2 / final_mse)
