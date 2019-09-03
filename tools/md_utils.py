import numpy as np
import os
import tarfile
from wly import mesh

from tools import data_utils


def load_rawscan(bin_path, marks_path, with_normal=True):
    face = mesh.Mesh()
    face.init_from_bin(bin_path)
    if with_normal:
        face.compute_normal()
        points = np.column_stack([face.points, face.normals])
    else:
        points = face.points

    landmarks = np.loadtxt(marks_path, dtype=np.float32, skiprows=1)
    return points, landmarks


def get_inner_marks(landmarks):
    return landmarks[15:75]


def seg_inner_face(points, landmarks, expand=0.1):
    inner_marks = get_inner_marks(landmarks)

    max_point = np.max(inner_marks, axis=0)
    min_point = np.min(inner_marks, axis=0)

    expand_p = (max_point - min_point) * expand
    max_point += expand_p
    min_point -= expand_p

    idx1 = np.all(points[:, :3] < max_point, axis=-1)
    idx2 = np.all(points[:, :3] > min_point, axis=-1)
    idx = np.logical_and(idx1, idx2)

    inner_points = np.copy(points[idx])
    inner_points[:, :3], scale, centroid = data_utils.normalize_points(inner_points[:, :3], return_centroid=True)

    return inner_points, scale, centroid


def un_pack(path, dir):
    files = os.listdir(path)
    for name in files:
        print(name)
        file_path = os.path.join(path, name)
        dir_path = os.path.join(dir, name)
        f = tarfile.open(file_path, 'r')
        f.extractall(dir_path)
        f.close()


if __name__ == '__main__':
    # points, landmarks = load_rawscan('/home/meidai/下载/dfmesh.bin', '/home/meidai/下载/marks.txt')
    # inner_points = seg_inner_face(points, landmarks)
    #
    # np.savetxt('/home/meidai/下载/dfmesh.xyz', inner_points, fmt='%.6f')

    un_pack('/data1/rawscan_data/tar', '/data1/rawscan_data/dir')
