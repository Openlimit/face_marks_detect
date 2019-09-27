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
    inner_points[:, :3], scale, centroid = data_utils.normalize_points(inner_points[:, :3])

    return inner_points, scale, centroid


def get_part_box(landmarks):
    box = np.zeros((4, 6), dtype=np.float32)

    eye = landmarks[27:35]
    box[0] = data_utils.compute_box(eye)
    box[0, 2] += (box[0, 5] / 2)
    box[0, 4] *= 2
    box[0, 5] *= 2

    eyebrow = landmarks[15:27]
    box[1] = data_utils.compute_box(eyebrow)
    box[1, 4] *= 1.5

    nose = landmarks[35:46]
    box[2] = data_utils.compute_box(nose)
    box[2, 2] += (box[2, 5] / 2)
    box[2, 3] *= 1.5
    box[2, 5] *= 2

    mouth = landmarks[46:64]
    box[3] = data_utils.compute_box(mouth)
    box[3, 4] *= 1.5
    box[3, 5] *= 2

    return box.flatten()


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
    points, landmarks = load_rawscan('/home/meidai/下载/0a7b03d339db49c6c0dbc75b95bb3b2dfile/mesh.bin',
                                     '/home/meidai/下载/0a7b03d339db49c6c0dbc75b95bb3b2dfile/marks.txt')
    # inner_points = seg_inner_face(points, landmarks)

    # np.savetxt('/home/meidai/下载/dfmesh.xyz', inner_points, fmt='%.6f')

    # un_pack('/data1/rawscan_data/tar', '/data1/rawscan_data/dir')

    box = get_part_box(landmarks)
    box = box.reshape(-1, 6)
    for i in range(box.shape[0]):
        center = box[i, :3]
        lwh = box[i, 3:]
        box_p = np.zeros((2, 3), dtype=np.float32)
        box_p[0] = center - lwh / 2
        box_p[1] = center + lwh / 2
        np.savetxt('/home/meidai/下载/0a7b03d339db49c6c0dbc75b95bb3b2dfile/{}.xyz'.format(i), box_p, fmt='%.6f')
