import os
import numpy as np
import random

from tools import pointfly as pf


def normalize_points(pc, marks=None, return_centroid=False):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / scale

    if marks is not None:
        marks = marks - centroid
        marks = marks / scale
        if return_centroid:
            return pc, marks, scale, centroid
        else:
            return pc, marks, scale
    else:
        if return_centroid:
            return pc, scale, centroid
        else:
            return pc, scale


def load_uv_data(name, point_path, data_dim=None,
                 landmarks_path='/data1/face_data/uv_coord/landmarks_nose', train=True):
    face_path = os.path.join(point_path, name)
    marks_path = os.path.join(landmarks_path, name)

    points_org = np.loadtxt(face_path, dtype=np.float32)
    landmarks = np.loadtxt(marks_path, dtype=np.float32)

    points = np.zeros((points_org.shape[0], points_org.shape[1] + 1), dtype=np.float32)
    points[:, :2] = points_org[:, :2]
    points[:, 3:] = points_org[:, 2:]

    mean = np.mean(points[:, :2], axis=0)
    points[:, :2] -= mean
    landmarks[:, :2] -= mean

    points[:, 3:6], _ = normalize_points(points[:, 3:6])

    if data_dim is not None:
        points = points[:, :data_dim]

    if train:
        landmarks = landmarks[:, :2]
        return points, landmarks
    else:
        points_org[:, :2] -= mean
        return points, landmarks, points_org


def load_ear_data(name, point_path, need_mean=False, landmarks_path='/data1/ear_data/ear_marks'):
    ear_path = os.path.join(point_path, name)
    ear_point_path = os.path.join(landmarks_path, name[:-4] + '_marks.xyz')
    ear = np.loadtxt(ear_path, dtype=np.float32)
    ear_point = np.loadtxt(ear_point_path, dtype=np.float32).reshape((-1, 3))

    mean = np.mean(ear[:, :3], axis=0)
    ear[:, :3] -= mean
    ear_point -= mean

    if need_mean:
        return ear, ear_point, mean
    else:
        return ear, ear_point


def load_face_data(name, point_path, data_dim=None, need_mean=False, normalize=False,
                   landmarks_path='/data1/face_data/landmarks',
                   nosemarks_path='/data1/face_data/landmarks_nose'):
    face_path = os.path.join(point_path, name)
    marks_path = os.path.join(landmarks_path, name[:-3] + 'bnd')
    nose_marks_path = os.path.join(nosemarks_path, name[:-3] + 'bnd')

    points = np.loadtxt(face_path, dtype=np.float32)
    landmarks = np.loadtxt(marks_path, dtype=np.float32)
    nose_marks = np.loadtxt(nose_marks_path, dtype=np.float32)
    landmarks = np.row_stack((landmarks, nose_marks))

    mean = np.mean(points[:, :3], axis=0)
    points[:, :3] -= mean
    landmarks -= mean

    if normalize:
        points[:, :3], landmarks, r = normalize_points(points[:, :3], marks=landmarks)

    if data_dim is not None:
        points = points[:, :data_dim]

    if need_mean:
        return points, landmarks, mean
    else:
        return points, landmarks


def face_augment(points, landmarks, xform, range=None, with_normal=False):
    points_xformed = np.copy(points)
    points_xformed[:, :3] = np.dot(points_xformed[:, :3], xform)
    landmarks_xformed = np.dot(landmarks, xform)

    if with_normal:
        points_xformed[:, 3:6] = np.dot(points_xformed[:, 3:6], xform)
        norm = np.linalg.norm(points_xformed[:, 3:6], axis=-1)
        norm = norm.reshape(-1, 1)
        points_xformed[:, 3:6] = points_xformed[:, 3:6] / norm

    if range is None:
        return points_xformed, landmarks_xformed
    N, C = points_xformed[:, :3].shape
    jittered_data = np.clip(range * np.random.randn(N, C), - 5 * range, 5 * range)
    points_xformed[:, :3] += jittered_data
    return points_xformed, landmarks_xformed


def augment(points, xform, range=None, with_normal=False):
    points_xformed = np.copy(points)
    points_xformed[:, :3] = np.dot(points_xformed[:, :3], xform)

    if with_normal:
        points_xformed[:, 3:6] = np.dot(points_xformed[:, 3:6], xform)

    if range is None:
        return points_xformed
    N, C = points_xformed[:, :3].shape
    jittered_data = np.clip(range * np.random.randn(N, C), - 5 * range, 5 * range)
    points_xformed[:, :3] += jittered_data
    return points_xformed


def get_noise(shape, radius=5):
    s = shape[:-1]
    x = np.random.randn(*s)
    y = np.random.randn(*s)
    z = np.random.randn(*s)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    noise = np.stack((x / r, y / r, z / r), axis=-1)
    k = np.random.rand(*noise.shape) * radius
    return noise * k


def get_inner_marks(landmarks):
    inner_marks = np.row_stack((landmarks[:36], landmarks[-10:], landmarks[48:68]))
    return inner_marks


def get_contour_marks(landmarks):
    return landmarks[68:83].flatten()


def get_part_marks(landmarks):
    return landmarks.flatten()


def compute_box(part_marks, expand=0.2):
    max_point = np.max(part_marks, axis=0)
    min_points = np.min(part_marks, axis=0)

    box = np.zeros((6,), dtype=np.float32)
    box[:3] = (max_point + min_points) / 2
    box[3:] = (max_point - min_points) * (1.0 + expand)
    return box


def get_part_box(landmarks):
    box = np.zeros((6, 6), dtype=np.float32)

    left_eye = landmarks[0:8]
    box[0] = compute_box(left_eye)
    box[0, 3] += 5
    box[0, 4] += 10

    right_eye = landmarks[8:16]
    box[1] = compute_box(right_eye)
    box[1, 3] += 5
    box[1, 4] += 10

    left_eyebrow = landmarks[16:26]
    box[2] = compute_box(left_eyebrow)
    box[2, 4] += 5

    right_eyebrow = landmarks[26:36]
    box[3] = compute_box(right_eyebrow)
    box[3, 4] += 5

    nose = landmarks[-10:]
    box[4] = compute_box(nose)
    box[4, 2] += (box[4, 5] / 2)
    box[4, 5] *= 2

    mouth = landmarks[48:68]
    box[5] = compute_box(mouth)
    box[5, 4] += 10

    return box.flatten()


def marks66_to_93(marks66):
    marks93 = np.zeros((93, 3), dtype=np.float32)
    marks93[:36] = marks66[:36]
    marks93[48:68] = marks66[46:66]
    marks93[-10:] = marks66[36:46]
    return marks93


def select_point(points, npoints, labels=None):
    if points.shape[0] < npoints:
        choice1 = np.random.choice(points.shape[0], points.shape[0], replace=False)
        choice2 = np.random.choice(points.shape[0], npoints - points.shape[0], replace=True)
        points = np.row_stack((points[choice1], points[choice2]))
        if labels is not None:
            labels = np.concatenate((labels[choice1], labels[choice2]))
    else:
        choice = np.random.choice(points.shape[0], npoints, replace=False)
        points = points[choice]
        if labels is not None:
            labels = labels[choice]

    if labels is not None:
        return points, labels
    else:
        return points


def select_one_of_two_parts(points_list, landmarks_list):
    idx = random.randint(0, 1)
    return points_list[idx], landmarks_list[idx]


def seg_nose(points, landmarks, expand=0.2, need_mean=False, box=None):
    nose = landmarks[-10:]

    if box is None:
        up = np.max(nose[:, 1])
        down = np.min(nose[:, 1])
        left = np.min(nose[:, 0])
        right = np.max(nose[:, 0])

        expand_h = (up - down) * expand
        expand_w = (right - left) * expand
        up += expand_h
        down -= expand_h
        right += expand_w
        left -= expand_w

        idx = np.where(
            (points[:, 1] >= down) & (points[:, 1] <= up)
            & (points[:, 0] >= left) & (points[:, 0] <= right))

    else:
        idx1 = np.all(points[:, :3] < box[8], axis=-1)
        idx2 = np.all(points[:, :3] > box[9], axis=-1)
        idx = np.logical_and(idx1, idx2)

    part_points = np.copy(points[idx])
    mean = np.mean(part_points[:, :3], axis=0)
    part_points[:, :3] -= mean
    part_marks = nose - mean

    if need_mean:
        return part_points, part_marks, mean
    else:
        return part_points, part_marks


def seg_eye(points, landmarks, expand=0.2, need_mean=False, box=None):
    left_eye = landmarks[0:8]
    right_eye = landmarks[8:16]

    parts = []
    parts_lk = []
    means = []
    for i, marks in enumerate([left_eye, right_eye]):
        if box is None:
            up = np.max(marks[:, 1])
            down = np.min(marks[:, 1])
            left = np.min(marks[:, 0])
            right = np.max(marks[:, 0])

            expand_h = (up - down) * expand
            expand_w = (right - left) * expand
            up += expand_h
            down -= expand_h
            right += expand_w
            left -= expand_w

            idx = np.where(
                (points[:, 1] >= down) & (points[:, 1] <= up)
                & (points[:, 0] >= left) & (points[:, 0] <= right))
        else:
            idx1 = np.all(points[:, :3] < box[i * 2], axis=-1)
            idx2 = np.all(points[:, :3] > box[i * 2 + 1], axis=-1)
            idx = np.logical_and(idx1, idx2)

        part_points = np.copy(points[idx])
        mean = np.mean(part_points[:, :3], axis=0)
        part_points[:, :3] -= mean
        part_marks = marks - mean

        if i == 1:
            part_points[:, 0] = -part_points[:, 0]
            part_marks[:, 0] = -part_marks[:, 0]

        parts.append(part_points)
        parts_lk.append(part_marks)
        means.append(mean)

    if need_mean:
        return parts, parts_lk, means
    else:
        return parts, parts_lk


def seg_eyebrow(points, landmarks, expand=0.2, need_mean=False, box=None):
    left_eyebrow = landmarks[16:26]
    right_eyebrow = landmarks[26:36]

    parts = []
    parts_lk = []
    means = []
    for i, marks in enumerate([left_eyebrow, right_eyebrow]):
        if box is None:
            up = np.max(marks[:, 1])
            down = np.min(marks[:, 1])
            left = np.min(marks[:, 0])
            right = np.max(marks[:, 0])

            expand_h = (up - down) * expand
            expand_w = (right - left) * expand
            up += expand_h
            down -= expand_h
            right += expand_w
            left -= expand_w

            idx = np.where(
                (points[:, 1] >= down) & (points[:, 1] <= up)
                & (points[:, 0] >= left) & (points[:, 0] <= right))
        else:
            idx1 = np.all(points[:, :3] < box[i * 2 + 4], axis=-1)
            idx2 = np.all(points[:, :3] > box[i * 2 + 5], axis=-1)
            idx = np.logical_and(idx1, idx2)

        part_points = np.copy(points[idx])
        mean = np.mean(part_points[:, :3], axis=0)
        part_points[:, :3] -= mean
        part_marks = marks - mean

        if i == 1:
            part_points[:, 0] = -part_points[:, 0]
            part_marks[:, 0] = -part_marks[:, 0]

        parts.append(part_points)
        parts_lk.append(part_marks)
        means.append(mean)

    if need_mean:
        return parts, parts_lk, means
    else:
        return parts, parts_lk


def seg_mouth(points, landmarks, expand=0.2, need_mean=False, box=None):
    mouth = landmarks[48:68]

    if box is None:
        up = np.max(mouth[:, 1])
        down = np.min(mouth[:, 1])
        left = np.min(mouth[:, 0])
        right = np.max(mouth[:, 0])

        expand_h = (up - down) * expand
        expand_w = (right - left) * expand
        up += expand_h
        down -= expand_h
        right += expand_w
        left -= expand_w

        idx = np.where(
            (points[:, 1] >= down) & (points[:, 1] <= up)
            & (points[:, 0] >= left) & (points[:, 0] <= right))

    else:
        idx1 = np.all(points[:, :3] < box[10], axis=-1)
        idx2 = np.all(points[:, :3] > box[11], axis=-1)
        idx = np.logical_and(idx1, idx2)

    part_points = np.copy(points[idx])
    mean = np.mean(part_points[:, :3], axis=0)
    part_points[:, :3] -= mean
    part_marks = mouth - mean

    if need_mean:
        return part_points, part_marks, mean
    else:
        return part_points, part_marks


def seg_inner_face(points, landmarks, expand=0.5):
    inner_marks = get_inner_marks(landmarks)

    max_point = np.max(inner_marks, axis=0)
    min_point = np.min(inner_marks, axis=0)

    expand_d = max_point - min_point
    s1 = np.array([pf.positive_uniform(expand), pf.positive_uniform(expand), pf.positive_uniform(expand)])
    s2 = np.array([pf.positive_uniform(expand), pf.positive_uniform(expand), pf.positive_uniform(expand)])

    max_point += expand_d * s1
    min_point -= expand_d * s2

    # 把鼻子包含起来
    max_point[2] += 5

    idx1 = np.all(points[:, :3] < max_point, axis=-1)
    idx2 = np.all(points[:, :3] > min_point, axis=-1)
    idx = np.logical_and(idx1, idx2)

    inner_points = np.copy(points[idx])
    inner_points[:, :3], _ = normalize_points(inner_points[:, :3])

    return inner_points


def cal_rotate_matrix(a, b):
    rot_axis = np.cross(b, a)
    rot_angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    rot_mat = np.zeros((3, 3), dtype=np.float32)

    rot_axis /= np.linalg.norm(rot_axis)

    rot_mat[0, 0] = np.cos(rot_angle) + rot_axis[0] * rot_axis[0] * (1 - np.cos(rot_angle))
    rot_mat[0, 1] = rot_axis[0] * rot_axis[1] * (1 - np.cos(rot_angle)) - rot_axis[2] * np.sin(rot_angle)
    rot_mat[0, 2] = rot_axis[1] * np.sin(rot_angle) + rot_axis[0] * rot_axis[2] * (1 - np.cos(rot_angle))

    rot_mat[1, 0] = rot_axis[2] * np.sin(rot_angle) + rot_axis[0] * rot_axis[1] * (1 - np.cos(rot_angle))
    rot_mat[1, 1] = np.cos(rot_angle) + rot_axis[1] * rot_axis[1] * (1 - np.cos(rot_angle))
    rot_mat[1, 2] = -rot_axis[0] * np.sin(rot_angle) + rot_axis[1] * rot_axis[2] * (1 - np.cos(rot_angle))

    rot_mat[2, 0] = -rot_axis[1] * np.sin(rot_angle) + rot_axis[0] * rot_axis[2] * (1 - np.cos(rot_angle))
    rot_mat[2, 1] = rot_axis[0] * np.sin(rot_angle) + rot_axis[1] * rot_axis[2] * (1 - np.cos(rot_angle))
    rot_mat[2, 2] = np.cos(rot_angle) + rot_axis[2] * rot_axis[2] * (1 - np.cos(rot_angle))

    return rot_mat


if __name__ == '__main__':
    # import math

    # points, landmarks = load_face_data('M0026_AN01AE_F3D.xyz', '/data1/face_data/val', data_dim=3)
    # rxyz, rotation = pf.get_rotation(rotation_range=[math.pi / 18, math.pi / 18, math.pi / 36, 'g'],
    #                                  order='rxyz')
    # print(rxyz)
    # points, landmarks = face_augment(points, landmarks, rotation)
    # inner_points = seg_inner_face(points, landmarks)
    # np.savetxt('/home/meidai/下载/inner_points.xyz', inner_points, fmt='%.6f')

    # points, landmarks = load_uv_data('F0002_HA01BL_F3D.uvxyz', '/data1/face_data/uv_coord/train')
    # np.savetxt('/home/meidai/下载/points.xyz', points[:, :6], fmt='%.6f')
    points, landmarks = load_face_data('M0026_AN01AE_F3D.xyz', '/data1/face_data/val', data_dim=3)
    np.savetxt('/home/meidai/下载/points.xyz', points, fmt='%.6f')

    box = get_part_box(landmarks)
    box = box.reshape(-1, 6)
    for i in range(box.shape[0]):
        center = box[i, :3]
        lwh = box[i, 3:]
        box_p = np.zeros((2, 3), dtype=np.float32)
        box_p[0] = center - lwh / 2
        box_p[1] = center + lwh / 2
        np.savetxt('/home/meidai/下载/{}.xyz'.format(i), box_p, fmt='%.6f')
