import os
import numpy as np

from tools import pointfly as pf


def normalize_points(pc, marks=None):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / scale

    if marks is not None:
        marks = marks - centroid
        marks = marks / scale
        return pc, marks, scale, centroid
    else:
        return pc, scale, centroid


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


def load_face_data(name, point_path, data_dim=None, need_mean=False,
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


def compute_box(part_marks, expand=0.2, with_noise=False):
    max_point = np.max(part_marks, axis=0)
    min_points = np.min(part_marks, axis=0)

    box = np.zeros((6,), dtype=np.float32)
    box[:3] = (max_point + min_points) / 2
    box[3:] = max_point - min_points

    if with_noise:
        for i in range(3):
            box[i] += (box[i + 3] * pf.gauss_clip(0, 0.1, 1))
            box[i + 3] *= (1.0 + expand + pf.gauss_clip(0, expand, 1))
    else:
        box[3:] *= (1.0 + expand)

    return box


def get_part_box(landmarks, with_noise=False):
    box = np.zeros((4, 6), dtype=np.float32)

    eye = landmarks[0:16]
    box[0] = compute_box(eye, with_noise=with_noise)
    box[0, 2] += (box[0, 5] / 2)
    box[0, 4] *= 2
    box[0, 5] *= 2

    eyebrow = landmarks[16:36]
    box[1] = compute_box(eyebrow, with_noise=with_noise)
    box[1, 4] *= 1.5

    nose = landmarks[-10:]
    box[2] = compute_box(nose, with_noise=with_noise)
    box[2, 2] += (box[2, 5] / 2)
    box[2, 3] *= 1.5
    box[2, 5] *= 2

    mouth = landmarks[48:68]
    box[3] = compute_box(mouth, with_noise=with_noise)
    box[3, 4] *= 1.5
    box[3, 5] *= 2

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


def seg_part(points, landmarks, box):
    center = box[:3]
    lwh = box[3:]
    min_point = center - lwh / 2
    max_point = center + lwh / 2

    idx1 = np.all(points[:, :3] <= max_point, axis=-1)
    idx2 = np.all(points[:, :3] >= min_point, axis=-1)
    idx = np.logical_and(idx1, idx2)

    part_points = np.copy(points[idx])
    centroid = np.mean(part_points[:, :3], axis=0)
    part_points[:, :3] = part_points[:, :3] - centroid
    part_marks = landmarks - centroid

    return part_points, part_marks, centroid


def seg_nose(points, landmarks, boxes):
    nose = landmarks[-10:]
    return seg_part(points, nose, boxes[2])


def seg_eye(points, landmarks, boxes):
    eye = landmarks[0:16]
    return seg_part(points, eye, boxes[0])


def seg_eyebrow(points, landmarks, boxes):
    eyebrow = landmarks[16:36]
    return seg_part(points, eyebrow, boxes[1])


def seg_mouth(points, landmarks, boxes):
    mouth = landmarks[48:68]
    return seg_part(points, mouth, boxes[3])


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
    inner_points[:, :3], landmarks, _, _ = normalize_points(inner_points[:, :3], marks=landmarks)

    return inner_points, landmarks


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

    points, landmarks = load_face_data('M0032_NE00WH_F3D.xyz', '/data1/face_data/val', data_dim=3)
    np.savetxt('/home/meidai/下载/data_utils/points.xyz', points, fmt='%.6f')

    box = get_part_box(landmarks, with_noise=False)
    box = box.reshape(-1, 6)
    for i in range(box.shape[0]):
        center = box[i, :3]
        lwh = box[i, 3:]
        box_p = np.zeros((2, 3), dtype=np.float32)
        box_p[0] = center - lwh / 2
        box_p[1] = center + lwh / 2
        np.savetxt('/home/meidai/下载/data_utils/{}.xyz'.format(i), box_p, fmt='%.6f')

    eye, eye_marks, _, _ = seg_eye(points, landmarks, box)
    np.savetxt('/home/meidai/下载/data_utils/eye.xyz', eye, fmt='%.6f')
    np.savetxt('/home/meidai/下载/data_utils/eye_marks.xyz', eye_marks, fmt='%.6f')

    eyebrow, eyebrow_marks, _, _ = seg_eyebrow(points, landmarks, box)
    np.savetxt('/home/meidai/下载/data_utils/eyebrow.xyz', eyebrow, fmt='%.6f')
    np.savetxt('/home/meidai/下载/data_utils/eyebrow_marks.xyz', eyebrow_marks, fmt='%.6f')

    nose, nose_marks, _, _ = seg_nose(points, landmarks, box)
    np.savetxt('/home/meidai/下载/data_utils/nose.xyz', nose, fmt='%.6f')
    np.savetxt('/home/meidai/下载/data_utils/nose_marks.xyz', nose_marks, fmt='%.6f')

    mouth, mouth_marks, _, _ = seg_mouth(points, landmarks, box)
    np.savetxt('/home/meidai/下载/data_utils/mouth.xyz', mouth, fmt='%.6f')
    np.savetxt('/home/meidai/下载/data_utils/mouth_marks.xyz', mouth_marks, fmt='%.6f')
