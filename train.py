import os
import sys
import argparse
import importlib
import numpy as np

from models.PartBoxDetector import PartBoxDetector
from models.PartMarksDetector import PartMarksDetector
from models.PoseEstimator import PoseEstimator

from tools import md_utils, pointfly


def train_partbox():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/train_normal',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/val_normal',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='part_box_setting', help='Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    partbox_setting = importlib.import_module(args.setting)

    predictor = PartBoxDetector(partbox_setting)
    predictor.train(args)


def test_partbox(data_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/train_normal',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/val_normal',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='part_box_setting', help='Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    partbox_setting = importlib.import_module(args.setting)
    predictor = PartBoxDetector(partbox_setting)
    predictor.restore()

    points, landmarks = md_utils.load_rawscan(os.path.join(data_path, 'mesh.bin'),
                                              os.path.join(data_path, 'marks.txt'))
    inner_points, scale, centroid = md_utils.seg_inner_face(points, landmarks)
    # np.savetxt(os.path.join(data_path, 'mesh.xyz'), inner_points[:, :3], fmt='%.6f')

    box = predictor.test(inner_points)
    for i in range(box.shape[0]):
        center = box[i, :3]
        half = box[i, 3:] / 2
        box_points = np.zeros((2, 3), dtype=np.float32)
        box_points[0] = center - half
        box_points[1] = center + half
        np.savetxt(os.path.join(data_path, 'box{}.xyz'.format(i)), box_points, fmt='%.6f')


def train_partmarks():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/train_normal',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/val_normal',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='part_marks_setting', help='Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    partMarks = PartMarksDetector(setting)
    partMarks.train(args)


# def test_partmarks(data_path):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', '-t', default='/data1/face_data/train_normal',
#                         help='Path to training set ground truth (.txt)', required=False)
#     parser.add_argument('--path_val', '-v', default='/data1/face_data/val_normal',
#                         help='Path to validation set ground truth (.txt)', required=False)
#     parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
#     parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
#                         help='Path to folder for saving check points and summary',
#                         required=False)
#     parser.add_argument('--setting', '-x', default='part_marks_setting', help='Setting to use',
#                         required=False)
#     parser.add_argument('--box_setting', '-bx', default='part_box_setting', help='Setting to use',
#                         required=False)
#     args = parser.parse_args()
#
#     setting_path = os.path.join(os.path.dirname(__file__), 'settings')
#     sys.path.append(setting_path)
#     setting = importlib.import_module(args.setting)
#     box_setting = importlib.import_module(args.box_setting)
#
#     partBox = PartBoxDetector(box_setting)
#     partBox.restore()
#
#     partMarks = PartMarksDetector(setting)
#     partMarks.restore()
#
#     points, landmarks = md_utils.load_rawscan(os.path.join(data_path, 'mesh.bin'),
#                                               os.path.join(data_path, 'marks.txt'))
#     inner_points, scale, centroid = md_utils.seg_inner_face(points, landmarks)
#
#     points[:, :3] = (points[:, :3] - centroid) / scale
#     np.savetxt(os.path.join(data_path, 'mesh.xyz'), points[:, :3], fmt='%.6f')
#
#     boxes = partBox.test(inner_points)
#     boxes = boxes.reshape(-1, 6)
#     predicts, parts, centroids = partMarks.test(points, boxes)
#     for name in setting.part_list:
#         predicts[name] = predicts[name] + centroids[name]
#         np.savetxt(os.path.join(data_path, '{}_marks.xyz'.format(name)), predicts[name], fmt='%.6f')
#         np.savetxt(os.path.join(data_path, '{}.xyz'.format(name)), parts[name], fmt='%.6f')


def test_partmarks(data_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/train_normal',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/val_normal',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='part_marks_setting', help='Setting to use',
                        required=False)
    parser.add_argument('--box_setting', '-bx', default='part_box_setting', help='Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)
    box_setting = importlib.import_module(args.box_setting)

    partBox = PartBoxDetector(box_setting)
    partBox.restore()

    partMarks = PartMarksDetector(setting)
    partMarks.restore()

    points, landmarks = md_utils.load_rawscan(os.path.join(data_path, 'mesh.bin'),
                                              os.path.join(data_path, 'marks.txt'))
    np.savetxt(os.path.join(data_path, 'mesh.xyz'), points, fmt='%.6f')
    np.savetxt(os.path.join(data_path, 'marks.xyz'), landmarks, fmt='%.6f')

    inner_points, scale, centroid = md_utils.seg_inner_face(points, landmarks)
    np.savetxt(os.path.join(data_path, 'inner_points.xyz'), inner_points, fmt='%.6f')

    boxes = partBox.test(inner_points)
    boxes = boxes.reshape(-1, 6)

    boxes[:, :3] = boxes[:, :3] * scale + centroid
    boxes[:, 3:] *= scale

    predicts, parts, centroids = partMarks.test(points, boxes)
    for name in setting.part_list:
        predicts[name] = predicts[name] + centroids[name]
        parts[name][:, :3] = parts[name][:, :3] + centroids[name]
        np.savetxt(os.path.join(data_path, '{}_marks.xyz'.format(name)), predicts[name], fmt='%.6f')
        np.savetxt(os.path.join(data_path, '{}.xyz'.format(name)), parts[name], fmt='%.6f')


def train_pose_estimator():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/train_normal',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/val_normal',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='pose_setting', help='Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    pose_setting = importlib.import_module(args.setting)

    predictor = PoseEstimator(pose_setting)
    predictor.train(args)


def test_pose_estimator(data_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/train_normal',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/val_normal',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='pose_setting', help='Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    pose_setting = importlib.import_module(args.setting)

    points, landmarks = md_utils.load_rawscan(os.path.join(data_path, 'mesh.bin'),
                                              os.path.join(data_path, 'marks.txt'))
    np.savetxt(os.path.join(data_path, 'mesh.xyz'), points[:, :3], fmt='%.6f')
    inner_points, scale, centroid = md_utils.seg_inner_face(points, landmarks)
    np.savetxt(os.path.join(data_path, 'inner_points.xyz'), inner_points[:, :3], fmt='%.6f')

    predictor = PoseEstimator(pose_setting)
    predictor.restore()
    rxyz = predictor.test(inner_points)
    print(rxyz)
    rotation = pointfly.get_rotation_mtx(rxyz[0], rxyz[1], 0)
    trans = pointfly.get_transform_mtx(rotation, centroid)
    points = pointfly.transform_points(points[:, :3], trans)
    np.savetxt(os.path.join(data_path, 'mesh_rt.xyz'), points, fmt='%.6f')


def test_pose_estimator_batch(dir_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/train_normal',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/val_normal',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='pose_setting', help='Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    pose_setting = importlib.import_module(args.setting)
    predictor = PoseEstimator(pose_setting)
    predictor.restore()

    dirs = os.listdir(dir_path)
    for dir in dirs:
        data_path = os.path.join(dir_path, dir)
        print(data_path)
        points, landmarks = md_utils.load_rawscan(os.path.join(data_path, 'mesh.bin'),
                                                  os.path.join(data_path, 'marks.txt'))
        np.savetxt(os.path.join(data_path, 'mesh.xyz'), points[:, :3], fmt='%.6f')
        inner_points, scale, centroid = md_utils.seg_inner_face(points, landmarks)

        rxyz = predictor.test(inner_points)
        print(rxyz)
        rotation = pointfly.get_rotation_mtx(rxyz[0], rxyz[1], rxyz[2])
        trans = pointfly.get_transform_mtx(rotation, centroid)
        points = pointfly.transform_points(points[:, :3], trans)
        np.savetxt(os.path.join(data_path, 'mesh_rt.xyz'), points, fmt='%.6f')
        np.savetxt(os.path.join(data_path, 'rotation.txt'), trans, fmt='%.6f')


if __name__ == '__main__':
    # train_pose_estimator()
    test_pose_estimator('/home/meidai/下载/9656212361fcbb21f6e290194422f039')
    # test_pose_estimator_batch('/data1/rawscan_data/test')
    # train_partbox()
    # test_partbox('/home/meidai/下载/rawscan2')
    # train_partmarks()
    # test_partmarks('/home/meidai/下载/refine_rawscan/521fd5e7e4ff8e2b82a4920f09a3e11bfile (1)')