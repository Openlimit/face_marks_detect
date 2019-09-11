import os
import sys
import argparse
import importlib
import numpy as np

from models.PartBoxDetector import PartBoxDetector
from models.PartMarksDetector import PartMarksDetector
from models.PoseEstimator import PoseEstimator
from models.UVMarksDetector import UVMarksDetector

from tools import md_utils, pointfly, data_utils
from wly import icp


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


def eval_seg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/val',
                        help='Path to test set ground truth (.txt)', required=False)
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='seg_only_points_setting', help='Setting to use',
                        required=False)
    # parser.add_argument('--coarse_setting', '-cx', default='only_points_setting', help='Coarse Setting to use',
    #                     required=False)
    parser.add_argument('--partbox_setting', '-cx', default='partbox_setting', help='Coarse Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'setting')
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)
    partbox_setting = importlib.import_module(args.partbox_setting)

    partBox = PartBoxDetector(partbox_setting)
    partBox.restore()

    segMarks = PartMarksDetector(setting)
    segMarks.restore()
    segMarks.eval(args, partBox)


def train_partmarks():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/train',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/val',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='part_marks_setting', help='Setting to use',
                        required=False)
    parser.add_argument('--partbox_setting', '-cx', default='part_box_setting', help='Coarse Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)
    partbox_setting = importlib.import_module(args.partbox_setting)

    partBox = PartBoxDetector(partbox_setting)
    partBox.restore()

    partMarks = PartMarksDetector(setting)
    partMarks.train(args, partBox)


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


def train_uv_marks():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/uv_coord/train',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/uv_coord/val',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='uv_marks_setting', help='Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    predictor = UVMarksDetector(setting)
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


def test_uv_marks():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', default='/data1/face_data/uv_coord/train',
                        help='Path to training set ground truth (.txt)', required=False)
    parser.add_argument('--path_val', '-v', default='/data1/face_data/uv_coord/val',
                        help='Path to validation set ground truth (.txt)', required=False)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='/data1/face_data/save_folder',
                        help='Path to folder for saving check points and summary',
                        required=False)
    parser.add_argument('--setting', '-x', default='uv_marks_setting', help='Setting to use',
                        required=False)
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), 'settings')
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    predictor = UVMarksDetector(setting)
    predictor.restore()

    points, landmarks, points_org = data_utils.load_uv_data('M0026_HA02AE_F3D.uvxyz', '/data1/face_data/uv_coord/val',
                                                            train=False)
    result = predictor.test(points)

    err_uv = np.mean(np.sqrt(np.sum((landmarks[:, :2] - result) ** 2, axis=-1)))
    print(err_uv)
    np.savetxt('/home/meidai/下载/points.xyz', points[:, :3], fmt='%.6f')
    np.savetxt('/home/meidai/下载/landmarks.xyz', landmarks, fmt='%.6f')
    np.savetxt('/home/meidai/下载/uv.xyz', result, fmt='%.6f')

    _, idx = icp.nearest_neighbor(result, points_org[:, :2])
    predict = points_org[idx, 2:5]
    err_xyz = np.mean(np.sqrt(np.sum((landmarks[:, 2:] - predict) ** 2, axis=-1)))
    print(err_xyz)


if __name__ == '__main__':
    # train_pose_estimator()
    # test_pose_estimator('/data1/rawscan_data/dir/0af4122fbaab5387afb162356a07740ffile')
    # test_pose_estimator_batch('/data1/rawscan_data/test')
    # train_uv_marks()
    # test_uv_marks()
    train_partbox()
