import tensorflow as tf
import numpy as np
import argparse
import os
import importlib
import sys

from models.PoseEstimator import PoseEstimator
from tools import md_utils, data_utils


def load_model(model_path, data_path):
    points, landmarks = md_utils.load_rawscan(os.path.join(data_path, 'mesh.bin'),
                                              os.path.join(data_path, 'marks.txt'))
    np.savetxt(os.path.join(data_path, 'mesh.xyz'), points[:, :3], fmt='%.6f')
    inner_points, scale, centroid = md_utils.seg_inner_face(points, landmarks)

    points_batch = np.zeros((32, 4096, 6), dtype=np.float32)
    for i in range(32):
        points_batch[i] = data_utils.select_point(inner_points, 4096)

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
        pts_fts = sess.graph.get_tensor_by_name('pts_fts:0')
        sample_num_real = sess.graph.get_tensor_by_name('sample_num_real:0')
        is_training = sess.graph.get_tensor_by_name('is_training:0')
        predicts = sess.graph.get_tensor_by_name('posenet_predicts/BiasAdd:0')

        result = sess.run(predicts, feed_dict={
            pts_fts: points_batch,
            is_training: False,
            sample_num_real: np.empty((1024,))
        })
        print(result)
        return result


def save_model(path):
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

    setting_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings')
    sys.path.append(setting_path)
    pose_setting = importlib.import_module(args.setting)
    predictor = PoseEstimator(pose_setting)
    predictor.restore()

    predictor.save_model(path)


if __name__ == '__main__':
    path = '/data1/face_data/save_folder/pose_setting_2019-09-03-14-29-44/model'
    # save_model(path)
    load_model(path, '/data1/rawscan_data/dir/0af4122fbaab5387afb162356a07740ffile')
