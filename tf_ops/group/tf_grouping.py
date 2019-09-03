import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so'))


def query_ball_point(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)


ops.NoGradient('QueryBallPoint')


def select_top_k(k, dist):
    '''
    Input:
        k: int32, number of k SMALLEST elements selected
        dist: (b,m,n) float32 array, distance matrix, m query points, n dataset points
    Output:
        idx: (b,m,n) int32 array, first k in n are indices to the top k
        dist_out: (b,m,n) float32 array, first k in n are the top k
    '''
    return grouping_module.selection_sort(dist, k)


ops.NoGradient('SelectionSort')


def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return grouping_module.group_point(points, idx)


@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]


def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    print(b, n, c, m)
    print(xyz1, (b, 1, n, c))
    xyz1 = tf.tile(tf.reshape(xyz1, (b, 1, n, c)), [1, m, 1, 1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b, m, 1, c)), [1, 1, n, 1])
    dist = tf.reduce_sum((xyz1 - xyz2) ** 2, -1)
    print(dist, k)
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])
    val = tf.slice(out, [0, 0, 0], [-1, -1, k])
    print(idx, val)
    # val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx


if __name__ == '__main__':
    import numpy as np
    import data_utils

    points_t = tf.placeholder(tf.float32, shape=(None, None, 3))
    landmarks_t = tf.placeholder(tf.float32, shape=(None, None, 3))
    idx, pts_cnt = query_ball_point(10, 64, points_t, landmarks_t)

    batch_indices = tf.tile(tf.reshape(tf.range(tf.shape(landmarks_t)[0]), (-1, 1, 1, 1)),
                            (1, tf.shape(landmarks_t)[1], 64, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(idx, axis=-1)], axis=-1)
    query_points = tf.gather_nd(points_t, indices)

    with tf.Session() as sess:
        points, landmarks = data_utils.load_face_data('F0001_AN01WH_F3D.xyz', '/data1/face_data/train', data_dim=3)
        noise = data_utils.get_noise(landmarks.shape, radius=5)
        init_marks = landmarks + noise

        points = np.expand_dims(points, axis=0)
        init_marks = np.expand_dims(init_marks[:-10], axis=0)

        query, pts_cnt_np = sess.run([query_points, pts_cnt], feed_dict={
            points_t: points,
            landmarks_t: init_marks
        })

        data_utils.save(query.reshape(-1, 3), '/home/meidai/下载/sample.xyz')
        print(pts_cnt_np)
