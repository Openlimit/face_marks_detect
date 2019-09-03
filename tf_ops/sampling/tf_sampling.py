''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017. 
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))


def prob_sample(inp, inpr):
    '''
input:
    batch_size * ncategory float32
    batch_size * npoints   float32
returns:
    batch_size * npoints   int32
    '''
    return sampling_module.prob_sample(inp, inpr)


ops.NoGradient('ProbSample')


# TF1.0 API requires set shape in C++
# @tf.RegisterShape('ProbSample')
# def _prob_sample_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(2)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
def gather_point(inp, idx):
    '''
input:
    batch_size * ndataset * 3   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    float32
    '''
    return sampling_module.gather_point(inp, idx)


# @tf.RegisterShape('GatherPoint')
# def _gather_point_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op, out_g):
    inp = op.inputs[0]
    idx = op.inputs[1]
    return [sampling_module.gather_point_grad(inp, idx, out_g), None]


def farthest_point_sample(npoint, inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample(inp, npoint)


ops.NoGradient('FarthestPointSample')


def farthest_point_sample_x(npoint, inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample_x(inp, npoint)


ops.NoGradient('FarthestPointSampleX')

if __name__ == '__main__':
    import numpy as np
    import data_utils

    points = np.loadtxt('/home/meidai/下载/frgc_test/seg/eyes.xyz', dtype=np.float32)
    points = np.expand_dims(points[:, :3], axis=0)

    inp = tf.placeholder(tf.float32, shape=(None, None, 3))
    n = tf.placeholder(tf.int32, shape=(None,))
    test = farthest_point_sample_x(n, inp)

    with tf.Session() as sess:
        pt = sess.run(test, feed_dict={
            inp: points,
            n: np.empty((512,))
        })
        data_utils.save(points[0, pt[0]], '/home/meidai/下载/sample.xyz')
