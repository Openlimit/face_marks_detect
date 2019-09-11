import os
import sys
import random
import math
from datetime import datetime
import numpy as np
from tools import pointfly as pf
import tensorflow as tf
from models.model import PointCNN
from tools import data_utils


class UVMarksDetector(object):
    def __init__(self, setting):
        self.setting = setting
        if 'data_load_func' not in self.setting.__dict__:
            self.setting.__dict__['data_load_func'] = data_utils.load_face_data

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.sample_num_real = tf.placeholder(tf.int32, shape=(None,), name='sample_num_real')
            self.pts_fts = tf.placeholder(tf.float32, shape=(None, None, setting.data_dim), name='pts_fts')

            features_sampled = None
            if setting.data_dim > 3:
                points_sampled, features_sampled = tf.split(self.pts_fts,
                                                            [3, setting.data_dim - 3],
                                                            axis=-1,
                                                            name='split_points_features')
            else:
                points_sampled = self.pts_fts

            # 重采样
            indices = pf.farthest_point_sample_x(points_sampled, self.sample_num_real)
            points_sampled_fps = tf.gather_nd(points_sampled, indices)
            features_sampled_fps = tf.gather_nd(features_sampled, indices) if features_sampled is not None else None

            net = PointCNN(points_sampled_fps, features_sampled_fps, self.is_training, setting,
                           scope='UVMarksDetector', out_shape=(setting.label_dim // 2, 2))
            self.predicts = net.predicts

    def save_model(self, export_path):
        with self.graph.as_default():
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            tensor_info_pts_fts = tf.saved_model.utils.build_tensor_info(self.pts_fts)
            tensor_info_sample_num_real = tf.saved_model.utils.build_tensor_info(self.sample_num_real)
            tensor_info_is_training = tf.saved_model.utils.build_tensor_info(self.is_training)
            tensor_info_predicts = tf.saved_model.utils.build_tensor_info(self.predicts)

            pose_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'pts_fts': tensor_info_pts_fts,
                            'sample_num_real': tensor_info_sample_num_real,
                            'is_training': tensor_info_is_training},
                    outputs={'predicts': tensor_info_predicts},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                self.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'pose_estimator': pose_signature
                })

            builder.save(as_text=True)
            print('Done exporting!')

    def restore(self):
        with self.graph.as_default():
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(self.sess, self.setting.load_ckpt)
            print('Checkpoint loaded from {}!'.format(self.setting.load_ckpt))

    def train(self, args):
        time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        root_folder = os.path.join(args.save_folder, '%s_%s' % (args.setting, time_string))
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)
        sys.stdout = open(os.path.join(root_folder, 'log.txt'), 'w')
        print(args)

        setting = self.setting

        # Prepare inputs
        train_names = os.listdir(args.path)
        val_names = os.listdir(args.path_val)
        random.shuffle(train_names)
        train_data = {}
        val_data = {}

        num_train = len(train_names)
        num_val = len(val_names)
        print('{}-{:d}/{:d} training/validation samples.'.format(datetime.now(), num_train, num_val))
        batch_num = (num_train * setting.num_epochs + setting.batch_size - 1) // setting.batch_size
        print('{}-{:d} training batches.'.format(datetime.now(), batch_num))
        batch_num_val = math.ceil(num_val / setting.batch_size)
        print('{}-{:d} testing batches per test.'.format(datetime.now(), batch_num_val))

        with self.graph.as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')
            labels = tf.placeholder(tf.float32, shape=(None, setting.label_dim // 2, 2), name='labels')
            loss_op = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(labels, self.predicts)), axis=-1))

            lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                                   setting.decay_rate, staircase=True)
            lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
            reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
            if setting.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
            elif setting.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum,
                                                       use_nesterov=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            saver = tf.train.Saver(max_to_keep=None)
            folder_ckpt = os.path.join(root_folder, 'ckpts')
            if not os.path.exists(folder_ckpt):
                os.makedirs(folder_ckpt)
            parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
            print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

            self.sess.run(init_op)
            for batch_idx_train in range(batch_num):
                if (batch_idx_train % setting.step_val == 0 and (batch_idx_train != 0 or args.load_ckpt is not None)) \
                        or batch_idx_train == batch_num - 1:
                    ######################################################################
                    # Validation
                    filename_ckpt = os.path.join(folder_ckpt, 'iter')
                    saver.save(self.sess, filename_ckpt, global_step=global_step)
                    print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))

                    avg_loss = 0
                    for batch_val_idx in range(batch_num_val):
                        start_idx = setting.batch_size * batch_val_idx
                        end_idx = min(start_idx + setting.batch_size, num_val)
                        batch_size_val = end_idx - start_idx
                        name_batch = val_names[start_idx:end_idx]

                        points_batch = np.zeros((batch_size_val, setting.origin_num, setting.data_dim),
                                                dtype=np.float32)
                        labels_batch = np.zeros((batch_size_val, setting.label_dim // 2, 2), dtype=np.float32)
                        for i, data_name in enumerate(name_batch):
                            if data_name not in val_data:
                                points, landmarks = self.setting.data_load_func(data_name, args.path_val,
                                                                                data_dim=setting.data_dim)
                                val_data[data_name] = (points, landmarks)

                            points, landmarks = val_data[data_name]
                            points_batch[i] = data_utils.select_point(points, setting.origin_num)
                            labels_batch[i] = landmarks

                        loss = self.sess.run(loss_op, feed_dict={
                            self.pts_fts: points_batch,
                            labels: labels_batch,
                            self.is_training: False,
                            self.sample_num_real: np.empty((setting.sample_num,))
                        })
                        avg_loss += loss

                    avg_loss /= batch_num_val
                    print('{}-[Val  ]-Average   Loss: {:.4f}'.format(datetime.now(), avg_loss))
                    sys.stdout.flush()
                    ######################################################################

                ######################################################################
                # Training
                start_idx = (setting.batch_size * batch_idx_train) % num_train
                end_idx = min(start_idx + setting.batch_size, num_train)
                batch_size_train = end_idx - start_idx
                name_batch = train_names[start_idx:end_idx]

                if end_idx == num_train:
                    random.shuffle(train_names)

                offset = int(random.gauss(0, setting.sample_num * setting.sample_num_variance))
                offset = max(offset, -setting.sample_num * setting.sample_num_clip)
                offset = min(offset, setting.sample_num * setting.sample_num_clip)
                sample_num_train = setting.sample_num + int(offset)

                points_batch = np.zeros((batch_size_train, setting.origin_num, setting.data_dim), dtype=np.float32)
                labels_batch = np.zeros((batch_size_train, setting.label_dim // 2, 2), dtype=np.float32)
                for i, data_name in enumerate(name_batch):
                    if data_name not in train_data:
                        points, landmarks = setting.data_load_func(data_name, args.path, data_dim=setting.data_dim)
                        train_data[data_name] = (points, landmarks)

                    points, landmarks = train_data[data_name]
                    points_batch[i] = data_utils.select_point(points, setting.origin_num)
                    labels_batch[i] = landmarks

                _, loss = self.sess.run([train_op, loss_op],
                                        feed_dict={
                                            self.pts_fts: points_batch,
                                            labels: labels_batch,
                                            self.is_training: True,
                                            self.sample_num_real: np.empty((sample_num_train,))
                                        })
                if batch_idx_train % 10 == 0:
                    print('{}-[Train]-Iter  {:06d}  Loss: {:.4f}'
                          .format(datetime.now(), batch_idx_train, loss))
                sys.stdout.flush()
                ######################################################################
            print('{}-Done!'.format(datetime.now()))

    def test(self, points):
        points_batch = np.zeros((self.setting.batch_size, self.setting.origin_num, self.setting.data_dim),
                                dtype=np.float32)
        for i in range(self.setting.batch_size):
            points_batch[i] = data_utils.select_point(points, self.setting.origin_num)

        predicts_np = self.sess.run(self.predicts, feed_dict={
            self.pts_fts: points_batch,
            self.is_training: False,
            self.sample_num_real: np.empty((self.setting.sample_num,))
        })
        result = np.mean(predicts_np, axis=0)
        return result
