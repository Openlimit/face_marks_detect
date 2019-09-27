import os
import sys
import random
import math
from datetime import datetime
from tools import data_utils
import numpy as np
from tools import pointfly as pf
import tensorflow as tf
from models.model import PointCNN


class PartMarksDetector(object):
    def __init__(self, setting):
        self.setting = setting
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.settings = {}
            for name in setting.part_list:
                s = type('Setting', (), setting.__dict__)()
                s.xconv_params = getattr(setting, name + '_xconv_params')
                s.fc_params = getattr(setting, name + '_fc_params')
                s.origin_num = getattr(setting, name + '_origin_num')
                s.sample_num = getattr(setting, name + '_sample_num')
                s.label_dim = getattr(setting, name + '_label_dim')
                s.seg_part = getattr(setting, name + '_seg_part')
                self.settings[name] = s

            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.pts_ftss = {}
            self.sample_num_reals = {}
            for name in setting.part_list:
                self.pts_ftss[name] = tf.placeholder(tf.float32, shape=(None, None, setting.data_dim),
                                                     name='{}_pts_fts'.format(name))
                self.sample_num_reals[name] = tf.placeholder(tf.int32, shape=(None,),
                                                             name='{}_sample_num_real'.format(name))

            self.predictss = {}
            for name in setting.part_list:
                features_sampled = None
                if setting.data_dim > 3:
                    points_sampled, features_sampled = tf.split(self.pts_ftss[name],
                                                                [3, setting.data_dim - 3],
                                                                axis=-1,
                                                                name=name + '_split_points_features')
                else:
                    points_sampled = self.pts_ftss[name]

                # 重采样
                indices = pf.farthest_point_sample_x(points_sampled, self.sample_num_reals[name])
                points_sampled_fps = tf.gather_nd(points_sampled, indices)
                features_sampled_fps = tf.gather_nd(features_sampled, indices) if features_sampled is not None else None

                net = PointCNN(points_sampled_fps, features_sampled_fps, self.is_training, self.settings[name],
                               scope=name, out_shape=(self.settings[name].label_dim // 3, 3))
                self.predictss[name] = net.predicts

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
        part_list_size = len(setting.part_list)

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
            labelss = {}
            loss_ops = {}
            for name in setting.part_list:
                labelss[name] = tf.placeholder(tf.float32, shape=(None, self.settings[name].label_dim // 3, 3),
                                               name='{}_labels'.format(name))
                loss_ops[name] = tf.reduce_mean(
                    tf.reduce_sum(tf.square(tf.subtract(labelss[name], self.predictss[name])), axis=-1))

            lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                                   setting.decay_rate, staircase=True)
            lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)

            if setting.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
            elif setting.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum,
                                                       use_nesterov=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            train_ops = {}
            with tf.control_dependencies(update_ops):
                for name in setting.part_list:
                    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss(scope=name)
                    train_ops[name] = optimizer.minimize(loss_ops[name] + reg_loss,
                                                         global_step=global_step if name == 'nose' else None)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            saver = tf.train.Saver(max_to_keep=None)
            folder_ckpt = os.path.join(root_folder, 'ckpts')
            if not os.path.exists(folder_ckpt):
                os.makedirs(folder_ckpt)
            parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
            print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))
            sys.stdout.flush()

            train_list = []
            loss_list = []
            for name in setting.part_list:
                train_list.append(train_ops[name])
                loss_list.append(loss_ops[name])

            self.sess.run(init_op)
            for batch_idx_train in range(batch_num):
                if (batch_idx_train % setting.step_val == 0 and (batch_idx_train != 0 or args.load_ckpt is not None)) \
                        or batch_idx_train == batch_num - 1:
                    ######################################################################
                    # Validation
                    filename_ckpt = os.path.join(folder_ckpt, 'iter')
                    saver.save(self.sess, filename_ckpt, global_step=global_step)
                    print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))

                    avg_loss = {}
                    for name in setting.part_list:
                        avg_loss[name] = 0

                    for batch_val_idx in range(batch_num_val):
                        start_idx = setting.batch_size * batch_val_idx
                        end_idx = min(start_idx + setting.batch_size, num_val)
                        batch_size_val = end_idx - start_idx
                        name_batch = val_names[start_idx:end_idx]

                        points_batches = {}
                        labels_batches = {}
                        for name in setting.part_list:
                            points_batches[name] = np.zeros(
                                (batch_size_val, self.settings[name].origin_num, setting.data_dim),
                                dtype=np.float32)
                            labels_batches[name] = np.zeros((batch_size_val, self.settings[name].label_dim // 3, 3),
                                                            dtype=np.float32)

                        for i, data_name in enumerate(name_batch):
                            if data_name not in val_data:
                                points, landmarks = data_utils.load_face_data(data_name, args.path_val,
                                                                              data_dim=setting.data_dim)
                                val_data[data_name] = (points, landmarks)

                            points, landmarks = val_data[data_name]
                            boxes = data_utils.get_part_box(landmarks, with_noise=False)
                            boxes = boxes.reshape(-1, 6)
                            for part in setting.part_list:
                                seg_points, seg_landmarks, _ = self.settings[part].seg_part(points, landmarks, boxes)
                                points_batches[part][i] = data_utils.select_point(seg_points,
                                                                                  self.settings[part].origin_num)
                                labels_batches[part][i] = seg_landmarks

                        feed_dict = {self.is_training: False}
                        for name in setting.part_list:
                            feed_dict[self.pts_ftss[name]] = points_batches[name]
                            feed_dict[labelss[name]] = labels_batches[name]
                            feed_dict[self.sample_num_reals[name]] = np.empty((self.settings[name].sample_num,))

                        result = self.sess.run(loss_list, feed_dict=feed_dict)
                        for i, name in enumerate(setting.part_list):
                            avg_loss[name] += result[i]

                    s = '{}-[Val]-Average Loss--'.format(datetime.now())
                    for name in setting.part_list:
                        avg_loss[name] /= batch_num_val
                        s += '   {}:{:.4f}'.format(name, avg_loss[name])
                    print(s)

                    learn_rate = self.sess.run(lr_clip_op)
                    print('learn_rate:{}'.format(learn_rate))
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

                sample_num_trains = {}
                for name in setting.part_list:
                    offset = int(random.gauss(0, self.settings[name].sample_num * setting.sample_num_variance))
                    offset = max(offset, -self.settings[name].sample_num * setting.sample_num_clip)
                    offset = min(offset, self.settings[name].sample_num * setting.sample_num_clip)
                    sample_num_trains[name] = self.settings[name].sample_num + int(offset)

                points_batches = {}
                labels_batches = {}
                for name in setting.part_list:
                    points_batches[name] = np.zeros(
                        (batch_size_train, self.settings[name].origin_num, setting.data_dim),
                        dtype=np.float32)
                    labels_batches[name] = np.zeros((batch_size_train, self.settings[name].label_dim // 3, 3),
                                                    dtype=np.float32)

                for i, data_name in enumerate(name_batch):
                    if data_name not in train_data:
                        points, landmarks = data_utils.load_face_data(data_name, args.path, data_dim=setting.data_dim)
                        train_data[data_name] = (points, landmarks)

                    points, landmarks = train_data[data_name]
                    boxes = data_utils.get_part_box(landmarks, with_noise=True)
                    boxes = boxes.reshape(-1, 6)
                    xform, _ = pf.get_xform(rotation_range=setting.rotation_range,
                                            scaling_range=setting.scaling_range,
                                            order=setting.rotation_order)

                    for part in setting.part_list:
                        seg_points, seg_landmarks, _ = self.settings[part].seg_part(points, landmarks, boxes)
                        seg_points, seg_landmarks = data_utils.face_augment(seg_points, seg_landmarks,
                                                                            xform, range=setting.jitter,
                                                                            with_normal=setting.with_normal)
                        points_batches[part][i] = data_utils.select_point(seg_points, self.settings[part].origin_num)
                        labels_batches[part][i] = seg_landmarks

                feed_dict = {self.is_training: True}
                for name in setting.part_list:
                    feed_dict[self.pts_ftss[name]] = points_batches[name]
                    feed_dict[labelss[name]] = labels_batches[name]
                    feed_dict[self.sample_num_reals[name]] = np.empty((sample_num_trains[name],))

                result = self.sess.run([*train_list, *loss_list], feed_dict=feed_dict)
                loss_val = result[part_list_size:]
                if batch_idx_train % 10 == 0:
                    s = '{}-[Train]-Iter  {:06d}--'.format(datetime.now(), batch_idx_train)
                    for i, name in enumerate(setting.part_list):
                        s += '   {}:{:.4f}'.format(name, loss_val[i])
                    print(s)
                sys.stdout.flush()
                ######################################################################
            print('{}-Done!'.format(datetime.now()))

    def test(self, points, boxes):
        points_batches = {}
        parts = {}
        centroids = {}
        for name in self.setting.part_list:
            points_batches[name] = np.zeros((32, self.settings[name].origin_num, self.setting.data_dim),
                                            dtype=np.float32)
            parts[name], _, centroids[name] = self.settings[name].seg_part(points, np.zeros((66, 3)),
                                                                                         boxes)
            for i in range(32):
                points_batches[name][i] = data_utils.select_point(parts[name],
                                                                  self.settings[name].origin_num)

        feed_dict = {self.is_training: False}
        predict_list = []
        for name in self.setting.part_list:
            feed_dict[self.pts_ftss[name]] = points_batches[name]
            feed_dict[self.sample_num_reals[name]] = np.empty((self.settings[name].sample_num,))
            predict_list.append(self.predictss[name])

        result = self.sess.run(predict_list, feed_dict=feed_dict)
        predict = {}
        for i, name in enumerate(self.setting.part_list):
            predict[name] = np.mean(result[i], axis=0)

        return predict, parts, centroids
