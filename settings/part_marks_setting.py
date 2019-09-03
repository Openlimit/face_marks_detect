import os
import sys
from tools import data_utils
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_ckpt = None

label_func = data_utils.get_part_marks

loss = 'MSE'

part_list = ['nose', 'eye', 'eyebrow', 'mouth']

nose_seg_part = data_utils.seg_nose
eye_seg_part = data_utils.seg_eye
eyebrow_seg_part = data_utils.seg_eyebrow
mouth_seg_part = data_utils.seg_mouth

nose_origin_num = 1024
nose_sample_num = 738

eye_origin_num = 512
eye_sample_num = 400

eyebrow_origin_num = 512
eyebrow_sample_num = 400

mouth_origin_num = 600
mouth_sample_num = 512

batch_size = 32

num_epochs = 1024

step_val = 500

learning_rate_base = 0.01
decay_steps = int((2000 / batch_size) * 50)
decay_rate = 0.5
learning_rate_min = 1e-6

weight_decay = 1e-5

jitter = 0.01

rotation_range = [math.pi / 18, math.pi / 18, math.pi / 18, 'g']
rotation_order = 'rxyz'

scaling_range = [0.1, 0.1, 0.1, 'g']

sample_num_variance = 1 / 8
sample_num_clip = 1 / 4

x = 3

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
nose_xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                     [(8, 1, -1, 16 * x, []),
                      (12, 2, 384, 32 * x, []),
                      (16, 2, 128, 64 * x, []),
                      (16, 3, 128, 128 * x, [])]]
eye_xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                    [(8, 1, -1, 16 * x, []),
                     (12, 2, 256, 32 * x, []),
                     (16, 2, 128, 64 * x, []),
                     (16, 3, 128, 128 * x, [])]]
eyebrow_xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                        [(8, 1, -1, 16 * x, []),
                         (12, 2, 256, 32 * x, []),
                         (16, 2, 128, 64 * x, []),
                         (16, 3, 128, 128 * x, [])]]
mouth_xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                      [(8, 1, -1, 16 * x, []),
                       (12, 2, 256, 32 * x, []),
                       (16, 2, 128, 64 * x, []),
                       (16, 3, 128, 128 * x, [])]]

fc_param_name = ('C', 'dropout_rate')
nose_fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                  [(128 * x, 0.5),
                   (64 * x, 0.5)]]
eye_fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                 [(128 * x, 0.5),
                  (64 * x, 0.5)]]
eyebrow_fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                     [(128 * x, 0.5),
                      (64 * x, 0.5)]]
mouth_fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                   [(128 * x, 0.5),
                    (64 * x, 0.5)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-2

data_dim = 3

nose_label_dim = 10 * 3
eye_label_dim = 8 * 3
eyebrow_label_dim = 10 * 3
mouth_label_dim = 20 * 3
label_dim = 66 * 3

with_X_transformation = True
sorting_method = None
with_global = True
select_one = True
