import os
import sys
from tools import data_utils
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_ckpt = '/data1/face_data/save_folder/part_box_setting_2019-09-26-15-33-55/ckpts/iter-21000'

label_func = data_utils.get_inner_marks

origin_num = 4096

sample_num = 1024

batch_size = 32

num_epochs = 1024

step_val = 500

learning_rate_base = 0.01
decay_steps = int((2000 / batch_size) * 50)
decay_rate = 0.5
learning_rate_min = 1e-6

weight_decay = 1e-5

jitter = 0.01

rotation_range = [math.pi / 36, math.pi / 36, math.pi / 36, 'g']
rotation_order = 'rxyz'

scaling_range = [0.1, 0.1, 0.1, 'g']

sample_num_variance = 1 / 8
sample_num_clip = 1 / 4

x = 4

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 16 * x, []),
                 (12, 2, 384, 32 * x, []),
                 (16, 2, 128, 64 * x, []),
                 (16, 3, 128, 128 * x, [])]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(128 * x, 0.5),
              (64 * x, 0.5)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-2

data_dim = 6
# x,y,z,l,w,h
label_dim = 4 * 6

with_X_transformation = True
sorting_method = None
with_global = True
with_normal = True
