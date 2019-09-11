import os
import numpy as np

from wly import mesh, plot
from concurrent.futures import ThreadPoolExecutor


def compute_normal(data_path, out_path, index_path):
    pool = ThreadPoolExecutor()
    data_list = os.listdir(data_path)

    def work(name):
        print(name)
        point_path = os.path.join(data_path, name)
        face_path = os.path.join(index_path, name[:-3] + 'index')
        out = os.path.join(out_path, name)
        points = np.loadtxt(point_path, dtype=np.float32)
        faces = np.loadtxt(face_path, dtype=np.int32)
        m = mesh.Mesh(points=points[:, :3], faces=faces)
        m.compute_normal()
        m.save_to_xyz(out, with_normal=True)

    for name in data_list:
        pool.submit(work, name)


def show_normal(data_path):
    points = np.loadtxt(data_path, dtype=np.float32)
    plot.plot_vector(points[:, :3], points[:, 3:])


if __name__ == '__main__':
    compute_normal('/data1/face_data/train', '/data1/face_data/train_normal', '/data1/BU3D_data/BU3D_index')
    # show_normal('/data1/face_data/val_normal/M0025_AN01AE_F3D.xyz')

    # import shutil
    #
    # path = '/data1/face_data/uv_coord/marks'
    # dst = '/data1/face_data/uv_coord/landmarks_nose'
    # names = os.listdir(path)
    # for name in names:
    #     if not name.endswith('uvxyz'):
    #         continue
    #     print(name)
    #     file_path = os.path.join(path, name)
    #     dst_path = os.path.join(dst, name)
    #     shutil.copy(file_path, dst_path)
