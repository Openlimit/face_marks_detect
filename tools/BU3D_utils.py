import os
import numpy as np

from wly import mesh, plot


def compute_normal(data_path, out_path, index_path):
    data_list = os.listdir(data_path)

    for name in data_list:
        print(name)
        point_path = os.path.join(data_path, name)
        face_path = os.path.join(index_path, name[:-3] + 'index')
        out = os.path.join(out_path, name)
        points = np.loadtxt(point_path, dtype=np.float32)
        faces = np.loadtxt(face_path, dtype=np.int32)
        m = mesh.Mesh(points=points[:, :3], faces=faces)
        m.compute_normal()
        m.save_to_xyz(out, with_normal=True)


def show_normal(data_path):
    points = np.loadtxt(data_path, dtype=np.float32)
    plot.plot_vector(points[:, :3], points[:, 3:])


if __name__ == '__main__':
    # compute_normal('/data1/face_data/train', '/data1/face_data/train_normal', '/data1/BU3D_data/BU3D_index')
    show_normal('/data1/face_data/val_normal/M0025_AN01AE_F3D.xyz')
