import open3d as o3d
import numpy as np
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

def get_files(path):
    files = os.listdir(path)
    ret_files = []
    for file in files:
        file = os.path.join(path, file)
        ret_files.append(file)
    return ret_files


def read_pcd(files):
    pcds = []
    for file in files:
        tmp = np.fromfile(file, dtype=np.float32).reshape(-1, 5)
        tmp = tmp[:, :3]
        tmp_viz = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tmp))
        tmp_viz.paint_uniform_color([0, 1, 0])  # 设置点云颜色为红色

        # o3d.visualization.draw_geometries([tmp_viz])
        print(tmp.shape)
        pcds.append(tmp)

    return pcds
def main():
    pcd_files = get_files('./L7/lmj/pts/')
    pcds = read_pcd(pcd_files)
    raw_point = pcds[0]  # 读取1.npy数据  N*[x,y,z]

    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="kitti")
    # 设置点云大小
    vis.get_render_option().point_size = 1
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    # 创建点云对象
    pcd = o3d.open3d.geometry.PointCloud()
    # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)
    # 设置点的颜色为白色
    pcd.paint_uniform_color([1, 1, 1])
    # 将点云加入到窗口中
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()