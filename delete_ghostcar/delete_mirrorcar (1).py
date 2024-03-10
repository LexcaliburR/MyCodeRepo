'''
Author: lexcaliburr 289427380@gmail.com
Date: 2023-12-05 13:30:10
LastEditors: lexcaliburr 289427380@gmail.com
LastEditTime: 2023-12-05 15:06:06
FilePath: /MyCodeRepo/delete_ghostcar/delete_ghostcar.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os, sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from pynput import keyboard


# read all files in path

def get_files(path):
    files = os.listdir(path)
    ret_files = []
    for file in files:
        file = os.path.join(path, file)
        ret_files.append(file)
    ret_files = sorted(ret_files)
    return ret_files


# read pcd file
def read_pcd(files):
    pointclouds = []
    for file in files:
        tmp = np.fromfile(file, dtype=np.float32).reshape(-1, 5)
        tmp = tmp[:, :3]
        print(tmp.shape)
        pointclouds.append(tmp)
    return pointclouds


def read_objs(files):
    objs = []
    for file in files:
        tmp = np.load(file, allow_pickle=True)
        objs.append(tmp)

    return objs


class Visual:
    def __init__(self):
        self.viz = o3d.visualization.Visualizer()
        self.windows = self.viz.create_window(window_name="kitti")
        pcd_files = get_files('./L7/lmj/pts/')
        self.pcds = read_pcd(pcd_files)
        self.viz_pcd = []
        self.board_points = []

        obstacle_files = get_files('./L7/lmj/results/')
        self.objs = read_objs(obstacle_files)

    def init_viz(self):
        self.viz = o3d.visualization.Visualizer()
        self.windows = self.viz.create_window(window_name="kitti")
        opt = self.viz.get_render_option()
        opt.point_size = 1
        opt.background_color = np.asarray([0, 0, 0])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        self.viz.add_geometry(coordinate_frame)

        mat_box = o3d.visualization.rendering.MaterialRecord()
        # mat_box.shader = 'defaultLitTransparency'
        mat_box.shader = 'defaultLitSSR'
        mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
        mat_box.base_roughness = 0.0
        mat_box.base_reflectance = 0.0
        mat_box.base_clearcoat = 1.0
        mat_box.thickness = 1.0
        mat_box.transmission = 1.0
        mat_box.absorption_distance = 10
        mat_box.absorption_color = [0.5, 0.5, 0.5]
        self.mat_box = mat_box

        points = []
        lines = []
        for i in range(10):
            points.append([i * 10, -15, 0])
            points.append([i * 10, 15, 0])
            lines.append([i * 2, i * 2 + 1])
        colors = [[0, 1, 0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.viz.add_geometry(line_set)

    def run_once(self, raw_pcd, obj):
        self.init_viz()
        # pcd
        self.viz_pcd = []
        self.board_points = []
        self._split_pcd(raw_pcd, 5)
        pcd = o3d.open3d.geometry.PointCloud()
        pcd.points = o3d.open3d.utility.Vector3dVector(self.viz_pcd)
        pcd.paint_uniform_color([1, 1, 1])
        self.viz.add_geometry(pcd)

        board_pcd = o3d.open3d.geometry.PointCloud()
        board_pcd.points = o3d.open3d.utility.Vector3dVector(self.board_points)
        board_pcd.paint_uniform_color([1, 0, 0])
        # self.viz.add_geometry(board_pcd)

        print(f"viz points: {len(self.viz_pcd)}, board points: {len(self.board_points)}")

        # cluster
        self._cluster()


        # box
        self._process_box(obj)
        self._draw_boxes(obj)

        # line
        self._create_line_set(obj)

        # 将点云加入到窗口中
        self.viz.run()
        self.viz.destroy_window()

    def run(self):
        for raw_pcd, obj in zip(self.pcds, self.objs):
            self.run_once(raw_pcd, obj)

    def _draw_boxes(self, obj):
        boxes = obj['pred_bboxes']
        scores = obj['pred_scores']
        labels = obj['pred_labels']

        for box, score, label in zip(boxes, scores, labels):
            center = np.array([box[0] - box[3] / 2, box[1] - box[4] / 2, box[2]])
            size = np.array([box[3], box[4], box[5]])
            orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            viz_box = self._create_3d_box(center, size, orientation)
            self.viz.add_geometry(viz_box)

    def _create_3d_box(self, center, size, orientation):
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
        mesh_box.rotate(orientation, center=center)
        mesh_box.translate(center)
        mesh_box.paint_uniform_color([0.6, 0.1, 0.5])
        return mesh_box

    def _create_line_set(self, obj):
        boxes = obj['pred_bboxes']
        scores = obj['pred_scores']
        labels = obj['pred_labels']

        for box, score, label in zip(boxes, scores, labels):
            x = box[0]
            y = box[1]
            z = box[2]
            l = box[3]
            w = box[4]
            h = box[5]
            yaw = box[6]
            rotate = np.array([[np.cos(yaw), -np.sin(yaw), 0, x],
                               [np.sin(yaw), np.cos(yaw), 0, y],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
            pt1 = [x - l / 2., y - w / 2., 0, 0]
            pt2 = [x + l / 2., y - w / 2., 0, 0]
            pt3 = [x + l / 2., y + w / 2., 0, 0]
            pt4 = [x - l / 2., y + w / 2., 0, 0]

            thethas = [np.arctan(pt[1] / pt[0]) for pt in [pt1, pt2, pt3, pt4]]
            i_max = np.argmax(thethas)
            i_min = np.argmin(thethas)

            print(f"i_max: {i_max}, i_min: {i_min}, thethas: {thethas}")
            # pt_c = [x, y, z]

            # points_cord = np.array([[0, 0, 0, 0], pt1, pt2, pt3, pt4]).transpose()
            # points_cord = np.array([[0, 0, 0, 0], pt_c]).transpose()
            # points_bev = np.matmul(rotate, points_cord)
            # points_bev = points_bev.transpose()[:, :3]
            # points_bev = [[0, 0, 0], pt_c]
            points_bev = np.array([[0, 0, 0, 0], pt1, pt2, pt3, pt4])
            points_bev = points_bev[:, :3]

            # lines = [[0, 1], [0, 2], [0, 3], [0, 4]]
            lines = [[0, 1 + i_min], [0, 1 + i_max]]

            colors = [[1, 0, 0] for i in range(len(lines))]

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points_bev),
                lines=o3d.utility.Vector2iVector(lines)
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.viz.add_geometry(line_set)

    def _process_box(self, objs):
        boxes = objs['pred_bboxes']
        scores = objs['pred_scores']
        labels = objs['pred_labels']
        saved_ids = range(0, len(boxes))
        cipv_main = []
        cipv_dists = []
        min_thetha_rads = []
        max_thetha_rads = []

        for box, score, label, saved_id in zip(boxes, scores, labels, saved_ids):

            x = box[0]
            y = box[1]
            z = box[2]
            l = box[3]
            w = box[4]
            h = box[5]
            yaw = box[6]

            pt1 = [x - l / 2., y - w / 2., 0, 0]
            pt2 = [x + l / 2., y - w / 2., 0, 0]
            pt3 = [x + l / 2., y + w / 2., 0, 0]
            pt4 = [x - l / 2., y + w / 2., 0, 0]
            pts = [pt1, pt2, pt3, pt4]

            theth_rad = [np.arctan(pt[1] / pt[0]) for pt in pts]
            theth_angle = [np.rad2deg(theth) for theth in theth_rad]

            min_thetha_rads.append(min(theth_rad))
            max_thetha_rads.append(max(theth_rad))

            print(theth_angle)

            if abs(y) < 2:
                cipv_main.append((saved_id, x, min(theth_rad), max(theth_rad)))
                continue

        cipv_main = sorted(cipv_main, key=lambda x: x[1])
        print(f"main: {cipv_main}")

        delete_car = []
        if len(cipv_main) > 1:
            pre_car = cipv_main[0]
            for i in range(1, len(cipv_main)):
                cur_car = cipv_main[i]

                if cur_car[2] > pre_car[2] and cur_car[3] < pre_car[3]:
                    delete_car.append(cur_car[0])
                    continue

        print(f"delete: {delete_car}")


        saved_boxes = []
        saved_scores = []
        saved_labels = []
        for i in range(len(boxes)):
            if i not in delete_car:
                saved_boxes.append(boxes[i])
                saved_scores.append(scores[i])
                saved_labels.append(labels[i])

        objs['pred_bboxes'] = saved_boxes
        objs['pred_scores'] = saved_scores
        objs['pred_labels'] = saved_labels


    def _split_pcd(self, raw_pcd, height):
        for pt in raw_pcd:
            if pt[2] >= height and abs(pt[1]) < 15 and pt[0] < 70:
                self.board_points.append(pt)
            else:
                self.viz_pcd.append(pt)


    def _cluster(self):
        pcd = o3d.open3d.geometry.PointCloud()
        pcd.points = o3d.open3d.utility.Vector3dVector(self.board_points)

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=1.0, min_points=10, print_progress=True))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        self.viz.add_geometry(pcd)


if __name__ == '__main__':
    pcd_files = get_files('./L7/lmj/pts/')
    print(pcd_files)
    pcds = read_pcd(pcd_files)

    viz = Visual()
    viz.run()