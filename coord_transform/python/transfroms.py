'''
Author: lexcaliburr 289427380@gmail.com
Date: 2024-03-10 17:35:09
LastEditors: lexcaliburr 289427380@gmail.com
LastEditTime: 2024-03-10 18:02:03
FilePath: /MyCodeRepo/coord_transform/python/transfroms.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
from pyquaternion import Quaternion



# 坐标系变换或者基变换时使用
# point: 3x1 vector, [x, y, z], 为点在原始坐标系下坐标
# trans_matrix: 4x4 matrix
def TransformXYZ(point, trans_matrix):
    new_point = np.array([point[0], point[1], point[2], 0])
    print(f"new point {new_point}")
    inv = np.linalg.inv(trans_matrix)
    print(f"inv {inv}")
    ret_point = np.matmul(inv, new_point)
    return ret_point



def TransformPosition(point, trans_matrix):
    pass


def TransformVelo(velo, trans_matrix):
    pass


def TransformVariants(variants, trans_matrix):
    pass


def GetRT(quat, trans):
    RT = np.eye(4)
    RT[:3, :3] = Quaternion(quat).rotation_matrix
    RT[:3,3] = np.array(trans)
    return RT


# 将egopose转换为变换矩阵
def GetTransformMatrix(egopose):
    quat = egopose[:4]
    