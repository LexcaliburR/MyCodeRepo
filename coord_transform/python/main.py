import numpy as np
import transfroms



# 坐标系变换测试
# B1坐标系坐标，[1, -2, -1]T
# B1到B2的过渡矩阵 T_B1_B2 = [[1, 0, 1, 0],[-1, 1, 1, 0],[1, -2, -2, 0], [0, 0, 0, 1]]
# =>B2坐标系坐标，[5, 7, -4]T
def test_transforms():
    point = np.array([1, -2, -1])
    # 坐标系往x正向平移3，往y正向平移2, 往z负向平移3
    trans_matrix = np.array([[1, 0, 1, 0],[-1, 1, 1, 0],[1, -2, -2, 0], [0, 0, 0, 1]])
    result = transfroms.TransformXYZ(point, trans_matrix)
    print(f"Result:\n{result}")



def test_transforms2():
    point = np.array([12.3, 4, 1])
    # 坐标系往x正向平移3，往y正向平移2, 往z负向平移3
    trans_matrix = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0, 100],[np.sin(np.pi / 4), np.cos(np.pi / 4), 0, 200],[0, 0, 1, 0], [0, 0, 0, 1]])
    result = transfroms.TransformXYZ(point, trans_matrix)
    print(f"Result:\n{result}")

# test_transforms()

test_transforms2()
# 从自车坐标系到世界坐标系的坐标变换的举例
