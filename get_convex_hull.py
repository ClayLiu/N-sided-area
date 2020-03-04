import math
import numpy as np 
import matplotlib.pyplot as plt

y_index = 1

def __o_distance__(vector : np.ndarray) -> float:
    return np.sqrt(np.sum(vector ** 2))

def __get_vector_cos_sita__(v_1 : np.ndarray, v_2 : np.ndarray) -> float:
    dot_product = np.sum(v_1 * v_2)
    
    l_1 = __o_distance__(v_1)
    l_2 = __o_distance__(v_2)

    return dot_product / (l_1 * l_2)

def __get_vector_sita__(v_1 : np.ndarray, v_2 : np.ndarray) -> float:
    cos_sita = __get_vector_cos_sita__(v_1, v_2)
    sita = math.acos(cos_sita)
    return sita

def convex_hull(points_set : np.ndarray) -> np.ndarray:
    ''' Graham 算法实现 '''
    
    points_count = len(points_set)
    
    ''' 先对点集角排序 '''
    base = np.sum(points_set, axis = 0) / points_count  # 选取均值点为基准点
    points_set -= base                                  # 将基准点移至原点
    
    v_base = np.array([1.0, 0.0])       # 基准向量
    sita_array = np.zeros(points_count) # 夹角数组
    
    # 获取夹角值
    for i, point in enumerate(points_set):
        sita = __get_vector_sita__(v_base, point)
        
        # 在 x 轴下方
        if point[y_index] < 0:
            sita = 2 * math.pi - sita
        sita_array[i] = sita

    # 以夹角值排序
    points_set = points_set[sita_array.argsort()]

    # 把基准点移回原位置
    points_set += base
    
    ''' 开始遍历 '''
    stack = [points_set[0], points_set[1]]
    for i in range(2, points_count):
        current_vector = stack[-1] - stack[-2]
        cross_prouct = np.cross(current_vector, points_set[i] - stack[-2])
        
        # 如果在右边，栈顶元素出栈
        if cross_prouct < 0:
            stack.pop()

        stack.append(points_set[i])

    current_vector = stack[-1] - stack[-2]
    cross_prouct = np.cross(current_vector, points_set[0] - stack[-2])

    # 如果在右边，栈顶元素出栈
    if cross_prouct < 0:
        stack.pop()

    # stack.append(points_set[0])

    return np.array(stack)

def remove_in_point(points_list : np.ndarray):
    ''' 去除在包内的点 '''

    mark = [True] * len(points_list)
    for i in range(-1, len(points_list) - 1):
        current_vector = points_list[i + 1] - points_list[i - 1]
        temp_vector = points_list[i] - points_list[i - 1]
        cross_product = np.cross(current_vector, temp_vector)
        
        # 为左，因为是逆时针绕圈，所以在前进方向的左边说明在内
        if cross_product >= 0:
            mark[i] = False

    return points_list[mark]

def true_convex_hull(points_set : np.ndarray) -> np.ndarray:
    true_convex = convex_hull(points_set)    # 获得初解
    points_count = len(true_convex)
    
    true_convex = remove_in_point(true_convex)
    points_count_after = len(true_convex)
    
    # 去除所有向内凹的边界点
    while points_count > points_count_after:
        true_convex = remove_in_point(true_convex)
        points_count = points_count_after
        points_count_after = len(true_convex)
        
    return true_convex

if __name__ == '__main__':
    
    temp = np.ones((30, 2))
    test_points_set = np.random.normal(10 * temp, 0.2)
    base = np.sum(test_points_set, axis = 0) / 30
    
    convex = convex_hull(test_points_set.copy())
    true_convex = true_convex_hull(test_points_set.copy())

    # 把头部包到尾处，以画图首尾相连
    convex = np.vstack((convex, convex[0]))
    true_convex = np.vstack((true_convex, true_convex[0]))

    plt.scatter(base[0], base[1], c = 'green')
    plt.scatter(test_points_set[:, 0], test_points_set[:, 1])
    plt.scatter(convex[0][0], convex[0][1], c = 'yellow')
    plt.plot(true_convex[:, 0], true_convex[:, 1], c='red')
    plt.plot(convex[:, 0], convex[:, 1])

    plt.show()
