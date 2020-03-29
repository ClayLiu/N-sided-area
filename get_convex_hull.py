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
    ''' Graham 算法实现，已首尾相连 '''
    
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
        
        # 如果在右边，栈顶元素出栈
        while len(stack) >= 2:

            current_vector = stack[-1] - stack[-2]
            cross_prouct = np.cross(current_vector, points_set[i] - stack[-2])

            if cross_prouct < 0:
                stack.pop()
            else:
                break

        stack.append(points_set[i])
    
    stack = deal_stack(stack)
    return np.array(stack)


def deal_stack(stack : list) -> list:
    # 首尾连接处理
    # 重新用该算法遍历一遍栈，用 i + 1 记录头部要去除点的个数
    i = 0
    length = len(stack)
    expend = 0

    while i < length:
        
        while len(stack) >= 2:
            current_vector = stack[-1] - stack[-2]
            cross_prouct = np.cross(current_vector, stack[i] - stack[-2])
            
            if cross_prouct < 0:
                stack.pop()
                expend -= 1
                if expend < 0: # 增长为负，说明要把尾部pop出，栈长度length - 1
                    length -= 1 
            else:
                break
        
        stack.append(stack[i])
        expend += 1
        i += 1
    
    return stack[length - 1:]   # 选取原栈长度后面的

if __name__ == '__main__':
    
    from points_set_set import *

    # test_points_set = test_points_set_5
    test_points_set = random_points_set(30)
    print(test_points_set)
    base = np.sum(test_points_set, axis = 0) / len(test_points_set)
    
    convex_list = convex_hull(test_points_set.copy())
    convex = np.array(convex_list)
    # print(convex)

    plt.scatter(base[0], base[1], c = 'green')
    plt.scatter(test_points_set[:, 0], test_points_set[:, 1])
    plt.scatter(convex[0, 0], convex[0, 1], c = 'yellow')
    plt.plot(convex[:, 0], convex[:, 1])

    plt.show()
