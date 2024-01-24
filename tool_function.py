import numpy as np
import scipy.ndimage as ndi
import math
from skimage import measure, color, draw
import matplotlib.pyplot as plt
import cv2


# 获得shape_error, input为measure.regionprops格式，yy为行数，xx为列数
def get_shape_error(input, yy, xx):
    boundary = np.zeros((yy, xx))
    final_area = input.coords
    for i in range(final_area.shape[0]):
        boundary[final_area[i][0], final_area[i][1]] = 1  # 单独构造出一个涡旋范围

    # 构造出拟合圆
    radius = math.sqrt(input.area/math.pi)
    [rr, cc] = draw.ellipse(input.centroid[0], input.centroid[1], radius, radius)

    boundary_circle = np.zeros((yy, xx))
    for i in range(cc.shape[0]):
        if 0 <= rr[i] < yy and 0 <= cc[i] < xx:
            boundary_circle[rr[i], cc[i]] = 1  # 单独构造出一个涡旋范围

    # 找出圆边界的像素点数
    boundary_temp = np.zeros((yy, xx))
    [rr_temp, cc_temp] = draw.ellipse(boundary_circle.shape[0]/2, boundary_circle.shape[1]/2, radius, radius)
    for j in range(rr_temp.shape[0]):
        boundary_temp[rr_temp[j], cc_temp[j]] = 1  # 单独构造出一个涡旋范围

    boundary_temp = np.array(boundary_temp, np.uint8)
    contours_temp, hierarchy_temp = cv2.findContours(boundary_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    circle_edge = []
    for x_judge in range(0, boundary_temp.shape[1]):
        for y_judge in range(0, boundary_temp.shape[0]):
            # 函数判断点是否在轮廓内，如果在则返回1，否则不在返回-1，在轮廓上则返回0
            flag_temp = cv2.pointPolygonTest(contours_temp[0], (x_judge, y_judge), False)
            if flag_temp == 0:
                circle_edge.append(x_judge)

    x_cross = []
    y_cross = []
    contour_edge = []

    boundary = np.array(boundary, np.uint8)
    boundary_circle = np.array(boundary_circle, np.uint8)
    contours, hierarchy = cv2.findContours(boundary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_circle, hierarchy_circle = cv2.findContours(boundary_circle, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for x_judge in range(0, boundary.shape[1]):
        for y_judge in range(0, boundary.shape[0]):
            # 函数判断点是否在轮廓内，如果在则返回1，否则不在返回-1，在轮廓上则返回0
            flag_temp = cv2.pointPolygonTest(contours[0], (x_judge, y_judge), False)
            circle_temp = cv2.pointPolygonTest(contours_circle[0], (x_judge, y_judge), False)
            # print(flag_temp+circle_temp)
            # 在两个面积交叉的地方
            if flag_temp == 1.0 and circle_temp == 1.0:
                x_cross.append(x_judge)
                y_cross.append(y_judge)
            elif flag_temp == 0:
                contour_edge.append(x_judge)
            # elif circle_temp == 0:
            #     circle_edge.append(y_judge)

    inter_contour = input.area - len(contour_edge)
    inter_circle = input.area - len(circle_edge)
    dark_shed = inter_contour - len(x_cross)
    light_shed = inter_circle - len(x_cross)

    shape_error = (dark_shed + light_shed) / (inter_circle + 0.0001)      # 增加0.0001防止计算出现0/0的情况
    # print('shape_error:', shape_error)
    return shape_error


# 把不满足涡旋条件的涡旋作为下一次循环的输入
def iteration_add(input_boundary, input_iteration):
    for index, element in np.ndenumerate(input_boundary):
        i = index[0]
        j = index[1]
        if element == 1:
            input_iteration[i, j] = 1


# 涡旋封闭性保证，对于空心的形状，将其填充
def eddy_closure(result, sla, connectivity_choose, is_cyc=True, fill_val=-1):
    if is_cyc:
        eddy_result = result + 2
    else:
        eddy_result = result
    eddy_result[np.isnan(eddy_result)] = 0
    eddy_result = eddy_result.astype(np.int)

    labels_add = measure.label(eddy_result, connectivity=connectivity_choose)
    # print('regions number:', labels_add.max())  # 显示连通区域块数(从0开始标记)
    boundary_add = np.zeros((sla.shape[0], sla.shape[1]))
    x_label = []
    y_label = []
    x_edge = []
    y_edge = []
    for region_add in measure.regionprops(labels_add):
        area_add = region_add.coords
        for i in range(area_add.shape[0]):
            boundary_add[area_add[i][0], area_add[i][1]] = 1  # 单独构造出一个涡旋范围

        boundary_add = np.array(boundary_add, np.uint8)
        # cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1
        contours_add, hierarchy_add = cv2.findContours(boundary_add, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        boundary_add = np.array(boundary_add, np.float)

        for x_add in range(0, sla.shape[1]):
            for y_add in range(0, sla.shape[0]):
                # 函数判断点是否在轮廓内，如果在则返回1，否则不在返回-1，在轮廓上则返回0
                flag_temp = cv2.pointPolygonTest(contours_add[0], (x_add, y_add), False)
                if flag_temp == 1.0:
                    x_label.append(x_add)
                    y_label.append(y_add)
                elif flag_temp == 0:
                    x_edge.append(x_add)
                    y_edge.append(y_add)

    for _temp_ in range(0, len(x_edge)):
        boundary_add[y_edge[_temp_]][x_edge[_temp_]] = fill_val
    for _temp_ in range(0, len(x_label)):
        boundary_add[y_label[_temp_]][x_label[_temp_]] = fill_val

    boundary_add[boundary_add == 0] = np.nan
    return boundary_add, labels_add.max()


# 主函数，输入分别为二值化后的子图、得到该子图的原始像素图、选择的连通方式
def CCL_cyc(input, sla, connectivity_choose, shape_error=0.55, amp_thresh=1, d_h=0.1):
    eddy_result = np.full((input.shape[0], input.shape[1]), np.nan)  # 用来记录结果
    sla_cyc = input
    times = 1  # 用来计算迭代次数
    while True:
        sla_cyc[np.isnan(sla_cyc)] = 0
        sla_cyc = sla_cyc.astype(np.int)  # float64 --> int64

        iteration = np.full((sla.shape[0], sla.shape[1]), np.nan)  # 用来记录每次迭代的结果
        sla_it = np.full((sla.shape[0], sla.shape[1]), np.nan)  # 用来作为SLA的输入

        labels = measure.label(sla_cyc, connectivity=connectivity_choose)  # 4连通区域标记
        # area = measure.mesh_surface_area(data)
        # dst = color.label2rgb(labels)  # 根据不同的标记显示不同的颜色
        # dst = morphology.remove_small_objects(sla_cyc, min_size=8, connectivity=1)
        # print('regions number:', labels.max() + 1)  # 显示连通区域块数(从0开始标记)

        count = 0
        count_it = 0
        for region in measure.regionprops(labels):
            # 条件1：大于提前给定的像素点个数，半径不能弱于0.4度，大于4.5度----------------------------------------------
            boundary = np.zeros((sla.shape[0], sla.shape[1]))

            final_area = region.coords
            for i in range(final_area.shape[0]):
                boundary[final_area[i][0], final_area[i][1]] = 1  # 单独构造出一个涡旋范围

            # s_min = math.pi * (0.4 * resolution) ** 2 - 1
            # s_max = math.pi * (4.5 * resolution) ** 2 - 1
            s_min = 8
            s_max = 1000

            if s_min <= region.area <= s_max:
                count += 1
                # print('第%d个满足条件的轮廓：', count)
                # print('像素点个数为：', region.area)
                # print('外接框：', region.bbox)
                # box = region.bbox
                # square = max(box[2] - box[0], box[3] - box[1])
                # 条件2：形状测试结果小于55%--------------------------------------------------
                if get_shape_error(region, sla.shape[0], sla.shape[1]) <= shape_error:

                    # 此时boundary为仅包含一个轮廓的二值图像，对条件的判断可以只针对该boundary
                    boundary = np.array(boundary, np.uint8)
                    # cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1
                    contours, hierarchy = cv2.findContours(boundary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    x_label = []
                    y_label = []
                    x_edge = []
                    y_edge = []
                    for x_judge in range(0, sla.shape[1]):
                        for y_judge in range(0, sla.shape[0]):
                            # 函数判断点是否在轮廓内，如果在则返回1，否则不在返回-1，在轮廓上则返回0
                            flag_temp = cv2.pointPolygonTest(contours[0], (x_judge, y_judge), False)
                            if flag_temp == 1.0:
                                x_label.append(x_judge)
                                y_label.append(y_judge)
                            elif flag_temp == 0:
                                x_edge.append(x_judge)
                                y_edge.append(y_judge)

                    in_point_sla = []  # 记录轮廓内部的点
                    edge_point_sla = []  # 记录轮廓边缘的点

                    for _temp_ in range(0, len(x_edge)):
                        edge_point_sla.append(sla[y_edge[_temp_], x_edge[_temp_]])

                    for _temp_ in range(0, len(x_label)):
                        in_point_sla.append(sla[y_label[_temp_], x_label[_temp_]])

                    # 条件3：振幅大于1cm，振幅为极值点-边缘点的均值--------------------------------------------------
                    if len(in_point_sla) is not 0:
                        if math.fabs(min(in_point_sla)) - math.fabs(np.mean(edge_point_sla)) >= amp_thresh\
                                and max(in_point_sla) < np.mean(edge_point_sla):
                            # print("这一步是否执行")
                            for index_2, element_2 in np.ndenumerate(boundary):
                                i_2 = index_2[0]
                                j_2 = index_2[1]
                                if element_2 == 1:
                                    eddy_result[i_2, j_2] = -1
                        else:
                            count_it += 1
                            # print('不满足条件1')
                            iteration_add(boundary, iteration)

                else:
                    count_it += 1
                    # print('不满足条件2')
                    iteration_add(boundary, iteration)

            elif region.area > s_max:
                count_it += 1
                # print('不满足条件3')
                iteration_add(boundary, iteration)

        # if times == 1 or times == 5 or times == 10 or times == 15:
        #     plt.imshow(eddy_result, cmap='viridis')
        #     plt.colorbar(extend='both', fraction=0.042, pad=0.04)
        #     # plt.clim(-0.25,0.25)
        #     plt.axis('on')
        #     plt.title('T=%d' %times, fontsize=15)
        #     plt.show()

        # if times == 100:
        #     break

        if count == 0 and count_it == 0:
            # print("循环停止")
            break

        else:
            for index_it, element_it in np.ndenumerate(iteration):
                i_it = index_it[0]
                j_it = index_it[1]
                if element_it == 1:
                    sla_it[i_it, j_it] = sla[i_it, j_it]

            sla_cyc = np.full((sla.shape[0], sla.shape[1]), np.nan)
            for index_final, element_final in np.ndenumerate(sla_it):
                i_final = index_final[0]
                j_final = index_final[1]
                if element_final < - d_h * times:   # 以1cm为单位进行增减遍历
                    sla_cyc[i_final, j_final] = 1  # 转化为二值图像

            times += 1

    return eddy_closure(eddy_result, sla, connectivity_choose, True, fill_val=-1)


def CCL_anti(input, sla, connectivity_choose, shape_error=0.55, amp_thresh=1, d_h=0.1):
    eddy_result = np.full((input.shape[0], input.shape[1]), np.nan)  # 用来记录结果
    sla_anti = input
    times = 1  # 用来计算迭代次数
    while True:
        sla_anti[np.isnan(sla_anti)] = 0
        sla_anti = sla_anti.astype(np.int)  # float64 --> int64

        iteration = np.full((sla.shape[0], sla.shape[1]), np.nan)  # 用来记录每次迭代的结果
        sla_it = np.full((sla.shape[0], sla.shape[1]), np.nan)  # 用来作为SLA的输入

        labels = measure.label(sla_anti, connectivity = connectivity_choose)  # 4连通区域标记
        # area = measure.mesh_surface_area(data)
        # dst = color.label2rgb(labels)  # 根据不同的标记显示不同的颜色
        # dst = morphology.remove_small_objects(sla_cyc, min_size=8, connectivity=1)
        # print('regions number:', labels.max() + 1)  # 显示连通区域块数(从0开始标记)

        count = 0
        count_it = 0
        for region in measure.regionprops(labels):
            # 条件1：大于提前给定的像素点个数，半径不能弱于0.4度，大于4.5度----------------------------------------------
            boundary = np.zeros((sla.shape[0], sla.shape[1]))

            final_area = region.coords
            for i in range(final_area.shape[0]):
                boundary[final_area[i][0], final_area[i][1]] = 1  # 单独构造出一个涡旋范围

            # s_min = math.pi * (0.4 * resolution) ** 2 - 1
            # s_max = math.pi * (4.5 * resolution) ** 2 - 1
            s_min = 8
            s_max = 1000

            if s_min <= region.area <= s_max:
                count += 1
                # print('第%d个满足条件的轮廓：', count)
                # print('像素点个数为：', region.area)
                # print('外接框：', region.bbox)
                # box = region.bbox
                # square = max(box[2] - box[0], box[3] - box[1])
                # 条件2：形状测试结果小于55%--------------------------------------------------
                if get_shape_error(region, sla.shape[0], sla.shape[1]) <= shape_error:

                    # 此时boundary为仅包含一个轮廓的二值图像，对条件的判断可以只针对该boundary
                    boundary = np.array(boundary, np.uint8)
                    # cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1
                    contours, hierarchy = cv2.findContours(boundary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    x_label = []
                    y_label = []
                    x_edge = []
                    y_edge = []
                    for x_judge in range(0, sla.shape[1]):
                        for y_judge in range(0, sla.shape[0]):
                            # 函数判断点是否在轮廓内，如果在则返回1，否则不在返回-1，在轮廓上则返回0
                            flag_temp = cv2.pointPolygonTest(contours[0], (x_judge, y_judge), False)
                            if flag_temp == 1.0:
                                x_label.append(x_judge)
                                y_label.append(y_judge)
                            elif flag_temp == 0:
                                x_edge.append(x_judge)
                                y_edge.append(y_judge)

                    in_point_sla = []  # 记录轮廓内部的点
                    edge_point_sla = []  # 记录轮廓边缘的点

                    for _temp_ in range(0, len(x_edge)):
                        edge_point_sla.append(sla[y_edge[_temp_], x_edge[_temp_]])

                    for _temp_ in range(0, len(x_label)):
                        in_point_sla.append(sla[y_label[_temp_], x_label[_temp_]])

                    # 条件3：振幅大于1cm，振幅为极值点-边缘点的均值--------------------------------------------------
                    if len(in_point_sla) is not 0:
                        if math.fabs(max(in_point_sla)) - math.fabs(np.mean(edge_point_sla)) >= amp_thresh\
                                and min(in_point_sla) > np.mean(edge_point_sla):
                            # print("这一步是否执行")
                            for index_2, element_2 in np.ndenumerate(boundary):
                                i_2 = index_2[0]
                                j_2 = index_2[1]
                                if element_2 == 1:
                                    eddy_result[i_2, j_2] = 1
                        else:
                            count_it += 1
                            # print('不满足条件3')
                            iteration_add(boundary, iteration)

                else:
                    count_it += 1
                    # print('不满足条件2')
                    iteration_add(boundary, iteration)

            elif region.area > s_max:
                count_it += 1
                # print('不满足条件1')
                iteration_add(boundary, iteration)

        # if times == 49:
        #     print("循环停止")
        #     plt.close()
        #     plt.imshow(iteration, cmap='viridis')
        #     plt.colorbar(extend='both', fraction=0.042, pad=0.04)
        #     # plt.clim(-0.25,0.25)
        #     plt.axis('on')
        #     plt.title('Result of CNAS', fontsize=24)
        #     plt.show()
        #     break
        # if times == 100:
        #     print("循环停止")
        #     break

        if count == 0 and count_it == 0:
            # print("循环停止")
            break

        else:
            for index_it, element_it in np.ndenumerate(iteration):
                i_it = index_it[0]
                j_it = index_it[1]
                if element_it == 1:
                    sla_it[i_it, j_it] = sla[i_it, j_it]

            sla_anti = np.full((sla.shape[0], sla.shape[1]), np.nan)
            for index_final, element_final in np.ndenumerate(sla_it):
                i_final = index_final[0]
                j_final = index_final[1]
                if element_final > d_h * times:   # 以1cm为单位进行增减遍历
                    sla_anti[i_final, j_final] = 1  # 转化为二值图像

            times += 1

    return eddy_closure(eddy_result, sla, connectivity_choose, False, fill_val=1)


# 找出图像上的局部极值点
def find_extra(sla):
    extra_point = np.full((sla.shape[0], sla.shape[1]), np.nan)
    for index, element in np.ndenumerate(sla):
        i = index[0]
        j = index[1]
        if i > 1 and j > 1 and i < (sla.shape[0]-2) and j < (sla.shape[1]-2):
            # if element < sla[i, j + 1] and element < sla[i + 1, j] and \
            #    element < sla[i, j - 1] and element < sla[i - 1, j] and \
            #    element < sla[i + 1, j + 1] and element < sla[i - 1, j - 1] and \
            #    element < sla[i + 1, j - 1] and element < sla[i - 1, j + 1]:
            minimum = []
            for min_i in range(i-2,i+3):
                for min_j in range(j-2,j+3):
                    if min_i != i or min_j != j:
                        if element < sla[min_i, min_j]:
                            minimum.append(True)
                        else:
                            minimum.append(False)
            if False not in minimum:
                    extra_point[i, j] = -1    # 极小值点标记为 -1
            # if element > sla[i, j + 1] and element > sla[i + 1, j] and \
            #    element > sla[i, j - 1] and element > sla[i - 1, j] and \
            #    element > sla[i + 1, j + 1] and element > sla[i - 1, j - 1] and \
            #    element > sla[i + 1, j - 1] and element > sla[i - 1, j + 1]:
            maximum = []
            for max_i in range(i-2, i+3):
                for max_j in range(j-2, j+3):
                    if max_i != i or max_j != j:
                        if element > sla[max_i, max_j]:
                            maximum.append(True)
                        else:
                            maximum.append(False)
            if False not in maximum:
                     extra_point[i, j] = 1    # 极大值点标记为 1

    return  extra_point