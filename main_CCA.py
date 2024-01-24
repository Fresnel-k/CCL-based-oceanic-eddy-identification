#######
# 基于联通域分析的方法对输入图像进行像素分类，达到涡旋识别的目的
#######

import numpy as np
import scipy.ndimage as ndi
from skimage import measure, color, draw, morphology
import matplotlib.pyplot as plt
import cv2
import math
import time
from netCDF4 import Dataset
from tool_function import get_shape_error, CCL_cyc, CCL_anti, find_extra, eddy_closure
import scipy.io as scio

path_group = []
path_group.append('NWP.nc')
# path_group.append('SEP.nc')
# path_group.append('SAO.nc')
# eddy_num = np.zeros((732, 4))

for path_index in range(0, 1):
    nc_obj = Dataset(path_group[path_index])
    print(nc_obj)
    print('---------------------------------------')
    print(nc_obj.variables.keys())
    print('---------------------------------------')

    lat = (nc_obj.variables['latitude'][:])
    lat = np.array(lat)
    lon = (nc_obj.variables['longitude'][:])
    lon = np.array(lon)
    data = (nc_obj.variables['sla'][:])
    date_time = (nc_obj.variables['time'][:])

    for day in range(0, len(date_time)):
        time_start = time.time()
        sla = data[day, :, :] * 100
        sla = np.flip(sla, axis=0)
        # sla = np.flip(sla, axis=0) * 100  # 把单位统一为cm

        # plt.imshow(sla, cmap='viridis')
        # plt.colorbar(extend='both', fraction=0.042, pad=0.04)
        # plt.axis('on')
        # plt.title('Input image', fontsize=15)
        # plt.show()

        # extra_point = find_extra(sla)
        # plt.imshow(extra_point, cmap='viridis')
        # plt.colorbar(extend='both', fraction=0.042, pad=0.04)
        # plt.axis('on')
        # plt.title('Input image', fontsize=15)
        # plt.show()

        # 分别对CE和AE进行分类
        sla_cyc = np.full((sla.shape[0], sla.shape[1]), np.nan)
        sla_anti = np.full((sla.shape[0], sla.shape[1]), np.nan)
        cyc_add = np.full((sla.shape[0], sla.shape[1]), np.nan)
        anti_add = np.full((sla.shape[0], sla.shape[1]), np.nan)

        T_h = 0
        for index, element in np.ndenumerate(sla):
            i = index[0]
            j = index[1]
            if element < -T_h:
                sla_cyc[i, j] = 1  # 转化为二值图像
                anti_add[i, j] = element
            elif element > T_h:
                sla_anti[i, j] = 1  # 转化为二值图像
                cyc_add[i, j] = element  # 对识别完成的图像，额外需要补充部分涡旋，例如对于CE，需要补充在正SSH场上的CE

        # sla_cyc[np.isnan(sla_cyc)] = 0
        # plt.imshow(sla_cyc, cmap='viridis')
        # plt.colorbar(extend='both', fraction=0.042, pad=0.04)
        # plt.axis('on')
        # plt.title('T=%d' %T0, fontsize=15)
        # plt.show()
        #
        # sla_anti[np.isnan(sla_anti)] = 0
        # plt.imshow(sla_anti, cmap='viridis')
        # plt.colorbar(extend='both', fraction=0.042, pad=0.04)
        # plt.axis('on')
        # plt.title('T=%d' %T0, fontsize=15)
        # plt.show()

        connectivity_choose = 1  # 连通方式选择，1为4连通，2为8连通，符合涡旋的定义
        delta_h = 0.1   # 设置成0.5存在效率过低的问题

        # -----------------------------------单独求CE-------------------------------------------------
        eddy_cyc, cyc_num_1 = CCL_cyc(sla_cyc, sla, connectivity_choose, amp_thresh=3, d_h=delta_h)
        sla_cyc_add = cyc_add - np.nanmean(cyc_add)  # 求均值异常
        binary_cyc_add = np.full((sla.shape[0], sla.shape[1]), np.nan)

        for index, element in np.ndenumerate(sla_cyc_add):
            i = index[0]
            j = index[1]
            if element < 0:
                binary_cyc_add[i, j] = 1  # 转化为二值图像

        eddy_cyc_add, cyc_num_2 = CCL_cyc(binary_cyc_add, sla_cyc_add, connectivity_choose,
                                          shape_error=0.55, amp_thresh=3, d_h=delta_h)

        eddy_cyc_add[np.isnan(eddy_cyc_add)] = 0
        eddy_cyc[np.isnan(eddy_cyc)] = 0
        eddy_cyc = eddy_cyc + eddy_cyc_add
        eddy_cyc[eddy_cyc == -2] = -1
        eddy_cyc[eddy_cyc == 1] = -1
        cyc_num = cyc_num_1 + cyc_num_2

        # eddy_cyc[eddy_cyc == 0] = np.nan
        #
        # plt.cla()
        # plt.close()
        # plt.imshow(eddy_cyc, cmap='viridis')
        # plt.colorbar(extend='both', fraction=0.042, pad=0.04)
        # # plt.clim(-0.25,0.25)
        # plt.axis('on')
        # plt.title('Result of CNAS', fontsize=24)
        # plt.show()

        # -----------------------------------单独求AE---------------------------------------------
        eddy_anti, anti_num_1 = CCL_anti(sla_anti, sla, connectivity_choose, amp_thresh=3, d_h=delta_h)
        sla_anti_add = anti_add - np.nanmean(anti_add)  # 求均值异常
        binary_anti_add = np.full((sla.shape[0], sla.shape[1]), np.nan)

        for index, element in np.ndenumerate(sla_anti_add):
            i = index[0]
            j = index[1]
            if element > 0:
                binary_anti_add[i, j] = 1  # 转化为二值图像

        eddy_anti_add, anti_num_2 = CCL_anti(binary_anti_add, sla_anti_add, connectivity_choose,
                                             shape_error=0.55, amp_thresh=3, d_h=delta_h)

        eddy_anti_add[np.isnan(eddy_anti_add)] = 0
        eddy_anti[np.isnan(eddy_anti)] = 0
        eddy_anti = eddy_anti + eddy_anti_add
        anti_num = anti_num_1 + anti_num_2

        # eddy_anti[eddy_anti == -1] = 1
        # eddy_anti[eddy_anti == 2] = 1

        # -----------------------------------合并---------------------------------------------
        # eddy_anti[eddy_anti == 0] = np.nan
        time_end = time.time()
        time_c = time_end - time_start
        print("Ocean: {}, day: {}, AE_num: {}, CE_num: {}, time_cost: {} s".format(
            path_group[path_index][:-3], day+1, anti_num, cyc_num, int(time_c)))

        eddy_result = eddy_cyc + eddy_anti
        eddy_result[eddy_result == 0] = np.nan
        # eddy_num[day, path_index * 2] = anti_num
        # eddy_num[day, path_index * 2 + 1] = cyc_num

        file_name = '.\{}\{}_{}.mat'.format(
            path_group[path_index][:-3], path_group[path_index][:-3], (day+1))
        mat_name = '{}_{}'.format(path_group[path_index][:-3], (day+1))
        scio.savemat(file_name, {mat_name: eddy_result})

        # plt.close()
        # plt.imshow(eddy_result, cmap='viridis')
        # # plt.colorbar(extend='both', fraction=0.042, pad=0.04)
        # plt.clim(-1,1)
        # plt.axis('on')
        # # plt.title('Result of CNAS', fontsize=15)
        # plt.show()
        # for i, value in np.ndenumerate(eddy_result):
        #     if eddy_result[i] == -1:
        #         eddy_result[i] = 1
        #
        # eddy_result[np.isnan(eddy_result)] = 0
        # eddy_result = eddy_result.astype(np.int)  # float64 --> int64
        # contours = measure.find_contours(eddy_result, 0.1)
        #
        # #绘制轮廓
        # fig, (ax0,ax1) = plt.subplots(1,2,figsize=(8,8))
        # ax0.imshow(eddy_result,plt.cm.gray)
        # ax1.imshow(eddy_result,plt.cm.gray)
        # for n, contour in enumerate(contours):
        #     ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
        # ax1.axis('image')
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # plt.show()

